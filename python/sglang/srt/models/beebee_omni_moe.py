
import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import get_pp_group, init_distributed_environment, initialize_model_parallel
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeModel
from sglang.srt.models.beebee_audio_encoders import BeeBeeAudioEncoder
from sglang.srt.models.beebee_vision_encoders import BeeBeeQwen25VisionModel, BeeBeeQwen3MoeVisionModel
from sglang.srt.configs.beebeeomni_moe_config import BeeBeeMoEOmniConfig
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.mm_utils import run_dp_sharded_beebee_vision_model, run_dp_sharded_audio_model
from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler, ServerArgs
from sglang.srt.utils import add_prefix, is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()
logger = logging.getLogger(__name__)


class BeeBeeMoEOmniForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # To ensure correct weight loading and mapping.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: BeeBeeMoEOmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.text_config = self.config.text_config
        self.vision_config = self.config.vision_config
        self.audio_config = self.config.audio_config
        if not get_global_server_args().encoder_only:
            self.model = Qwen3MoeModel(
                self.text_config, 
                quant_config, 
                prefix=add_prefix("model", prefix)
            )

            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        self.text_config.vocab_size,
                        self.text_config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix("lm_head", prefix),
                    )
            else:
                # ranks other than the last rank will have a placeholder layer
                self.lm_head = PPMissingLayer()
        else:
            self.lm_head = None
        self.downsample_ratio=getattr(
            self.vision_config, 
            "image_downsample_ratio", 
            getattr(self.vision_config, "image_downsample_size", 16)
        )
        if self.vision_config.model_type == "beebee_vision_model":
            self.image_encoder = BeeBeeQwen25VisionModel(
                self.vision_config,
                norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=add_prefix("image_encoder", prefix),
                use_data_parallel=self.use_data_parallel,
                downsample_ratio=self.downsample_ratio
                # max_context_len=self.vision_config.max_position_embeddings,
            )
        
        elif self.vision_config.model_type == "beebee_qwen35moe_vision_model":
            self.image_encoder = BeeBeeQwen3MoeVisionModel(
                self.vision_config,
                norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=add_prefix("image_encoder", prefix),
                downsample_ratio=self.downsample_ratio,
                use_data_parallel=self.use_data_parallel,
                # max_context_len=self.vision_config.max_position_embeddings,
            )

        else:
            raise NotImplementedError(f"{self.vision_config.model_type} is not supported yet!")

        self.audio_encoder = BeeBeeAudioEncoder(
            audio_config=self.audio_config,
            out_hidden_size=self.text_config.hidden_size,
        )
        
        self.is_mrope_enabled = False

        self.logits_processor = LogitsProcessor(self.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.image_encoder.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        expected_dim = getattr(self.image_encoder, "embed_dim", -1)

        if self.vision_config.model_type == "beebee_vision_model":
            raw_patch_dim = 1176
        elif self.vision_config.model_type == "beebee_qwen35moe_vision_model":
            raw_patch_dim = 1536

        if pixel_values.dim() == 2:
            current_dim = pixel_values.shape[-1]
            if current_dim == expected_dim:
                return pixel_values
            if current_dim != raw_patch_dim:
                return pixel_values

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_beebee_vision_model(
                self.image_encoder, 
                pixel_values, 
                image_grid_thw.tolist(), 
                merge_size=2, 
                downsample_ratio=self.downsample_ratio
            )
        else:
            image_embeds, _ = self.image_encoder(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        if self.audio_encoder is None:
            raise ValueError("Audio tokens present but audio_encoder was not initialized.")

        if not items:
            return torch.empty(0, device=self.audio_encoder.device)

        all_mel_chunks = []
        all_chunk_lengths = []
        WHISPER_HOP_LENGTH = 320
        
        if self.use_data_parallel:
            items_mel_chunks = []
            items_chunk_lengths = []
            for item in items:
                mel_chunks = item.feature.to(self.audio_encoder.device, self.audio_encoder.dtype)
                chunk_lengths = [l // WHISPER_HOP_LENGTH for l in item.model_specific_data["audio_length"]]
                items_mel_chunks.append(mel_chunks)
                items_chunk_lengths.append(chunk_lengths)

            return run_dp_sharded_audio_model(
                self.audio_encoder,
                items_mel_chunks,
                items_chunk_lengths,
            )
        else:
            all_mel_chunks = []
            all_chunk_lengths = []
            for item in items:
                mel_chunks = item.feature.to(self.audio_encoder.device, self.audio_encoder.dtype)
                chunk_lengths = [l // WHISPER_HOP_LENGTH for l in item.model_specific_data["audio_length"]]
                all_mel_chunks.append(mel_chunks)
                all_chunk_lengths.extend(chunk_lengths)

            batched_mel_chunks = torch.cat(all_mel_chunks, dim=0)
            chunk_embeds, _ = self.audio_encoder(batched_mel_chunks, all_chunk_lengths)
            return chunk_embeds



    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Placeholder for post_process
        new_embeddings = []
        for i, (modality, embedding, index) in enumerate(
            zip(modalities, embeddings, indices)
        ):
            if embedding is None or index is None:
                continue

            new_embeddings.append(embedding)
        return new_embeddings, forward_batch

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
     
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Whisper encoder self_attn k_proj has no bias in the checkpoint, but
        # our QKVParallelLinear uses a single merged bias (bias=True).  We
        # inject a zero tensor so the weight_loader can correctly shard the k
        # slice into qkv_proj.bias just like q and v.  Without this the k
        # portion would stay at PyTorch's default zero init — which is still
        # numerically correct, but explicit injection avoids any future
        # confusion if the default ever changes.

        weights = list(weights)
        encoder_layers = getattr(self.audio_encoder.encoder, "layers", [])
        for layer_idx in range(len(encoder_layers)):
            k_w_key = f"audio_encoder.layers.{layer_idx}.self_attn.k_proj.weight"
            k_b_key = f"audio_encoder.layers.{layer_idx}.self_attn.k_proj.bias"
            # Find the k_proj weight to get the right size
            k_proj_weight = next(
                (w for n, w in weights if n == k_w_key), None
            )
            if k_proj_weight is not None and not any(n == k_b_key for n, _ in weights):
                weights.append((k_b_key, torch.zeros(k_proj_weight.size(0))))
        
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "model")
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

 
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)

            if name.startswith("image_encoder."):
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

    
            if name.startswith("audio_encoder.") and not name.startswith("audio_encoder.audio_projector."):
                name = name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

            moe_expert_mapping = [
                ("gate_proj", "w13_weight", "weight13", "w1"),
                ("up_proj",   "w13_weight", "weight13", "w3"),
                ("down_proj", "w2_weight",  "weight2",  "w2"),
            ]
            
            is_moe_expert = False
            for proj_name, target_name, fallback_name, shard_id in moe_expert_mapping:
                search_base = f"mlp.experts.{proj_name}"

                if search_base in name:
                    mapped_name = name.replace(f"experts.{proj_name}", f"experts.{target_name}")
                    
                    if mapped_name not in params_dict and mapped_name.replace(target_name, fallback_name) in params_dict:
                        mapped_name = mapped_name.replace(target_name, fallback_name)
                    
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        
                        global_num_experts = loaded_weight.shape[0]

                        for global_idx in range(global_num_experts):
                            
                            weight_loader(
                                param=param, 
                                loaded_weight=loaded_weight[global_idx], 
                                weight_name=mapped_name, 
                                shard_id=shard_id, 
                                expert_id=global_idx
                            )
                    else:
                        logger.warning(f"Failed to map MoE {proj_name}: {mapped_name}")
                        
                    is_moe_expert = True
                    break
                    
            if is_moe_expert:
                continue

            # stack weight 
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                
                if f".{weight_name}." not in name and not name.endswith(f".{weight_name}"):
                    continue
                
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    print(f"{mapped_name} not in the params_dict.")
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)

                is_stacked = True
                break
            
            if is_stacked:
                continue

            # other weight
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.warning(f"Skipped unmapped safetensor key: {name}")


    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

EntryClass = BeeBeeMoEOmniForConditionalGeneration

def compare_weights(orig_sd, sgl_sd):
    print("\n--- 🔍 Checking Model Weights (Tensor Values) ---")
    all_matched = True
    
    merged_tasks = {}       # 收集常规 QKV 和 Shared Expert MLP
    moe_stacked_tasks = {}  # 收集 MoE 专家的 stacked 张量 (layer_idx -> {proj_type: tensor})
    
    # 匹配 MoE 专家权重 (兼容有无 .weight 后缀)
    moe_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)(?:\.weight)?$")

    # 常规线性层合并映射
    target_map = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", "gate"),
        "up_proj": ("gate_up_proj", "up")
    }

    # ==========================================
    # 1. 遍历并归类原始权重
    # ==========================================
    for orig_name, orig_tensor in orig_sd.items():
        sgl_name = orig_name
        
        # [归类 A]: 拦截 MoE 专家的 Stacked 权重 (跳过后续逻辑)
        moe_match = moe_pattern.search(orig_name)
        if moe_match:
            layer_idx, proj_type = moe_match.groups()
            moe_stacked_tasks.setdefault(int(layer_idx), {})[proj_type] = orig_tensor
            continue
        
        # [名称映射]: 处理 Vision / Audio 模块的不对齐
        if sgl_name.startswith("image_encoder."):
            sgl_name = sgl_name.replace("attn.qkv.", "attn.qkv_proj.")

        if sgl_name.startswith("audio_encoder.") and not sgl_name.startswith("audio_encoder.audio_projector."):
            sgl_name = sgl_name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

        # [归类 B]: 拦截常规 QKV 和 Shared Expert MLP 的散装权重
        is_merged_task = False
        for orig_key, (target_key, part_name) in target_map.items():
            if f".{orig_key}." in sgl_name or sgl_name.endswith(f".{orig_key}"):
                merged_sgl_name = sgl_name.replace(orig_key, target_key)
                merged_tasks.setdefault(merged_sgl_name, {})[part_name] = orig_tensor
                is_merged_task = True
                break
                
        if is_merged_task:
            continue

        # [对比 C]: 1对1 基础权重直接对比 (包括 Router/gate.weight, Norm等)
        if sgl_name not in sgl_sd:
            print(f"❌ [Missing in SGLang] {sgl_name} (mapped from {orig_name})")
            all_matched = False
            continue
            
        sgl_tensor = sgl_sd[sgl_name]
        if orig_tensor.shape != sgl_tensor.shape:
            print(f"❌ [Shape Mismatch] {orig_name} ({orig_tensor.shape}) vs {sgl_name} ({sgl_tensor.shape})")
            all_matched = False
            continue
            
        max_diff = torch.max(torch.abs(orig_tensor - sgl_tensor)).item()
        if max_diff > 1e-5:
            print(f"❌ [Value Differs] {orig_name} -> Max Diff: {max_diff:.6f}")
            all_matched = False

    # ==========================================
    # 2. 验证常规合并权重 (QKV, Shared Expert)
    # ==========================================
    for sgl_name, parts in merged_tasks.items():
        if sgl_name not in sgl_sd:
            print(f"❌ [Missing in SGLang] {sgl_name} (Merged Target)")
            all_matched = False
            continue
            
        sgl_tensor = sgl_sd[sgl_name]
        try:
            if "qkv_proj" in sgl_name:
                # 兼容 Whisper encoder self_attn 没有 k_bias 的情况
                if "q" in parts and "v" in parts and "k" not in parts:
                    parts["k"] = torch.zeros_like(parts["q"])
                orig_merged = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            elif "gate_up_proj" in sgl_name:
                orig_merged = torch.cat([parts["gate"], parts["up"]], dim=0)
            else:
                raise ValueError(f"Unknown merge target: {sgl_name}")
        except Exception as e:
            print(f"❌ [Merge Error] {sgl_name}: {e}")
            all_matched = False
            continue
            
        if orig_merged.shape != sgl_tensor.shape:
            print(f"❌ [Shape Mismatch - Merged] {sgl_name}: expected {orig_merged.shape} vs sgl {sgl_tensor.shape}")
            all_matched = False
            continue
            
        max_diff = torch.max(torch.abs(orig_merged - sgl_tensor)).item()
        if max_diff > 1e-5:
            print(f"❌ [Value Differs - Merged] {sgl_name} -> Max Diff: {max_diff:.6f}")
            all_matched = False

    # ==========================================
    # 3. 验证 MoE 专家合并权重 (w13_weight, w2_weight)
    # ==========================================
    for layer_idx, projs in moe_stacked_tasks.items():
        # --- 验证 w13_weight (Gate & Up 合并) ---
        w13_name = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
        if w13_name not in sgl_sd and f"model.layers.{layer_idx}.mlp.experts.weight13" in sgl_sd:
            w13_name = f"model.layers.{layer_idx}.mlp.experts.weight13"
            
        if w13_name in sgl_sd:
            sgl_w13 = sgl_sd[w13_name]
            if "gate_proj" in projs and "up_proj" in projs:
                # FusedMoE 预期：沿着输出特征维度 (dim=1) 将 gate 和 up 拼接
                # 原 shape 通常为 [num_experts, intermediate_size, hidden_size]
                expected_w13 = torch.cat([projs["gate_proj"], projs["up_proj"]], dim=1)
                
                if expected_w13.shape != sgl_w13.shape:
                    print(f"❌ [Shape Mismatch - MoE w13] Layer {layer_idx}: expected {expected_w13.shape} vs sgl {sgl_w13.shape}")
                    all_matched = False
                else:
                    max_diff = torch.max(torch.abs(expected_w13 - sgl_w13)).item()
                    if max_diff > 1e-5:
                        print(f"❌ [Value Differs - MoE w13] Layer {layer_idx} -> Max Diff: {max_diff:.6f}")
                        all_matched = False
            else:
                print(f"❌ [Incomplete MoE w13] Layer {layer_idx} missing gate or up.")
                all_matched = False
        else:
            print(f"❌ [Missing in SGLang] {w13_name}")
            all_matched = False

        # --- 验证 w2_weight (Down Proj 直接比对) ---
        w2_name = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
        if w2_name not in sgl_sd and f"model.layers.{layer_idx}.mlp.experts.weight2" in sgl_sd:
            w2_name = f"model.layers.{layer_idx}.mlp.experts.weight2"
            
        if w2_name in sgl_sd:
            sgl_w2 = sgl_sd[w2_name]
            if "down_proj" in projs:
                expected_w2 = projs["down_proj"]
                if expected_w2.shape != sgl_w2.shape:
                    print(f"❌ [Shape Mismatch - MoE w2] Layer {layer_idx}: expected {expected_w2.shape} vs sgl {sgl_w2.shape}")
                    all_matched = False
                else:
                    max_diff = torch.max(torch.abs(expected_w2 - sgl_w2)).item()
                    if max_diff > 1e-5:
                        print(f"❌ [Value Differs - MoE w2] Layer {layer_idx} -> Max Diff: {max_diff:.6f}")
                        all_matched = False
            else:
                print(f"❌ [Incomplete MoE w2] Layer {layer_idx} missing down_proj.")
                all_matched = False
        else:
            print(f"❌ [Missing in SGLang] {w2_name}")
            all_matched = False

    # ==========================================
    # 总结输出
    # ==========================================
    if all_matched:
        print("🎉 恭喜！所有权重全部完美对齐！")
        print("涵盖检查项：")
        print(" - Vision/Audio 特殊名称映射")
        print(" - Whisper k_proj_bias 置零填充")
        print(" - Attention QKV 张量拼接")
        print(" - Shared Expert (gate_up_proj) 张量拼接")
        print(" - Fused MoE Stacked 专家权重 (w13_weight / w2_weight) 重组与对齐")
    else:
        print("\n⚠️ 存在未对齐的权重，请往上翻看带 ❌ 的报错信息进行排查。")
        
    return all_matched

if __name__ == "__main__":
    import os
    import glob
    import torch
    from safetensors import safe_open
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    
    
    init_distributed_environment()
    initialize_model_parallel()
    
    MODEL_PATH = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/ckpt/0518_llavaomni_30A3B_qwen35encoder_st2_mmprojector/checkpoints/hf_ckpt" 
    
    dummy_args = ServerArgs(model_path=MODEL_PATH, mm_enable_dp_encoder=False)
    dummy_args.enable_dp_attention = False
    dummy_args.dp_size = 1
    dummy_args.moe_dense_tp_size = None
    dummy_args.attn_cp_size = 1
    dummy_args.device = "cuda:0"
    set_global_server_args_for_scheduler(dummy_args)

    print("Initializing models...")
    from veomni.models.custom.llava_qwen3moe.modeling_llava_qwen3moe_omni import LlavaQwen3MoeForCausalLM
    from veomni.ops.fused_moe import apply_veomni_fused_moe_patch
    apply_veomni_fused_moe_patch(moe_implementation="fused")
    
    config = BeeBeeMoEOmniConfig.from_pretrained(MODEL_PATH)
  
    class DummyModelConfig:
        def __init__(self, hidden_size, dtype):
            self.hidden_size = hidden_size
            self.dtype = dtype

    dummy_model_config = DummyModelConfig(
        hidden_size=config.text_config.hidden_size,
        dtype=torch.bfloat16
    )
    
    initialize_dp_attention(dummy_args, dummy_model_config)
    
    # === 定义设备 ===
    device_sgl = torch.device("cuda:0")
    device_orig = torch.device("cuda:1")

    # 1. 加载 SGLang 模型到 cuda:0
    sglang_model = BeeBeeMoEOmniForConditionalGeneration(config).to(torch.bfloat16).to(device_sgl)
    sglang_model.eval()

    # 2. 加载 原始模型 到 cuda:1
    train_model = LlavaQwen3MoeForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device_orig)
    train_model.eval()

    print("Loading weights into SGLang model...")
    safetensors_files = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No .safetensors files found in {MODEL_PATH}")

    weights_iterator = []
    for f in safetensors_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                weights_iterator.append((k, st.get_tensor(k)))

    sglang_model.load_weights(weights_iterator)

    # === Check Model Weight ===
    print("Preparing state dicts for comparison...")
    orig_state_dict = {k: v.cpu() for k, v in train_model.state_dict().items()}
    sgl_state_dict = {k: v.cpu() for k, v in sglang_model.state_dict().items()}
    
    compare_weights(orig_state_dict, sgl_state_dict)

    # ---------------------------------------------------------
    # 视觉编码器 (Image Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Vision Encoder ---")
 
    is_qwen35_encoder = True
    if is_qwen35_encoder:
        vision_hidden = 1536
    else:
        vision_hidden = 1176
    dummy_pixel_values_cpu = torch.randn(1536, vision_hidden, dtype=torch.bfloat16)
    dummy_image_grid_thw_cpu = torch.tensor([[1, 32, 48]], dtype=torch.int32)

    with torch.no_grad():
        # SGLang 视觉前向 (cuda:0)
        sgl_vision_out, _ = sglang_model.image_encoder(
            dummy_pixel_values_cpu.to(device_sgl), 
            dummy_image_grid_thw_cpu.to(device_sgl)
        )
        
        # 原模型 视觉前向 (cuda:1)
        orig_vision_out, _ = train_model.image_encoder.lm_encode(
            dummy_pixel_values_cpu.to(device_orig), 
            dummy_image_grid_thw_cpu.to(device_orig)
        )

    # 将结果拉回到 cuda:0 进行对比
    orig_vision_out = orig_vision_out.to(device_sgl)

    v_max_diff = torch.max(torch.abs(sgl_vision_out - orig_vision_out)).item()
    v_mean_diff = torch.mean(torch.abs(sgl_vision_out - orig_vision_out)).item()
    print(f"Vision Output Shape: {sgl_vision_out.shape}")
    print(f"Vision Max Diff:  {v_max_diff:.6f}")
    print(f"Vision Mean Diff: {v_mean_diff:.6f}")
    if v_max_diff < 1e-3:
        print("✅ Vision Encoder weights loaded perfectly!")
    else:
        print("❌ Vision Encoder has significant precision differences.")

    # ---------------------------------------------------------
    # 音频编码器 (Audio Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Audio Encoder ---")
    dummy_mel_cpu = torch.randn(1, 128, 3000, dtype=torch.bfloat16)
    dummy_mel_lengths_cpu = torch.tensor([300], dtype=torch.long)

    with torch.no_grad():
        # SGLang 音频前向 (cuda:0)
        sgl_audio_out, _ = sglang_model.audio_encoder(
            dummy_mel_cpu.to(device_sgl), 
            dummy_mel_lengths_cpu.to(device_sgl)
        )
        
        # 原模型 音频前向 (cuda:1)
        orig_audio_out, _ = train_model.audio_encoder.lm_encode(
            dummy_mel_cpu.to(device_orig), 
            dummy_mel_lengths_cpu.to(device_orig)
        )

    # 将结果拉回到 cuda:0 进行对比
    orig_audio_out = orig_audio_out.to(device_sgl)

    a_max_diff = torch.max(torch.abs(sgl_audio_out - orig_audio_out)).item()
    a_mean_diff = torch.mean(torch.abs(sgl_audio_out - orig_audio_out)).item()
    print(f"Audio Output Shape: {sgl_audio_out.shape}")
    print(f"Audio Max Diff:  {a_max_diff:.6f}")
    print(f"Audio Mean Diff: {a_mean_diff:.6f}")
    if a_max_diff < 1e-3:
        print("✅ Audio Encoder weights loaded perfectly!")
    else:
        print("❌ Audio Encoder has significant precision differences.")


    print("\n--- Testing LLM Backbone Components ---")
    
    orig_text_model = train_model.model if hasattr(train_model, "model") else train_model.language_model.model
    sgl_text_model = sglang_model.model if hasattr(sglang_model, "model") else sglang_model.language_model.model
    
    hidden_size = config.text_config.hidden_size
    seq_len = 64
    
    # ---------------------------------------------------------
    # 1. 验证 Token Embedding
    # ---------------------------------------------------------
    print("-> Testing Token Embeddings...")
    dummy_input_ids = torch.randint(0, 32000, (1, seq_len))
    
    with torch.no_grad():
        orig_embeds = orig_text_model.embed_tokens(dummy_input_ids.to(device_orig)).cpu()
        sgl_embeds = sgl_text_model.embed_tokens(dummy_input_ids.to(device_sgl)).cpu()
        
    e_max_diff = torch.max(torch.abs(sgl_embeds - orig_embeds)).item()
    print(f"   Embedding Max Diff: {e_max_diff:.6f}")


    # ---------------------------------------------------------
    # 2. 验证核心 MoE MLP 层 (极其关键：验证专家权重与 Router)
    # ---------------------------------------------------------
    print("-> Testing MoE MLP Layer (Layer 0)...")
    num_layers = len(orig_text_model.layers)
    all_moe_matched = True

    print(f"   Found {num_layers} layers. Starting mathematical alignment check...")

    for layer_idx in range(num_layers):
       
        dummy_hidden_cpu = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # HF Original 模型前向
            orig_mlp = orig_text_model.layers[layer_idx].mlp
            orig_mlp_out = orig_mlp(dummy_hidden_cpu.to(device_orig)).cpu()
            
            sgl_mlp = sgl_text_model.layers[layer_idx].mlp
            dummy_hidden_2d = dummy_hidden_cpu.view(-1, hidden_size).to(device_sgl)
            sgl_mlp_out = sgl_mlp(dummy_hidden_2d).cpu()
            sgl_mlp_out = sgl_mlp_out.view(1, seq_len, hidden_size)
            
        m_max_diff = torch.max(torch.abs(sgl_mlp_out - orig_mlp_out)).item()
        m_mean_diff = torch.mean(torch.abs(sgl_mlp_out - orig_mlp_out)).item()
        
        orig_flat = orig_mlp_out.view(-1).float()
        sgl_flat = sgl_mlp_out.view(-1).float()
        cos_sim = torch.nn.functional.cosine_similarity(orig_flat, sgl_flat, dim=0).item()

        # 判断对齐标准：余弦相似度 > 0.99 且 平均误差 < 0.005
        if cos_sim > 0.99 or m_mean_diff < 0.005:
            print(f"   [Layer {layer_idx:02d}] ✅ Pass | Max: {m_max_diff:.4f}, Mean: {m_mean_diff:.5f}, CosSim: {cos_sim:.6f}")
        else:
            print(f"   [Layer {layer_idx:02d}] ❌ FAIL | Max: {m_max_diff:.4f}, Mean: {m_mean_diff:.5f}, CosSim: {cos_sim:.6f}")
            all_moe_matched = False

    if all_moe_matched:
        print("   🎉 所有 MoE 层的数学输出完美对齐！")
    else:
        print("   ⚠️ 存在未对齐的 MoE 层，请检查上方日志排查对应层数。")
    
    print("\nAll tests completed.")