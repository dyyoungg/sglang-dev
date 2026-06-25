
import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
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
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.beebee_audio_encoders import BeeBeeAudioEncoder
from sglang.srt.models.beebee_vision_encoders import BeeBeeQwen25VisionModel, BeeBeeQwen3MoeVisionModel
from sglang.srt.configs.beebeeomni_config import BeeBeeOmniConfig
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.mm_utils import run_dp_sharded_beebee_vision_model, run_dp_sharded_audio_model
from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler, ServerArgs
from sglang.srt.utils import add_prefix, is_cuda, is_npu


_is_cuda = is_cuda()

logger = logging.getLogger(__name__)

class BeeBeeOmniForConditionalGeneration(nn.Module):
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
        config: BeeBeeOmniConfig,
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
            self.model = Qwen2Model(
                self.text_config,
                quant_config,
                prefix=add_prefix("model", prefix),
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
        self.max_image_bs = getattr(get_global_server_args(), "max_image_bs", 16)
        self.downsample_ratio=getattr(
            self.vision_config,
            "image_downsample_ratio",
            getattr(self.vision_config, "image_downsample_size", 16)
        )
        # In language_only mode, encoders are remote — skip building them
        # to save GPU memory.
        language_only = getattr(get_global_server_args(), "language_only", False)
        if not language_only:
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
        else:
            self.image_encoder = None
            self.audio_encoder = None
        
        self.is_mrope_enabled = False

        self.logits_processor = LogitsProcessor(self.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        nvtx.range_push("get_image_feature")

        if not items:
            nvtx.range_pop()
            return torch.empty(0, device=self.image_encoder.device)

        all_image_embeds = []

        expected_dim = getattr(self.image_encoder, "embed_dim", -1)
        if self.vision_config.model_type == "beebee_vision_model":
            raw_patch_dim = 1176
        elif self.vision_config.model_type == "beebee_qwen35moe_vision_model":
            raw_patch_dim = 1536
        else:
            raw_patch_dim = -1

        # 使用 self.max_image_bs 对 items 进行切片分块推理
        for i in range(0, len(items), self.max_image_bs):
            chunk_items = items[i : i + self.max_image_bs]

            pixel_values = torch.cat([item.feature for item in chunk_items], dim=0).type(
                self.image_encoder.dtype
            )
            image_grid_thw = torch.concat([item.image_grid_thw for item in chunk_items], dim=0)

            # 提前返回逻辑
            if pixel_values.dim() == 2:
                current_dim = pixel_values.shape[-1]
                if current_dim == expected_dim or current_dim != raw_patch_dim:
                    all_image_embeds.append(pixel_values)
                    continue

            assert pixel_values.dim() == 2, pixel_values.dim()
            assert image_grid_thw.dim() == 2, image_grid_thw.dim()
            
            # 正常执行视觉模型推理
            if self.use_data_parallel:
                chunk_embeds = run_dp_sharded_beebee_vision_model(
                    self.image_encoder,
                    pixel_values,
                    image_grid_thw.tolist(),
                    merge_size=2,
                    downsample_ratio=self.downsample_ratio
                )
            else:
                chunk_embeds, _ = self.image_encoder(pixel_values, grid_thw=image_grid_thw)
                
            all_image_embeds.append(chunk_embeds)

        # 拼接所有分块的特征
        if all_image_embeds:
            final_embeds = torch.cat(all_image_embeds, dim=0)
        else:
            final_embeds = torch.empty(0, device=self.image_encoder.device, dtype=self.image_encoder.dtype)

        nvtx.range_pop()
        return final_embeds

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        nvtx.range_push("get_audio_feature")

        if self.audio_encoder is None:
            raise ValueError("Audio tokens present but audio_encoder was not initialized.")

        if not items:
            nvtx.range_pop()
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

            result = run_dp_sharded_audio_model(
                self.audio_encoder,
                items_mel_chunks,
                items_chunk_lengths,
            )
            nvtx.range_pop()
            return result
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
            nvtx.range_pop()
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
        nvtx.range_push("BeeBeeOmni.forward")

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

        nvtx.range_push("general_mm_embed_routine")
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        nvtx.range_pop()  # general_mm_embed_routine

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                nvtx.range_push("logits_processor")
                result = self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
                nvtx.range_pop()  # logits_processor
                nvtx.range_pop()  # BeeBeeOmni.forward
                return result
            else:
                result = self.pooler(hidden_states, forward_batch)
                nvtx.range_pop()  # BeeBeeOmni.forward
                return result
        else:
            nvtx.range_pop()  # BeeBeeOmni.forward
            return hidden_states


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Custom weight loader for SGLang/vLLM backend.
        Handles PP layer filtering, TP tensor sharding, and module name mapping.
        """
        # 定义需要合并 (Concat) 的参数映射 (用于张量并行 TP)
        # 格式: (代码中的合并参数名, 权重文件中的独立参数名, shard_id)
        stacked_params_mapping = [
            # LLM Attention Q/K/V 合并
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # LLM & Vision MLP Gate/Up 合并 (SwiGLU 结构)
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
        encoder_layers = getattr(self.audio_encoder, "encoder", None)
        encoder_layers = getattr(encoder_layers, "layers", []) if encoder_layers is not None else []
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

            # 5. Vision Encoder 命名映射
            if name.startswith("image_encoder."):
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

    
            if name.startswith("audio_encoder.") and not name.startswith("audio_encoder.audio_projector."):
                name = name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

            # 7. 处理分片参数合并 (Stacked Params)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                
                if f".{weight_name}." not in name and not name.endswith(f".{weight_name}"):
                    continue
                
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                    
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 传入 shard_id 告诉 SGLang 的 ParallelLinear 应该把这个切片拼到哪个位置
                weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break
            
            if is_stacked:
                continue

       
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                # In encoder_only / language_only mode, weights for the
                # missing half (LM or encoders) are absent from params_dict.
                # Silently skip them instead of warning.
                if getattr(self.config, "encoder_only", False) or getattr(
                    self.config, "language_only", False
                ):
                    continue
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

EntryClass = BeeBeeOmniForConditionalGeneration



def compare_weights(orig_sd, sgl_sd):
  
    print("\n--- 🔍 Checking Model Weights (Tensor Values) ---")
    all_matched = True
    merged_tasks = {}
    
    target_map = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", "gate"),
        "up_proj": ("gate_up_proj", "up")
    }

    for orig_name, orig_tensor in orig_sd.items():
    
        sgl_name = orig_name
        
        if sgl_name.startswith("image_encoder."):
            sgl_name = sgl_name.replace("attn.qkv.", "attn.qkv_proj.")

     
        if sgl_name.startswith("audio_encoder.") and not sgl_name.startswith("audio_encoder.audio_projector."):
            sgl_name = sgl_name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

       
        is_stacked = False
        for orig_key, (target_key, part_name) in target_map.items():
            if f".{orig_key}." in sgl_name or sgl_name.endswith(f".{orig_key}"):
                merged_sgl_name = sgl_name.replace(orig_key, target_key)
                if merged_sgl_name not in merged_tasks:
                    merged_tasks[merged_sgl_name] = {}
                merged_tasks[merged_sgl_name][part_name] = orig_tensor
                is_stacked = True
                break
                
        if is_stacked:
            continue

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

    for sgl_name, parts in merged_tasks.items():
        if sgl_name not in sgl_sd:
            print(f"❌ [Missing in SGLang] {sgl_name} (Merged Target)")
            all_matched = False
            continue
            
        sgl_tensor = sgl_sd[sgl_name]
        
        try:
            if "qkv_proj" in sgl_name:
                
                if "q" in parts and "v" in parts and "k" not in parts:
                    parts["k"] = torch.zeros_like(parts["q"])
                    
                orig_merged = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            elif "gate_up_proj" in sgl_name:
                orig_merged = torch.cat([parts["gate"], parts["up"]], dim=0)
            else:
                print(f"❌ [Unknown Merge Target] {sgl_name}")
                all_matched = False
                continue
                
        except KeyError as e:
            print(f"❌ [Incomplete Merge Parts] {sgl_name}: Missing fragment {e}")
            all_matched = False
            continue
        except RuntimeError as e:
            print(f"❌ [Concat Error] {sgl_name}: {e}")
            all_matched = False
            continue
            
        if orig_merged.shape != sgl_tensor.shape:
            print(f"❌ [Shape Mismatch - Merged] {sgl_name}: orig_merged({orig_merged.shape}) vs sgl({sgl_tensor.shape})")
            all_matched = False
            continue
            
        max_diff = torch.max(torch.abs(orig_merged - sgl_tensor)).item()
        if max_diff > 1e-5:
            print(f"❌ [Value Differs - Merged] {sgl_name} -> Max Diff: {max_diff:.6f}")
            all_matched = False

    if all_matched:
        print("🎉 所有权重全部完美对齐！(包含 Vision/Audio 的命名映射、合并 QKV 及 Whisper 的 k_bias 注零验证)")
    else:
        print("⚠️ 存在未对齐的权重，请检查上面的报错信息。")
        
    return all_matched


if __name__ == "__main__":
    import os
    import glob
    import torch
    from safetensors import safe_open
    
    init_distributed_environment()
    initialize_model_parallel()
    MODEL_PATH = "/mnt/afs/share/llava_qwen2_14B-qwen35encoder-veomni-down16" 
    
    dummy_args = ServerArgs(model_path=MODEL_PATH, mm_enable_dp_encoder=False)
    set_global_server_args_for_scheduler(dummy_args)

    print("Initializing models...")
    
    from veomni.models.custom.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM
    
    
    config = BeeBeeOmniConfig.from_pretrained(MODEL_PATH)
    
    sglang_model = BeeBeeOmniForConditionalGeneration(config).to(torch.bfloat16).cuda()
    sglang_model.eval()

    # 初始化原始模型
    train_model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()
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
    
    sglang_model = sglang_model.cuda()

    # check model weight
    orig_state_dict = train_model.state_dict()
    sgl_state_dict = sglang_model.state_dict()
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
    dummy_pixel_values = torch.randn(1536, vision_hidden, dtype=torch.bfloat16, device="cuda")
  
    dummy_image_grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        # SGLang 视觉前向
        sgl_vision_out, _ = sglang_model.image_encoder(dummy_pixel_values, dummy_image_grid_thw)
        
        orig_vision_out, _ = train_model.image_encoder.lm_encode(dummy_pixel_values, dummy_image_grid_thw)

        print(sgl_vision_out.shape, orig_vision_out.shape)


    v_max_diff = torch.max(torch.abs(sgl_vision_out - orig_vision_out)).item()
    v_mean_diff = torch.mean(torch.abs(sgl_vision_out - orig_vision_out)).item()
    print(f"Vision Output Shape: {sgl_vision_out.shape}")
    print(f"Vision Max Diff:  {v_max_diff:.6f}")
    print(f"Vision Mean Diff: {v_mean_diff:.6f}")
    if v_mean_diff < 1e-3 and v_max_diff < 1e-2:
        print("✅ Vision Encoder weights loaded perfectly!")
    else:
        print("❌ Vision Encoder has significant precision differences.")

    # ---------------------------------------------------------
    # 音频编码器 (Audio Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Audio Encoder ---")
    # 构造 dummy audio 输入 (参照 Whisper 规范)
    # 假设输入为 1 条音频，包含 1 个 chunk，128个mel bins，长度为 3000
    dummy_mel = torch.randn(1, 128, 3000, dtype=torch.bfloat16, device="cuda")
    dummy_mel_lengths = torch.tensor([300], device="cuda")

    with torch.no_grad():
       
        sgl_audio_out, _ = sglang_model.audio_encoder(dummy_mel, dummy_mel_lengths)
        
        orig_audio_out, _ = train_model.audio_encoder.lm_encode(dummy_mel, dummy_mel_lengths)

        print(sgl_audio_out.shape, orig_audio_out.shape)

    a_max_diff = torch.max(torch.abs(sgl_audio_out - orig_audio_out)).item()
    a_mean_diff = torch.mean(torch.abs(sgl_audio_out - orig_audio_out)).item()
    print(f"Audio Output Shape: {sgl_audio_out.shape}")
    print(f"Audio Max Diff:  {a_max_diff:.6f}")
    print(f"Audio Mean Diff: {a_mean_diff:.6f}")
    if a_mean_diff < 1e-3 and a_max_diff < 1e-2:
        print("✅ Audio Encoder weights loaded perfectly!")
    else:
        print("❌ Audio Encoder has significant precision differences.")

    print("\nAll tests completed.")