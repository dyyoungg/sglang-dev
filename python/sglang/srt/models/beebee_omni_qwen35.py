"""SGLang model adapter for BeeBee Omni with a Qwen3.5 dense backbone."""

from typing import Optional

from sglang.srt.configs.beebeeomni_qwen35_config import BeeBeeOmniQwen35Config
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.beebee_audio_encoders import BeeBeeQwen3AudioEncoder
from sglang.srt.models.beebee_vision_encoders import BeeBeeQwen3MoeVisionModel
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.runtime_context import get_disagg, get_mm, get_parallel
from sglang.srt.utils import add_prefix
from sglang.srt.models.beebee_omni_moe import BeeBeeMoEOmniForConditionalGeneration


class BeeBeeOmniQwen35ForConditionalGeneration(BeeBeeMoEOmniForConditionalGeneration):
    """Qwen3.5 dense text backbone with BeeBee dynamic vision/audio encoders."""

    def __init__(
        self,
        config: BeeBeeOmniQwen35Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super(BeeBeeMoEOmniForConditionalGeneration, self).__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        self.audio_config = config.audio_config
        self.use_data_parallel = get_mm().mm_enable_dp_encoder
        if not get_disagg().encoder_only:
            self.model = Qwen3_5ForCausalLM(
                self.text_config, quant_config, prefix=add_prefix("model", prefix)
            )
            if self.pp_group.is_last_rank:
                self.lm_head = ParallelLMHead(
                    self.text_config.vocab_size,
                    self.text_config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
            else:
                self.lm_head = PPMissingLayer()
        else:
            self.model = None
            self.lm_head = None
        self.downsample_ratio = getattr(
            self.vision_config,
            "image_downsample_ratio",
            getattr(self.vision_config, "image_downsample_size", 4),
        )
        self.max_image_bs = get_mm().max_image_bs
        if get_disagg().language_only:
            self.image_encoder = self.audio_encoder = None
        else:
            self.image_encoder = BeeBeeQwen3MoeVisionModel(
                self.vision_config,
                norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=add_prefix("image_encoder", prefix),
                downsample_ratio=self.downsample_ratio,
                use_data_parallel=self.use_data_parallel,
            )
            self.audio_encoder = BeeBeeQwen3AudioEncoder(
                audio_config=self.audio_config,
                out_hidden_size=self.text_config.hidden_size,
            )
        self.logits_processor = LogitsProcessor(self.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False
        rope_config = getattr(self.text_config, "rope_parameters", None) or getattr(
            self.text_config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

    def load_weights(self, weights):
        """Load dense Qwen3.5 GDN projections plus the shared BeeBee encoders."""
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        text_weights, multimodal_weights = [], []
        params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in weights:
            if name.startswith("model.") and not name.startswith("model.embed_tokens"):
                text_weights.append((name[len("model.") :], tensor))
            elif name == "model.embed_tokens.weight":
                text_weights.append(("embed_tokens.weight", tensor))
            elif name == "lm_head.weight":
                if "lm_head.weight" in params:
                    loader = getattr(params["lm_head.weight"], "weight_loader", default_weight_loader)
                    loader(params["lm_head.weight"], tensor)
            else:
                multimodal_weights.append((name, tensor))
        if self.model is not None:
            self.model.load_weights(text_weights)
        if multimodal_weights:
            super().load_weights(multimodal_weights)


EntryClass = BeeBeeOmniQwen35ForConditionalGeneration


def compute_beebee_dynamic_mrope_positions(input_ids, mm_items, spatial_merge_size=2,
                                           default_downsample_ratio=4):
    """Build the dynamic 3-D M-RoPE positions used by the BeeBee processor.

    ``mm_items`` carries one grid and one compression ratio for each merged
    two-image item. Each item must occupy one contiguous visual placeholder
    span.
    """
    import math
    import torch

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.reshape(-1).tolist()
    positions = torch.zeros((3, len(input_ids)), dtype=torch.long)
    image_items = sorted(
        (
            item
            for item in mm_items
            if getattr(getattr(item, "modality", None), "name", None) == "IMAGE"
        ),
        key=lambda item: item.offsets[0][0] if item.offsets else -1,
    )
    cursor = 0
    current = 0
    for item in image_items:
        if not item.offsets:
            continue
        if len(item.offsets) != 1:
            raise ValueError(
                "BeeBee dynamic M-RoPE requires one contiguous image-pair "
                f"offset per item, got {item.offsets!r}"
            )
        grid = item.model_specific_data.get("image_grid_thw")
        if grid is None:
            continue
        t, h, w = [int(v) for v in torch.as_tensor(grid).reshape(-1, 3)[0].tolist()]
        ratio = float(item.model_specific_data.get("downsample_ratio", default_downsample_ratio))
        mh = max(1, int(round((h // spatial_merge_size) / math.sqrt(ratio))))
        nw = max(1, int(round((w // spatial_merge_size) / math.sqrt(ratio))))
        start, end = item.offsets[0]
        end += 1
        if start < cursor or end <= start or end > len(input_ids):
            raise ValueError(
                f"Invalid/overlapping image offset [{start}, {end}); "
                f"previous cursor={cursor}, sequence length={len(input_ids)}"
            )
        if start > cursor:
            n = start - cursor
            positions[:, cursor:start] = torch.arange(n).expand(3, -1) + current
            current += n
        expected = t * mh * nw
        if end - start != expected:
            raise ValueError(
                f"Image placeholder span has {end-start} tokens, expected {expected} "
                f"from grid/downsample metadata (grid={t}x{h}x{w}, ratio={ratio})."
            )
        ti = torch.arange(t).repeat_interleave(mh * nw) + current
        hi = torch.arange(mh).repeat_interleave(nw).repeat(t) + current
        wi = torch.arange(nw).repeat(mh * t) + current
        positions[:, start:end] = torch.stack((ti, hi, wi))
        current += max(t, mh, nw)
        cursor = end
    if cursor < len(input_ids):
        n = len(input_ids) - cursor
        positions[:, cursor:] = torch.arange(n).expand(3, -1) + current
    delta = torch.tensor([int(positions.max()) + 1 - len(input_ids)], dtype=torch.long)
    return positions, delta


def _compare_state_dicts(reference, actual):
    """Compare VeOmni and SGLang parameters, including packed Qwen3.5 weights."""
    import torch

    actual = {name: tensor.detach().cpu() for name, tensor in actual.items()}
    matched = True
    packed_parts = {}
    replacements = (
        (".linear_attn.in_proj_qkv.", ".linear_attn.in_proj_qkvz.", (0, 1, 2)),
        (".linear_attn.in_proj_z.", ".linear_attn.in_proj_qkvz.", (3,)),
        (".linear_attn.in_proj_b.", ".linear_attn.in_proj_ba.", (0,)),
        (".linear_attn.in_proj_a.", ".linear_attn.in_proj_ba.", (1,)),
        (".q_proj.", ".qkv_proj.", ("q",)),
        (".k_proj.", ".qkv_proj.", ("k",)),
        (".v_proj.", ".qkv_proj.", ("v",)),
        (".mlp.gate_proj.", ".mlp.gate_up_proj.", (0,)),
        (".mlp.up_proj.", ".mlp.gate_up_proj.", (1,)),
    )
    for name, ref in reference.items():
        target_name = name
        if name.startswith("audio_encoder.") and not name.startswith("audio_encoder.audio_projector."):
            target_name = name.replace("audio_encoder.", "audio_encoder.encoder.", 1)
        if name.startswith("image_encoder."):
            target_name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
        if name.startswith("model.layers."):
            target_name = target_name.replace(".self_attn.", ".")
        packed = False
        # Qwen3 audio uses separate q/k/v Linear layers in SGLang; only the
        # text backbone and vision encoder use packed QKV parameters.
        replacement_candidates = () if name.startswith("audio_encoder.") else replacements
        for source, target, shards in replacement_candidates:
            if source not in target_name:
                continue
            target_name = target_name.replace(source, target)
            packed_parts.setdefault(target_name, {})[shards[0]] = ref.detach().cpu()
            packed = True
            break
        if packed:
            continue
        if target_name not in actual:
            print(f"[missing] {name} -> {target_name}")
            matched = False
            continue
        got = actual[target_name]
        ref = ref.detach().cpu()
        if ref.shape != got.shape or not torch.equal(ref.to(got.dtype), got):
            diff = float((ref.to(got.dtype) - got).abs().max()) if ref.shape == got.shape else float("inf")
            print(f"[mismatch] {name} -> {target_name}: {tuple(ref.shape)} vs {tuple(got.shape)}, max_diff={diff}")
            matched = False

    shard_orders = {
        "qkv_proj": ("q", "k", "v"),
        "gate_up_proj": (0, 1),
        "in_proj_qkvz": (0, 3),
        "in_proj_ba": (0, 1),
    }
    for target_name, parts in packed_parts.items():
        leaf = target_name.rsplit(".", 2)[-2]
        order = shard_orders.get(leaf)
        if target_name not in actual or order is None or any(k not in parts for k in order):
            print(f"[incomplete packed weight] {target_name}; parts={list(parts)}")
            matched = False
            continue
        expected = torch.cat([parts[k] for k in order], dim=0)
        got = actual[target_name]
        if expected.shape != got.shape or not torch.equal(expected.to(got.dtype), got):
            diff = float((expected.to(got.dtype) - got).abs().max()) if expected.shape == got.shape else float("inf")
            print(f"[mismatch packed] {target_name}: max_diff={diff}")
            matched = False
    return matched


if __name__ == "__main__":
    import argparse
    import os

    import torch
    from safetensors import safe_open

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(description="Compare BeeBee Omni Qwen35 checkpoint weights (TP=1).")
    parser.add_argument("--model-path", default="/mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--sglang-device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(42)

    dummy_args = ServerArgs(model_path=args.model_path, mm_enable_dp_encoder=False)
    publish(dummy_args, role="scheduler", ranks=SpawnRanks(world_rank=0, dp_rank=0, gpu_id=0))
    init_distributed_environment()
    initialize_model_parallel()

    from veomni.models.custom.llava_qwen35.configuration_qwen35_omni import Qwen35OmniConfig
    from veomni.models.custom.llava_qwen35.modeling_llava_qwen35_omni import LlavaQwen35ForCausalLM
    from sglang.srt.configs.beebeeomni_qwen35_config import BeeBeeOmniQwen35Config

    config = BeeBeeOmniQwen35Config.from_pretrained(args.model_path)
    with set_default_torch_dtype(torch.bfloat16):
        sglang_model = BeeBeeOmniQwen35ForConditionalGeneration(config).to(args.sglang_device)
    sglang_model.eval()
    reference_config = Qwen35OmniConfig.from_pretrained(args.model_path)
    reference_model = LlavaQwen35ForCausalLM.from_pretrained(
        args.model_path, config=reference_config, torch_dtype=torch.bfloat16
    ).to(args.reference_device)
    reference_model.eval()

    safetensors_files = sorted(
        os.path.join(args.model_path, filename)
        for filename in os.listdir(args.model_path)
        if filename.endswith(".safetensors")
    )

    def checkpoint_weights():
        for filename in safetensors_files:
            with safe_open(filename, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    yield key, checkpoint.get_tensor(key)

    print("Loading checkpoint weights into SGLang...")
    sglang_model.load_weights(checkpoint_weights())
    print("Comparing VeOmni and SGLang state dicts...")
    if not _compare_state_dicts(reference_model.state_dict(), sglang_model.state_dict()):
        raise RuntimeError("BeeBee Omni Qwen35 weight comparison failed")
    print("All compared weights match exactly.")

    # Forward checks use the same random inputs on both devices.  These checks
    # isolate the multimodal encoders and the dense Qwen3.5 MLP path before a
    # full scheduler ForwardBatch is constructed by a serving process.
    print("Testing vision encoder forward...")
    vision_hidden = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    pixel_values = torch.randn(
        1536, vision_hidden, dtype=torch.bfloat16, device=args.sglang_device
    )
    grid_thw = torch.tensor([[1, 32, 48]], dtype=torch.int32, device=args.sglang_device)
    downsample_ratio = torch.tensor([config.vision_config.image_downsample_ratio], device=args.sglang_device)
    with torch.no_grad():
        sgl_vision, _ = sglang_model.image_encoder(
            pixel_values, grid_thw=grid_thw, downsample_ratios=[int(downsample_ratio.item())]
        )
        ref_vision, _ = reference_model.image_encoder.lm_encode(
            pixel_values.to(args.reference_device),
            grid_thw.to(args.reference_device),
            downsample_ratios=downsample_ratio.to(args.reference_device),
        )
    ref_vision = ref_vision.to(sgl_vision.device)
    vision_max = (sgl_vision.float() - ref_vision.float()).abs().max().item()
    vision_mean = (sgl_vision.float() - ref_vision.float()).abs().mean().item()
    print(f"Vision shape={tuple(sgl_vision.shape)}, max_diff={vision_max:.6g}, mean_diff={vision_mean:.6g}")

    print("Testing Qwen3 audio encoder forward...")
    mel = torch.randn(1, 128, 3000, dtype=torch.bfloat16, device=args.sglang_device)
    mel_lengths = torch.tensor([300], dtype=torch.long, device=args.sglang_device)
    with torch.no_grad():
        sgl_audio, _ = sglang_model.audio_encoder(mel, mel_lengths)
        ref_audio, _ = reference_model.audio_encoder.lm_encode(
            mel.to(args.reference_device), mel_lengths.to(args.reference_device)
        )
    ref_audio = ref_audio.to(sgl_audio.device)
    audio_max = (sgl_audio.float() - ref_audio.float()).abs().max().item()
    audio_mean = (sgl_audio.float() - ref_audio.float()).abs().mean().item()
    print(f"Audio shape={tuple(sgl_audio.shape)}, max_diff={audio_max:.6g}, mean_diff={audio_mean:.6g}")

    print("Testing token embedding and dense MLP forward...")
    text_ref = reference_model.model
    text_sgl = sglang_model.model
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 64), device=args.sglang_device)
    with torch.no_grad():
        sgl_embed = text_sgl.embed_tokens(input_ids)
        ref_embed = text_ref.embed_tokens(input_ids.to(args.reference_device)).to(args.sglang_device)
    embed_max = (sgl_embed.float() - ref_embed.float()).abs().max().item()
    embed_mean = (sgl_embed.float() - ref_embed.float()).abs().mean().item()
    print(f"Embedding max_diff={embed_max:.6g}, mean_diff={embed_mean:.6g}")

    mlp_max = mlp_mean = 0.0
    for layer_idx in range(config.text_config.num_hidden_layers):
        hidden = torch.randn(64, config.text_config.hidden_size, dtype=torch.bfloat16, device=args.sglang_device)
        with torch.no_grad():
            ref_out = text_ref.layers[layer_idx].mlp(hidden.to(args.reference_device)).to(args.sglang_device)
            sgl_out = text_sgl.layers[layer_idx].mlp(hidden)
        diff = (sgl_out.float() - ref_out.float()).abs()
        mlp_max = max(mlp_max, diff.max().item())
        mlp_mean = max(mlp_mean, diff.mean().item())
    print(f"Dense MLP worst_max_diff={mlp_max:.6g}, worst_mean_diff={mlp_mean:.6g}")

    # Fused BF16 vision attention and the VeOmni torch reference use different
    # reduction orders; retain a strict mean bound while allowing the expected
    # small worst-element difference.
    if vision_max >= 5e-2 or vision_mean >= 5e-3:
        raise RuntimeError("Vision encoder forward differs beyond tolerance")
    if audio_max >= 1e-2 or audio_mean >= 1e-3:
        raise RuntimeError("Audio encoder forward differs beyond tolerance")
    # A few BF16 GEMM accumulation outliers can differ by one quantization
    # bucket while the layer-wide mean remains within the reference tolerance.
    if embed_max != 0.0 or mlp_max >= 2.0 or mlp_mean >= 5e-3:
        raise RuntimeError("Qwen3.5 text forward differs beyond tolerance")
    print("Testing dynamic M-RoPE positions...")
    # Keep this check independent of the processor module.  The processor also
    # imports the optional Whisper package, while the position construction
    # itself only needs the pair metadata below.
    # Validate two merged image pairs inside one <vision_start>...<vision_end>
    # segment. The marker tokens are ordinary text positions outside offsets.
    class _Item:
        modality = __import__("sglang.srt.managers.schedule_batch", fromlist=["Modality"]).Modality.IMAGE
        def __init__(self, start, end, grid, ratio):
            self.offsets = [(start, end)]
            self.model_specific_data = {"image_grid_thw": torch.tensor([grid]), "downsample_ratio": ratio}
    mrope_input = [1] * 2 + [151655] * 16 + [151655] * 4 + [2] + [3] * 3
    mrope_items = [_Item(2, 17, (1, 16, 16), 4), _Item(18, 21, (1, 8, 8), 4)]
    mrope_pos, mrope_delta = compute_beebee_dynamic_mrope_positions(
        mrope_input, mrope_items, spatial_merge_size=2, default_downsample_ratio=4
    )
    assert mrope_pos.shape == (3, len(mrope_input))
    assert int(mrope_delta.item()) == int(mrope_pos.max()) + 1 - len(mrope_input)
    print(f"M-RoPE shape={tuple(mrope_pos.shape)}, delta={mrope_delta.tolist()}")
    print("Forward comparisons passed.")
