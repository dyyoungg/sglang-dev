"""Local checkpoint integration test; requires VeOmni on PYTHONPATH and CUDA.

Run with torchrun --standalone --nproc-per-node=1 (or 2 for TP/DP checks).
TP=1 and replicated encoders require bitwise equality. Sharded TP reports
rounding differences; pass --require-exact to require equality there as well.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.configs.beebeeomni_config import BeeBeeOmniConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models.beebee_omni import BeeBeeOmniForConditionalGeneration
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


def compare(label, actual, expected, exact):
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert torch.isfinite(actual).all(), label
    delta = actual.float() - expected.float()
    result = {
        "case": label,
        "shape": list(actual.shape),
        "equal": torch.equal(actual, expected),
        "max_diff": delta.abs().max().item(),
        "mean_diff": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / expected.float().norm()).item(),
    }
    if torch.distributed.get_rank() == 0:
        print(json.dumps(result), flush=True)
    if exact:
        assert result["equal"], result
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/mnt/afs/share/20260907_beebee_14B_veomni_28k"
    )
    parser.add_argument("--dp-encoder", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path=args.model_path,
            encoder_only=True,
            tp_size=world_size,
            mm_enable_dp_encoder=args.dp_encoder,
            mm_attention_backend="fa2",
            mm_use_compiled_rope=False,
        )
    )
    server_args = get_global_server_args()
    initialize_dp_attention(server_args, server_args.get_model_config())
    torch.manual_seed(42)
    config = BeeBeeOmniConfig.from_pretrained(args.model_path)
    with set_default_torch_dtype(torch.bfloat16):
        sgl = BeeBeeOmniForConditionalGeneration(config).cuda().eval()

    # Import after SGLang config construction: both repositories register the
    # same Hugging Face model types with different config wrappers.
    from veomni.models.custom.llava_qwen2.configuration_llava_qwen2 import (
        LlavaQwen2Config,
    )
    from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import (
        BeeBeeVLAudioModel,
    )
    from veomni.models.custom.vision_encoder.modeling_qwen25_vision_encoder import (
        BeeBeeVLVisionModel,
    )
    from veomni.distributed.parallel_state import _init_parallel_state

    _init_parallel_state(
        dp_size=world_size, dp_shard_size=world_size, device_type="cuda"
    )

    ref_config = LlavaQwen2Config.from_pretrained(args.model_path)
    audio = (
        BeeBeeVLAudioModel._from_config(
            ref_config.encoder_config.audio_config, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    vision = (
        BeeBeeVLVisionModel._from_config(
            ref_config.encoder_config.image_config, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    weights = {}
    for shard in sorted(Path(args.model_path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(("audio_encoder.", "image_encoder.")):
                    weights[key] = f.get_tensor(key)
    sgl.load_weights(weights.items())
    for prefix, model in (("audio_encoder.", audio), ("image_encoder.", vision)):
        model.load_state_dict(
            {
                k.removeprefix(prefix): v
                for k, v in weights.items()
                if k.startswith(prefix)
            },
            strict=True,
        )
    del weights
    exact = world_size == 1 or args.dp_encoder or args.require_exact
    results = []
    for seed, lengths in ((42, [300]), (123, [51, 301]), (2026, [1500])):
        torch.manual_seed(seed)
        mel = torch.randn(len(lengths), 128, 3000, device="cuda", dtype=torch.bfloat16)
        lengths = torch.tensor(lengths, device="cuda")
        actual, actual_lengths = sgl.audio_encoder(mel, lengths)
        expected, expected_lengths = audio.lm_encode(mel, lengths)
        assert list(actual_lengths) == list(expected_lengths)
        results.append(compare(f"audio/{seed}", actual, expected, exact))
        if args.dp_encoder and world_size == 2:
            from sglang.srt.multimodal.mm_utils import run_dp_sharded_audio_model

            # At most two requests: each DP rank executes at most one, so the
            # reference uses the same per-request batch shapes as dispatch.
            actual = run_dp_sharded_audio_model(
                sgl.audio_encoder,
                list(mel.split(1)),
                [[int(length)] for length in lengths],
            )
            expected = torch.cat(
                [
                    audio.lm_encode(mel[i : i + 1], lengths[i : i + 1])[0]
                    for i in range(len(lengths))
                ]
            )
            results.append(compare(f"audio_dispatch/{seed}", actual, expected, True))
    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    for seed, grids, ratios in (
        (42, [[1, 32, 48]], [16]),
        (123, [[1, 16, 16], [1, 16, 24]], [4, 16]),
        (2026, [[2, 16, 16]], [16]),
    ):
        torch.manual_seed(seed)
        grid = torch.tensor(grids, device="cuda", dtype=torch.int32)
        tokens = sum(t * h * w for t, h, w in grids)
        pixels = torch.randn(tokens, patch_width, device="cuda", dtype=torch.bfloat16)
        actual, _ = sgl.image_encoder(pixels, grid, downsample_ratios=ratios)
        expected, _ = vision.lm_encode(
            pixels, grid, downsample_ratios=torch.tensor(ratios, device="cuda")
        )
        results.append(compare(f"vision/{seed}", actual, expected, exact))
        if args.dp_encoder and world_size == 2:
            from sglang.srt.multimodal.mm_utils import (
                run_dp_sharded_beebee_vision_model,
            )

            actual = run_dp_sharded_beebee_vision_model(
                sgl.image_encoder, pixels, grids, downsample_ratios=ratios
            )
            expected_parts = []
            start = 0
            for i, (t, h, w) in enumerate(grids):
                stop = start + t * h * w
                expected_parts.append(
                    vision.lm_encode(
                        pixels[start:stop],
                        grid[i : i + 1],
                        downsample_ratios=torch.tensor(
                            ratios[i : i + 1], device="cuda"
                        ),
                    )[0]
                )
                start = stop
            results.append(
                compare(
                    f"vision_dispatch/{seed}", actual, torch.cat(expected_parts), True
                )
            )
    # Graph workspaces must preserve FP32 RoPE, including after growth.
    from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner

    runner = ViTCudaGraphRunner(sgl.image_encoder)
    for size in (16, 32):
        runner._ensure_sin_cos_ws(size, 80)
        assert all(t.dtype == torch.float32 for t in runner.sin_cos_ws)
    if torch.distributed.get_rank() == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "dp_encoder": args.dp_encoder,
                    "all_equal": all(r["equal"] for r in results),
                }
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
