"""Paired eager encoder timing before/after the BeeBee alignment fixes.

Requires CUDA and the local checkpoint. Run with torchrun, one or two processes.
Baseline encoder sources come from --baseline-ref; only the Transformers 5 RoPE
API compatibility fix and propagation of the noncompiled RoPE flag are retained.
No working-tree source is reverted. Both models use the same loaded weights.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch
from safetensors import safe_open

from sglang.srt.configs.beebeeomni_config import BeeBeeOmniConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import beebee_omni
from sglang.srt.multimodal.mm_utils import (
    run_dp_sharded_audio_model,
    run_dp_sharded_beebee_vision_model,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def baseline_module(repo, ref, name):
    path = f"python/sglang/srt/models/{name}.py"
    source = subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=repo, text=True
    )
    if name == "beebee_vision_encoders":
        old = "rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)"
        assert source.count(old) == 1
        source = source.replace(
            old,
            "inv_freq = self.rotary_pos_emb.inv_freq\n"
            "        positions = torch.arange(max_grid_size, device=inv_freq.device, "
            "dtype=inv_freq.dtype)\n"
            "        rotary_pos_emb_full = torch.outer(positions, inv_freq)",
        )
        old = "num_dummy_heads=num_dummy_heads,\n            use_data_parallel=use_data_parallel,\n"
        assert source.count(old) == 1
        source = source.replace(
            old,
            old
            + "            use_compiled_rope=get_global_server_args().mm_use_compiled_rope,\n",
        )
    module = types.ModuleType(f"_alignment_baseline_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return module, source


def measure(fn, iterations):
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000 / iterations
    gpu_ms = begin.elapsed_time(end) / iterations
    # Report the slowest rank, including the actual DP dispatch/gather path.
    times = torch.tensor([wall_ms, gpu_ms], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
    return times.tolist()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default="/mnt/afs/share/20260907_beebee_14B_veomni_28k"
    )
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--dp-encoder", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world)
    server_args = ServerArgs(
        model_path=args.model_path,
        encoder_only=True,
        tp_size=world,
        mm_enable_dp_encoder=args.dp_encoder,
        mm_attention_backend="fa2",
        mm_use_compiled_rope=False,
    )
    set_global_server_args_for_scheduler(server_args)
    initialize_dp_attention(server_args, server_args.get_model_config())
    config = BeeBeeOmniConfig.from_pretrained(args.model_path)
    baseline_ref = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=repo, text=True
    ).strip()
    audio_module, audio_source = baseline_module(
        repo, baseline_ref, "beebee_audio_encoders"
    )
    vision_module, vision_source = baseline_module(
        repo, baseline_ref, "beebee_vision_encoders"
    )
    with set_default_torch_dtype(torch.bfloat16):
        after = beebee_omni.BeeBeeOmniForConditionalGeneration(config).cuda().eval()
        saved_audio = beebee_omni.BeeBeeAudioEncoder
        saved_vision = beebee_omni.BeeBeeQwen25VisionModel
        try:
            beebee_omni.BeeBeeAudioEncoder = audio_module.BeeBeeAudioEncoder
            beebee_omni.BeeBeeQwen25VisionModel = vision_module.BeeBeeQwen25VisionModel
            before = (
                beebee_omni.BeeBeeOmniForConditionalGeneration(config).cuda().eval()
            )
        finally:
            beebee_omni.BeeBeeAudioEncoder = saved_audio
            beebee_omni.BeeBeeQwen25VisionModel = saved_vision
    weights = {}
    for shard in sorted(Path(args.model_path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(("audio_encoder.", "image_encoder.")):
                    weights[key] = f.get_tensor(key)
    before.load_weights(weights.items())
    after.load_weights(weights.items())
    del weights
    before_sd = before.state_dict()
    after_sd = after.state_dict()
    assert before_sd.keys() == after_sd.keys()
    assert all(torch.equal(value, after_sd[key]) for key, value in before_sd.items())
    del before_sd, after_sd

    cases = []
    for seed, lengths in ((42, [300]), (123, [51, 301]), (2026, [1500])):
        torch.manual_seed(seed)
        mel = torch.randn(len(lengths), 128, 3000, device="cuda", dtype=torch.bfloat16)
        lens = torch.tensor(lengths, device="cuda")
        if args.dp_encoder:
            chunks = list(mel.split(1))
            lens_list = [[n] for n in lengths]

            def invoke(model, chunks=chunks, lens_list=lens_list):
                return run_dp_sharded_audio_model(
                    model.audio_encoder, chunks, lens_list
                )

        else:

            def invoke(model, mel=mel, lens=lens):
                return model.audio_encoder(mel, lens)[0]

        cases.append((f"audio/{lengths}", invoke))
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
        pixels = torch.randn(
            sum(t * h * w for t, h, w in grids),
            patch_width,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if args.dp_encoder:

            def invoke(model, pixels=pixels, grids=grids, ratios=ratios):
                return run_dp_sharded_beebee_vision_model(
                    model.image_encoder, pixels, grids, downsample_ratios=ratios
                )

        else:

            def invoke(model, pixels=pixels, grid=grid, ratios=ratios):
                return model.image_encoder(pixels, grid, downsample_ratios=ratios)[0]

        cases.append((f"vision/{grids}", invoke))
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "baseline_ref": baseline_ref,
        "baseline_compatibility": "HF5 rotary API and noncompiled RoPE flag propagation",
        "world_size": world,
        "dp_encoder": args.dp_encoder,
        "warmup": args.warmup,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "backend": "fa2",
        "compiled_rope": False,
        "cuda_graph": False,
        "dtype": "bfloat16",
        "model_path": args.model_path,
        "source_sha256": {},
        "results": [],
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for name, source in (("audio", audio_source), ("vision", vision_source)):
            snapshot = args.output.with_name(f"baseline_{name}.py")
            snapshot.write_text(source)
            report["source_sha256"][f"before_{name}"] = hashlib.sha256(
                source.encode()
            ).hexdigest()
            current = (
                repo / f"python/sglang/srt/models/beebee_{name}_encoders.py"
            ).read_bytes()
            report["source_sha256"][f"after_{name}"] = hashlib.sha256(
                current
            ).hexdigest()
        print(
            json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True
        )
    for label, invoke in cases:
        functions = {"before": lambda: invoke(before), "after": lambda: invoke(after)}
        outputs = [fn() for fn in functions.values()]
        assert outputs[0].shape == outputs[1].shape
        assert all(torch.isfinite(out).all() for out in outputs)
        output_diff = (outputs[0].float() - outputs[1].float()).abs().max().item()
        del outputs
        for fn in functions.values():
            for _ in range(args.warmup):
                fn()
        samples = {key: [] for key in functions}
        for repeat in range(args.rounds):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            for key in order:
                samples[key].append(measure(functions[key], args.iterations))
        result = {"case": label, "before_after_max_diff": output_diff}
        for key, values in samples.items():
            wall = [v[0] for v in values]
            result[key] = {
                "wall_ms": statistics.median(wall),
                "gpu_ms": statistics.median(v[1] for v in values),
                "wall_min_ms": min(wall),
                "wall_max_ms": max(wall),
                "samples_wall_gpu_ms": values,
            }
        result["latency_increase_pct"] = (
            result["after"]["wall_ms"] / result["before"]["wall_ms"] - 1
        ) * 100
        report["results"].append(result)
        if rank == 0:
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
