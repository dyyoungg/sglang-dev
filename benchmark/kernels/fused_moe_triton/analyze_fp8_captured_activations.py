"""Analyze request-gated format-v2 captures, preserving layer/expert pairing.

Usage (after running api_test.py in images and video mode):
  OMP_NUM_THREADS=4 /usr/bin/python3 benchmark/kernels/fused_moe_triton/analyze_fp8_captured_activations.py \
    MODEL --capture-dir fp8_analysis/real_assets_v3 --modality images \
    --device cuda --output fp8_analysis/real_assets_v3/images_results.json
  # Repeat with --modality video and --output .../video_results.json.

All captured experts and FP8 Linear modules are analyzed. No random fallback.
We compare FP32 X@W.T to X@dequant(FP8(W)).T, isolating weight error, not full
W8A8/end-to-end quality. Captured X comes from the FP8 service; upstream FP8
error is present in both candidates. Block=128x128 uses partial tail blocks
when needed; those are mathematical candidates, not claims of kernel support.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def block_max(w, block=128):
    n, k = w.shape
    padded = F.pad(w.abs(), (0, (-k) % block, 0, (-n) % block))
    maxima = padded.reshape(
        padded.shape[0] // block, block, padded.shape[1] // block, block
    ).amax((1, 3))
    return maxima.repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]


def dq(w, maximum):
    scale = (maximum / 448.0).clamp_min(1e-12)
    return (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def compare(w, x, baseline, device):
    w, x = w.to(device=device, dtype=torch.float32), x.to(
        device=device, dtype=torch.float32
    )
    if not torch.isfinite(w).all() or not torch.isfinite(x).all():
        raise ValueError("Nonfinite weights or activations")
    base = dq(
        w,
        w.abs().amax(dim=1, keepdim=True) if baseline == "channel" else w.abs().amax(),
    )
    block = dq(w, block_max(w))
    stats = torch.zeros(6, device=device, dtype=torch.float64)
    for chunk in x.split(256):
        y = chunk @ w.T
        e0, e1 = chunk @ base.T - y, chunk @ block.T - y
        stats += torch.stack(
            (
                y.double().square().sum(),
                e0.double().square().sum(),
                e1.double().square().sum(),
                y.double().abs().sum(),
                e0.double().abs().sum(),
                e1.double().abs().sum(),
            )
        )
    ref, e0, e1, absref, a0, a1 = stats.tolist()
    return {
        "tokens": len(x),
        "shape": list(w.shape),
        "baseline": baseline,
        "reference_energy": ref,
        "baseline_error_energy": e0,
        "block_error_energy": e1,
        "baseline_rmse": math.sqrt(e0 / max(ref, 1e-30)),
        "block_rmse": math.sqrt(e1 / max(ref, 1e-30)),
        "baseline_mae": a0 / max(absref, 1e-30),
        "block_mae": a1 / max(absref, 1e-30),
        "block_aligned": all(dim % 128 == 0 for dim in w.shape),
    }


def summarize(rows):
    if not rows:
        return {"count": 0}
    ref = sum(v["reference_energy"] for v in rows)
    return {
        "count": len(rows),
        "tokens_sum": sum(v["tokens"] for v in rows),
        "baseline_mean_rmse": sum(v["baseline_rmse"] for v in rows) / len(rows),
        "block_mean_rmse": sum(v["block_rmse"] for v in rows) / len(rows),
        "baseline_pooled_rmse": math.sqrt(
            sum(v["baseline_error_energy"] for v in rows) / max(ref, 1e-30)
        ),
        "block_pooled_rmse": math.sqrt(
            sum(v["block_error_energy"] for v in rows) / max(ref, 1e-30)
        ),
        "block_better": sum(v["block_rmse"] < v["baseline_rmse"] for v in rows),
        "block_worse": sum(v["block_rmse"] > v["baseline_rmse"] for v in rows),
    }


def checkpoint_keys(runtime_key):
    if runtime_key.startswith("model.") and runtime_key.endswith(".qkv_proj.weight"):
        prefix = runtime_key[: -len("qkv_proj.weight")]
        return [
            prefix + suffix + ".weight" for suffix in ("q_proj", "k_proj", "v_proj")
        ]
    return [runtime_key.replace(".attn.qkv_proj.weight", ".attn.qkv.weight")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=["images", "video"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def weight(key):
        with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    groups = defaultdict(list)
    manifest = [
        json.loads(line)
        for line in (args.capture_dir / "requests.jsonl").read_text().splitlines()
    ]
    cases = {
        r["case_id"]
        for r in manifest
        if r["modality"] == args.modality and r["status"] == "ok"
    }
    for case in sorted(cases):
        for file in sorted((args.capture_dir / args.modality / case).glob("*.pt")):
            # filename ends .<chunk offset>.pt; all earlier dots are the key
            groups[file.name.rsplit(".", 2)[0]].append(file)
    if not groups:
        raise ValueError("No successful request captures")
    report = {
        "modality": args.modality,
        "cases": sorted(cases),
        "model": str(args.model),
        "capture_format": 2,
        "phase": "prefill",
        "block_shape": [128, 128],
        "note": "Weight-only FP32 matmul simulation on matched FP8-service inputs; no activation quantization. Partial block tails simulated.",
        "moe": {},
        "linears": {},
        "coverage": {},
    }
    for i, (key, paths) in enumerate(sorted(groups.items())):
        records = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
        assert all(
            r["metadata"]["format_version"] == 2
            and r["metadata"]["key"] == key
            and r["metadata"]["phase"] == "prefill"
            for r in records
        )
        if records[0]["kind"] == "moe":
            ids = torch.cat([r["expert_ids"] for r in records]).long()
            x13 = torch.cat([r["w13"] for r in records])
            x2 = torch.cat([r["w2"] for r in records])
            gate, up, down = (
                weight(key + "." + s) for s in ("gate_proj", "up_proj", "down_proj")
            )
            assert x13.shape[1] == gate.shape[-1] and x2.shape[1] == down.shape[-1]
            experts = {}
            for eid in sorted(ids.unique().tolist()):
                mask = ids == eid
                experts[str(eid)] = {
                    "w13": compare(
                        torch.cat((gate[eid], up[eid])),
                        x13[mask],
                        "expert_tensor",
                        args.device,
                    ),
                    "w2": compare(down[eid], x2[mask], "expert_tensor", args.device),
                }
            report["moe"][key] = {
                "experts": experts,
                "observed_experts": len(experts),
                "total_experts": len(gate),
                "routed_rows": len(ids),
                "cases": len({r["metadata"]["case_id"] for r in records}),
            }
            del gate, up, down, x13, x2, ids
        else:
            x = torch.cat([r["x"] for r in records])
            for ck in checkpoint_keys(key):
                report["linears"][ck] = dict(
                    compare(weight(ck), x, "channel", args.device),
                    runtime_key=key,
                    cases=len({r["metadata"]["case_id"] for r in records}),
                )
            del x
        del records
        print(f"{args.modality}: {i+1}/{len(groups)} {key}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary = {}
    for kind in ("w13", "w2"):
        summary["moe_" + kind] = summarize(
            [e[kind] for l in report["moe"].values() for e in l["experts"].values()]
        )
    for component in ("model", "image_encoder"):
        rows = [
            v for k, v in report["linears"].items() if k.startswith(component + ".")
        ]
        summary[component + "_linear_all"] = summarize(rows)
        summary[component + "_linear_aligned"] = summarize(
            [v for v in rows if v["block_aligned"]]
        )
    summary["linear_all"] = summarize(list(report["linears"].values()))
    summary["linear_aligned"] = summarize(
        [v for v in report["linears"].values() if v["block_aligned"]]
    )
    report["summary"] = summary
    report["coverage"] = {
        "moe_layers": len(report["moe"]),
        "linear_checkpoint_matrices": len(report["linears"]),
        "runtime_modules": len(groups),
        "successful_requests": len(cases),
        "files": sum(map(len, groups.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
