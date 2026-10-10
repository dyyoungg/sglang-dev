"""Analyze an FP8 candidate quantization for a BF16 checkpoint.

Recommended sequential workflow::

    MODEL=/path/to/checkpoint
    OUT=./fp8_analysis
    mkdir -p "$OUT"

    # 1. All MoE layers. This includes runtime-shaped w13=[gate;up] and w2.
    python3 benchmark/kernels/fused_moe_triton/analyze_fp8_weight_distribution.py \
        "$MODEL" --skip-linears --output "$OUT/moe.json"

    # 2. All ordinary model/vision Linear checkpoint tensors.
    python3 benchmark/kernels/fused_moe_triton/analyze_fp8_weight_distribution.py \
        "$MODEL" --skip-moe --output "$OUT/linear.json"

    # 3. Quick synthetic activation validation (optional).
    python3 benchmark/kernels/fused_moe_triton/analyze_fp8_weight_distribution.py \
        "$MODEL" --layers 0 24 47 --skip-linears \
        --validation-tokens 256 --output "$OUT/validation_random.json"

    # 4. Representative activation validation (preferred for final decisions).
    # The path can be one Tensor/dict file or a directory of per-TP files;
    # directory inputs are concatenated by key. Dict keys may be w13, w2,
    # linear, default, or an exact checkpoint key. Shapes are [tokens, input_width].
    python3 benchmark/kernels/fused_moe_triton/analyze_fp8_weight_distribution.py \
        "$MODEL" --skip-linears --activation-file "$OUT/acts.pt" \
        --output "$OUT/validation_data.json"

New request-gated format-v2 runtime captures are analyzed by the companion
analyze_fp8_captured_activations.py script. This script accepts legacy flat
Tensor/dict activation files only; those do not preserve routed-expert identity.

The default block shape is [128, 128]. The tool is read-only. It reports
per-expert tensor scales versus block scales, static FP8 reconstruction error,
and optional matrix-output error. It does not create an FP8 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FP8_E4M3_MAX = 448.0
MOE_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)$"
)


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    return {
        "min": values.min().item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "p99": torch.quantile(values, 0.99).item(),
        "max": values.max().item(),
    }


def _block_absmax(weight: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
    experts, n, k = weight.shape
    if n % block_n or k % block_k:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not divisible by "
            f"block_shape=[{block_n}, {block_k}]"
        )
    return weight.abs().reshape(
        experts, n // block_n, block_n, k // block_k, block_k
    ).amax(dim=(2, 4))


def _matrix_block_absmax(weight: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
    n, k = weight.shape
    if n % block_n or k % block_k:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not divisible by "
            f"block_shape=[{block_n}, {block_k}]"
        )
    return weight.abs().reshape(n // block_n, block_n, k // block_k, block_k).amax(
        dim=(1, 3)
    )


def _quant_error(weight: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    scales = torch.clamp(scales.float() / FP8_E4M3_MAX, min=1e-12)
    weight = weight.float()
    quantized = (weight / scales).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    dequantized = quantized.float() * scales
    error = dequantized - weight
    return {
        "relative_rmse": (
            error.square().mean().sqrt() / weight.square().mean().sqrt()
        ).item(),
        "relative_mae": (error.abs().mean() / weight.abs().mean()).item(),
        "saturation_percent": (
            quantized.float().abs().ge(FP8_E4M3_MAX).float().mean() * 100
        ).item(),
    }


def _matmul_validation(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tokens: int,
    seed: int,
    activation: torch.Tensor | None = None,
) -> dict[str, float]:
    if activation is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        activation = torch.randn(
            tokens, weight.shape[-1], generator=generator, dtype=torch.float32
        )
    else:
        activation = activation.float()
        if activation.ndim != 2 or activation.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"activation shape {tuple(activation.shape)} is incompatible with "
                f"weight shape {tuple(weight.shape)}"
            )
    scale = torch.clamp(scales.float() / FP8_E4M3_MAX, min=1e-12)
    quantized = (weight.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    reference = activation @ weight.float().T
    candidate = activation @ (quantized.float() * scale).T
    error = candidate - reference
    return {
        "relative_rmse": (
            error.square().mean().sqrt() / reference.square().mean().sqrt()
        ).item(),
        "relative_mae": (error.abs().mean() / reference.abs().mean()).item(),
    }


def _load(index: dict[str, str], base: Path, key: str) -> torch.Tensor:
    with safe_open(str(base / index[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _load_activations(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    if path.is_dir():
        files = sorted(path.glob("*.pt"))
        if not files:
            raise ValueError(f"--activation-file directory has no .pt files: {path}")
        merged: dict[str, list[torch.Tensor]] = {}
        for file in files:
            value = torch.load(file, map_location="cpu", weights_only=False)
            if isinstance(value, torch.Tensor):
                value = {"default": value}
            if not isinstance(value, dict):
                continue
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    merged.setdefault(key, []).append(tensor)
        return {
            key: torch.cat(tensors, dim=0)
            for key, tensors in merged.items()
        }
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return {"default": value}
    if isinstance(value, dict):
        return {key: tensor for key, tensor in value.items() if isinstance(tensor, torch.Tensor)}
    raise ValueError("--activation-file must contain a Tensor or dict[str, Tensor]")


def _find_activation(
    activations: dict[str, torch.Tensor], kind: str, key: str, input_size: int
) -> torch.Tensor | None:
    candidates = [key]
    # Runtime modules fuse Q/K/V (and vision QKV), while the checkpoint keeps
    # those projections under separate tensor names. They all consume the
    # same input activation, so reuse the fused module's captured input.
    if key.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
        candidates.append(key.rsplit(".", 2)[0] + ".qkv_proj.weight")
    if key.endswith(".attn.qkv.weight"):
        candidates.append(key[: -len(".attn.qkv.weight")] + ".attn.qkv_proj.weight")
    candidates.extend((kind, "default"))
    for candidate in candidates:
        tensor = activations.get(candidate)
        if tensor is not None and tensor.ndim == 2 and tensor.shape[-1] == input_size:
            return tensor
    return None


def _projection_report(
    weight: torch.Tensor,
    block_n: int,
    block_k: int,
    sample_experts: list[int],
    validation_tokens: int,
    validation_seed: int,
    validation_activation: torch.Tensor | None = None,
) -> dict[str, Any]:
    weight = weight.float()
    per_expert = weight.abs().amax(dim=(1, 2))
    blocks = _block_absmax(weight, block_n, block_k)
    block_stats = _percentiles(blocks)
    sample = weight[sample_experts]
    sample_block = blocks[sample_experts]
    expanded_block = sample_block.repeat_interleave(block_n, dim=1).repeat_interleave(
        block_k, dim=2
    )
    sample_tensor = per_expert[sample_experts, None, None]
    report = {
        "shape": list(weight.shape),
        "per_expert_absmax": _percentiles(per_expert),
        "block_absmax": block_stats,
        "block_p95_over_p50": block_stats["p95"] / max(block_stats["p50"], 1e-30),
        "sample_experts": sample_experts,
        "sample_error_per_expert_tensor": _quant_error(sample, sample_tensor),
        "sample_error_block": _quant_error(sample, expanded_block),
    }
    if validation_tokens or validation_activation is not None:
        report["sample_matmul_validation_per_expert_tensor"] = _matmul_validation(
            sample[0], sample_tensor[0], validation_tokens, validation_seed,
            validation_activation,
        )
        report["sample_matmul_validation_block"] = _matmul_validation(
            sample[0], expanded_block[0], validation_tokens, validation_seed,
            validation_activation,
        )
    return report


def _merged_w13_report(
    gate: torch.Tensor,
    up: torch.Tensor,
    block_n: int,
    block_k: int,
    sample_experts: list[int],
    validation_tokens: int,
    validation_seed: int,
    validation_activation: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report the runtime-shaped w13=[gate; up] tensor."""
    gate, up = gate.float(), up.float()
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shapes differ: {gate.shape} vs {up.shape}")
    gate_blocks = _block_absmax(gate, block_n, block_k)
    up_blocks = _block_absmax(up, block_n, block_k)
    merged_blocks = torch.cat([gate_blocks, up_blocks], dim=1)
    merged_expert = torch.maximum(
        gate.abs().amax(dim=(1, 2)), up.abs().amax(dim=(1, 2))
    )
    stats = _percentiles(merged_blocks)
    sample = torch.cat([gate[sample_experts], up[sample_experts]], dim=1)
    sample_block = merged_blocks[sample_experts]
    expanded_block = sample_block.repeat_interleave(block_n, dim=1).repeat_interleave(
        block_k, dim=2
    )
    sample_tensor = merged_expert[sample_experts, None, None]
    report = {
        "shape": [gate.shape[0], gate.shape[1] * 2, gate.shape[2]],
        "runtime_note": "w13 is gate_proj concatenated with up_proj",
        "per_expert_absmax": _percentiles(merged_expert),
        "block_absmax": stats,
        "block_p95_over_p50": stats["p95"] / max(stats["p50"], 1e-30),
        "sample_experts": sample_experts,
        "sample_error_per_expert_tensor": _quant_error(sample, sample_tensor),
        "sample_error_block": _quant_error(sample, expanded_block),
    }
    if validation_tokens or validation_activation is not None:
        report["sample_matmul_validation_per_expert_tensor"] = _matmul_validation(
            sample[0], sample_tensor[0], validation_tokens, validation_seed,
            validation_activation,
        )
        report["sample_matmul_validation_block"] = _matmul_validation(
            sample[0], expanded_block[0], validation_tokens, validation_seed,
            validation_activation,
        )
    return report


def _is_linear_candidate(key: str) -> bool:
    if not (
        key.endswith(".weight")
        and (key.startswith("model.") or key.startswith("image_encoder."))
    ):
        return False
    if ".experts." in key or key.endswith("embed_tokens.weight"):
        return False
    excluded = ("norm", "layernorm", ".mlp.gate.weight", "pos_embed")
    return not any(part in key.lower() for part in excluded)


def _matrix_report(
    weight: torch.Tensor,
    key: str,
    block_n: int,
    block_k: int,
    validation_tokens: int,
    validation_seed: int,
    validation_activation: torch.Tensor | None = None,
) -> dict[str, Any]:
    weight = weight.float()
    row_scale = weight.abs().amax(dim=1, keepdim=True)
    report: dict[str, Any] = {
        "shape": list(weight.shape),
        "per_output_channel_absmax": _percentiles(row_scale),
        "sample_error_per_output_channel": _quant_error(weight, row_scale),
        "checkpoint_key": key,
    }
    if weight.shape[0] % block_n == 0 and weight.shape[1] % block_k == 0:
        blocks = _matrix_block_absmax(weight, block_n, block_k)
        expanded = blocks.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
        stats = _percentiles(blocks)
        report.update(
            {
                "block_absmax": stats,
                "block_p95_over_p50": stats["p95"] / max(stats["p50"], 1e-30),
                "sample_error_block": _quant_error(weight, expanded),
            }
        )
        if validation_tokens or validation_activation is not None:
            report["matmul_validation_per_output_channel"] = _matmul_validation(
                weight, row_scale, validation_tokens, validation_seed,
                validation_activation,
            )
            report["matmul_validation_block"] = _matmul_validation(
                weight, expanded, validation_tokens, validation_seed,
                validation_activation,
            )
    else:
        report["block_supported"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--block-shape", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--sample-experts", nargs="+", type=int, default=[0, 1, 63, 127])
    parser.add_argument("--skip-moe", action="store_true")
    parser.add_argument("--skip-linears", action="store_true")
    parser.add_argument("--validation-tokens", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument(
        "--activation-file",
        type=Path,
        default=None,
        help="torch.save Tensor or dict; keys may be exact checkpoint key, w13, w2, linear, or default",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    activations = _load_activations(args.activation_file)

    index = json.loads(
        (args.model_path / "model.safetensors.index.json").read_text()
    )["weight_map"]
    projections: dict[int, dict[str, str]] = {}
    linear_keys: list[str] = []
    for key in index:
        match = MOE_EXPERT_RE.fullmatch(key)
        if match:
            projections.setdefault(int(match.group(1)), {})[match.group(2)] = key
        elif not args.skip_linears and _is_linear_candidate(key):
            linear_keys.append(key)

    layers = sorted(projections)
    if args.layers is not None:
        layers = [layer for layer in args.layers if layer in projections]
    if not layers and not args.skip_moe:
        raise ValueError("No model.layers.*.mlp.experts.* weights found")

    block_n, block_k = args.block_shape
    output: dict[str, Any] = {
        "model_path": str(args.model_path),
        "block_shape": args.block_shape,
        "fp8_dtype": "float8_e4m3fn",
        "fp8_max": FP8_E4M3_MAX,
        "validation": {
            "tokens": args.validation_tokens,
            "seed": args.validation_seed,
            "activation_file": str(args.activation_file) if args.activation_file else None,
            "note": "Uses --activation-file when a matching key and input width exist; otherwise random Gaussian activations.",
        },
        "moe_layers": {},
        "linear_count": len(linear_keys),
        "linears": {},
    }

    for layer in [] if args.skip_moe else layers:
        gate = _load(index, args.model_path, projections[layer]["gate_proj"])
        up = _load(index, args.model_path, projections[layer]["up_proj"])
        down = _load(index, args.model_path, projections[layer]["down_proj"])
        sample = [expert for expert in args.sample_experts if expert < gate.shape[0]]
        output["moe_layers"][str(layer)] = {
            "w13_merged": _merged_w13_report(
                gate,
                up,
                block_n,
                block_k,
                sample,
                args.validation_tokens,
                args.validation_seed,
                _find_activation(activations, "w13", f"model.layers.{layer}.mlp.experts.w13", gate.shape[-1]),
            ),
            "w2_down": _projection_report(
                down,
                block_n,
                block_k,
                sample,
                args.validation_tokens,
                args.validation_seed,
                _find_activation(activations, "w2", f"model.layers.{layer}.mlp.experts.w2", down.shape[-1]),
            ),
        }
        del gate, up, down
        gc.collect()

    for key in linear_keys:
        with safe_open(
            str(args.model_path / index[key]), framework="pt", device="cpu"
        ) as handle:
            shape = handle.get_slice(key).get_shape()
        if len(shape) != 2:
            continue
        tensor = _load(index, args.model_path, key)
        if tensor.shape[0] % block_n or tensor.shape[1] % block_k:
            report = {
                "shape": list(tensor.shape),
                "checkpoint_key": key,
                "block_supported": False,
            }
        else:
            report = _matrix_report(
                tensor,
                key,
                block_n,
                block_k,
                args.validation_tokens,
                args.validation_seed,
                _find_activation(activations, "linear", key, tensor.shape[-1]),
            )
        output["linears"][key] = report
        del tensor
        gc.collect()

    serialized = json.dumps(output, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
