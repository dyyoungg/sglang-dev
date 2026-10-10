"""Request-gated, prefill-only FP8 activation capture (single TP/PP rank).

Set SGLANG_FP8_ACTIVATION_DUMP_DIR and use api_test.py's asset mode with
API_TEST_CAPTURE_DIR pointing at the same directory. The client arms control.json
only during a measured request. No startup/health/decode activations are saved.
Each request saves separate Linear and MoE records, with exact module names and
routed expert IDs. SGLANG_FP8_ACTIVATION_MAX_TOKENS (default 1024) is a PER REQUEST,
PER MODULE quota, sampled evenly across the available rows. Files are atomic and
written immediately; no dependence on worker atexit/shutdown behavior.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

_CAPTURE = None


def sample_indices(size, limit, device):
    count = min(size, limit)
    return torch.linspace(0, size - 1, count, device=device).long()


class Fp8ActivationCapture:
    def __init__(self, output_dir, tp_rank, pp_rank):
        if tp_rank != 0 or pp_rank != 0:
            raise ValueError("Matched checkpoint analysis requires TP=PP=1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = int(os.getenv("SGLANG_FP8_ACTIVATION_MAX_TOKENS", "1024"))
        if self.max_tokens < 1:
            raise ValueError("capture token limit must be positive")
        self.context = None
        self.counts = {}
        self.weights = {}

    def begin(self, module, args, kwargs):
        self.context = None
        batch = kwargs.get("forward_batch")
        if batch is None:
            batch = next((v for v in args if hasattr(v, "forward_mode")), None)
        if batch is None or not batch.forward_mode.is_extend():
            return
        control = self.output_dir / "control.json"
        if not control.exists():
            return
        ctx = json.loads(control.read_text())
        case = ctx["case_id"]
        if Path(case).name != case:
            raise ValueError("case_id must be a single path component")
        if ctx.get("enabled"):
            self.context = ctx

    def remaining(self, key):
        if self.context is None:
            return 0
        return max(
            0, self.max_tokens - self.counts.get((self.context["case_id"], key), 0)
        )

    def write(self, key, record, rows):
        case = self.context["case_id"]
        folder = self.output_dir / self.context["modality"] / case
        folder.mkdir(parents=True, exist_ok=True)
        count = self.counts.get((case, key), 0)
        path = folder / f"{key}.{count:06d}.pt"
        record["metadata"] = dict(
            self.context,
            key=key,
            phase="prefill",
            quota=self.max_tokens,
            format_version=2,
            activation_source="FP8 service before activation quantization",
        )
        tmp = path.with_suffix(".tmp")
        torch.save(record, tmp)
        tmp.replace(path)
        self.counts[case, key] = count + rows

    def linear(self, name, x):
        key = name + ".weight"
        remaining = self.remaining(key)
        if not remaining or not x.numel():
            return
        x = x.detach().reshape(-1, x.shape[-1])
        ids = sample_indices(len(x), remaining, x.device)
        self.write(
            key,
            {
                "x": x[ids].cpu(),
                "row_ids": ids.cpu(),
                "source_rows": len(x),
                "kind": "linear",
            },
            len(ids),
        )

    def moe(self, w1, x, down_input, topk_ids, sorted_ids, sorted_layout):
        name = self.weights.get(w1.data_ptr())
        if name is None:
            return
        remaining = self.remaining(name)
        if not remaining or not x.numel():
            return
        # Down activations may be expert-sorted with interspersed padding.
        # Recover the routed-token index for each physical row, never slice
        # the first num_tokens*topk rows of a sorted/padded buffer.
        routes = topk_ids.reshape(-1)
        if sorted_layout:
            physical = torch.arange(
                min(len(sorted_ids), len(down_input)), device=x.device
            )
            route_ids = sorted_ids[: len(physical)].long()
            valid = (route_ids >= 0) & (route_ids < routes.numel())
            physical, route_ids = physical[valid], route_ids[valid]
        else:
            route_ids = torch.arange(routes.numel(), device=x.device)
            physical = route_ids
        experts = routes[route_ids]
        valid = (experts >= 0) & (experts < w1.shape[0])
        physical, route_ids, experts = physical[valid], route_ids[valid], experts[valid]
        if not len(route_ids):
            return
        selected = sample_indices(len(route_ids), remaining, x.device)
        physical, route_ids, experts = (
            physical[selected],
            route_ids[selected],
            experts[selected],
        )
        tokens = route_ids // topk_ids.shape[1]
        self.write(
            name,
            {
                "kind": "moe",
                "w13": x[tokens].detach().cpu(),
                "w2": down_input[physical].detach().cpu(),
                "expert_ids": experts.cpu(),
                "route_ids": route_ids.cpu(),
                "source_tokens": len(x),
                "top_k": topk_ids.shape[1],
                "sorted_layout": sorted_layout,
            },
            len(selected),
        )


def configure_fp8_activation_capture(output_dir, tp_rank, pp_rank):
    global _CAPTURE
    _CAPTURE = Fp8ActivationCapture(output_dir, tp_rank, pp_rank)
    return _CAPTURE


def get_fp8_activation_capture():
    return _CAPTURE


def register_fp8_linear_activation_hooks(model):
    capture = _CAPTURE
    if capture is None:
        return 0
    # ModelRunner calls model.forward directly, bypassing root nn.Module hooks.
    original_forward = model.forward

    @functools.wraps(original_forward)
    def captured_forward(*args, **kwargs):
        capture.begin(model, args, kwargs)
        try:
            return original_forward(*args, **kwargs)
        finally:
            capture.context = None

    model.forward = captured_forward
    registered = 0
    for name, module in model.named_modules():
        method = type(getattr(module, "quant_method", None)).__name__
        if method == "Fp8MoEMethod":
            capture.weights[module.w13_weight.data_ptr()] = name
        elif method == "Fp8LinearMethod":

            def hook(module, args, kwargs, name=name):
                x = args[0] if args else kwargs.get("input_", kwargs.get("x"))
                if isinstance(x, torch.Tensor):
                    capture.linear(name, x)

            module.register_forward_pre_hook(hook, with_kwargs=True)
            registered += 1
    return registered
