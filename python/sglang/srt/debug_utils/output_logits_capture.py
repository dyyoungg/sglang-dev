"""Opt-in full-vocabulary logits capture and reference-token replay.

For isolated serial diagnostics only: TP=PP=1, eager execution, no overlap.
Set SGLANG_OUTPUT_LOGITS_DIR and atomically write control.json in that directory:
  {"case_id": "image", "rid": "unique-request-id", "max_steps": 96,
   "forced_token_ids": [...]}
Omit forced_token_ids for natural generation. Only the exact rid is recorded.
Raw logits are copied before sampling biases/temperature; candidate runs return
the reference tokens to keep every compared prefix identical. HTTP logprobs are
disallowed during replay since the sampler would report its own selected token.
"""

import functools
import json
from collections import Counter
from pathlib import Path

import torch


def install_output_logits_capture(runner, output_dir):
    args = runner.server_args
    if runner.tp_size != 1 or runner.pp_size != 1:
        raise ValueError("Output logits diagnostics require TP=PP=1")
    if not args.disable_cuda_graph or not args.disable_overlap_schedule:
        raise ValueError(
            "Output logits diagnostics require disabling CUDA graphs and overlap"
        )
    if args.speculative_algorithm is not None:
        raise ValueError(
            "Output logits diagnostics do not support speculative decoding"
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for name, module in runner.model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            inventory[name] = {
                "method": type(method).__name__,
                "parameters": {
                    n: {"shape": list(p.shape), "dtype": str(p.dtype)}
                    for n, p in module.named_parameters(recurse=False)
                },
            }
    (root / "inventory.json").write_text(
        json.dumps(
            {
                "quantization": args.quantization,
                "dtype": str(runner.dtype),
                "methods": dict(Counter(v["method"] for v in inventory.values())),
                "modules": inventory,
            },
            indent=2,
        )
        + "\n"
    )
    original = runner.sample
    counts = {}

    @functools.wraps(original)
    def captured_sample(logits_output, forward_batch):
        control = root / "control.json"
        if not control.exists():
            return original(logits_output, forward_batch)
        ctx = json.loads(control.read_text())
        if forward_batch.rids != [ctx["rid"]]:
            return original(logits_output, forward_batch)
        case = ctx["case_id"]
        if Path(case).name != case or case in {".", ".."}:
            raise ValueError("case_id must be a single safe path component")
        step = counts.get(ctx["rid"], 0)
        if step >= ctx["max_steps"]:
            raise ValueError("More sampled steps than the declared capture limit")
        if forward_batch.return_logprob:
            raise ValueError(
                "Use saved raw logits; HTTP logprobs are disabled in replay"
            )
        raw = logits_output.next_token_logits
        if raw.ndim != 2 or raw.shape[0] != 1:
            raise ValueError("Only one sequence per capture is supported")
        logits = raw.detach().float().cpu().clone()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite raw logits")
        natural = original(logits_output, forward_batch)
        forced = ctx.get("forced_token_ids")
        output = natural
        if forced is not None:
            token = forced[step]
            if not 0 <= token < raw.shape[-1]:
                raise ValueError("Forced token out of vocabulary")
            output = torch.full_like(natural, token)
        folder = root / case
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "logits": logits.squeeze(0),
            "step": step,
            "rid": ctx["rid"],
            "case_id": case,
            "mode": "replay" if forced is not None else "natural",
            "input_ids": forward_batch.input_ids.detach().cpu().clone(),
            "seq_lens": forward_batch.seq_lens.detach().cpu().clone(),
            "natural_token_id": int(natural.item()),
            "returned_token_id": int(output.item()),
            "forward_mode": str(forward_batch.forward_mode),
        }
        path = folder / f"{step:04d}.pt"
        if path.exists():
            raise ValueError(f"Refusing to overwrite logits capture {path}")
        temp = path.with_suffix(".tmp")
        torch.save(record, temp)
        temp.replace(path)
        counts[ctx["rid"]] = step + 1
        return output

    runner.sample = captured_sample
