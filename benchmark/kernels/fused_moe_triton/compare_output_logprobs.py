"""Full-vocabulary, same-prefix comparison of BF16, FP8 and selective BF16.

Run from the repository root. Start one server at a time with TP=PP=1,
SGLANG_OUTPUT_LOGITS_DIR=OUT/VARIANT, --disable-cuda-graph,
--disable-overlap-schedule, --disable-radix-cache, --skip-server-warmup.
For VARIANT=bf16, use SGLANG_QUANTIZATION=none with launch_fp8.sh; fp8 is
its default. For fp8_skip_fc2 additionally set SGLANG_FP8_IGNORED_LAYERS to
image_encoder.blocks.26.mlp.linear_fc2. Use identical remaining server args.

  /usr/bin/python3 benchmark/kernels/fused_moe_triton/compare_output_logprobs.py \
    collect --root fp8_analysis/output_logits_v1 --variant bf16
  # Restart the server as fp8, then fp8_skip_fc2 and repeat collect.
  /usr/bin/python3 benchmark/kernels/fused_moe_triton/compare_output_logprobs.py \
    analyze --root fp8_analysis/output_logits_v1 --model /path/to/checkpoint

Uses test1.jpg for description/OCR and 16 video frames for description.
Reference greedy tokens are replayed at every step in both FP8 variants.
A second natural-generation request is recorded for each FP8 variant/case.
Reports full-vocabulary KL/JS/TV at temperature=1 (not truncated top-20 KL),
top-20 IDs, probabilities, ranks, shared mass and reference-token likelihood.
This measures output fidelity, not task accuracy or serving throughput.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from api_test import construct_prompt, smart_resize

VARIANTS = ("bf16", "fp8", "fp8_skip_fc2")
TARGET = "image_encoder.blocks.26.mlp.linear_fc2"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def load_records(folder):
    records = [
        torch.load(p, map_location="cpu", weights_only=True)
        for p in sorted(folder.glob("*.pt"))
    ]
    if not records:
        raise ValueError(f"No raw logits in {folder}")
    assert [r["step"] for r in records] == list(range(len(records)))
    return records


def make_cases(assets):
    with Image.open(assets / "test1.jpg") as image:
        still = image.convert("RGB")
    path = assets / "test_video.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    total = 0
    while cap.grab():
        total += 1
    cap.release()
    if total < 16:
        raise ValueError("Need at least 16 decodable video frames")
    indices = np.linspace(0, total - 1, 16, dtype=int).tolist()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode video frame {idx}")
        video.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    definitions = [
        ("image_describe", [still, still], "描述这张图片的内容。", "test1.jpg", {}),
        (
            "image_ocr",
            [still, still],
            "识别并抄录这张图片中的所有文字，没有文字请说明。",
            "test1.jpg",
            {},
        ),
        (
            "video_describe",
            video,
            "按时间顺序描述这段视频的内容和主要变化。",
            path.name,
            {"frame_indices": indices, "decoded_frame_count": total, "fps": fps},
        ),
    ]
    cases = []
    for name, frames, query, asset, extra in definitions:
        h, w = smart_resize(
            frames[0].height, frames[0].width, factor=128, max_pixels=1024**2
        )
        encoded = []
        for frame in frames:
            out = io.BytesIO()
            frame.resize((w, h), Image.Resampling.BICUBIC).save(out, format="PNG")
            encoded.append(base64.b64encode(out.getvalue()).decode("ascii"))
        payload = {
            "text": construct_prompt(query, len(frames)),
            "image_data": encoded,
            "image_downsample_ratios": [4] * (len(frames) // 2),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        cases.append(
            (
                name,
                payload,
                {
                    "asset": str((assets / asset).resolve()),
                    "query": query,
                    "size": [w, h],
                    "frames": len(frames),
                    "input_sha256": digest,
                    **extra,
                },
            )
        )
    return cases


def collect(args):
    root = args.root.resolve()
    folder = root / args.variant
    inventory = read_json(folder / "inventory.json")
    target = inventory["modules"][TARGET]
    expected = "Fp8LinearMethod" if args.variant == "fp8" else "UnquantizedLinearMethod"
    assert target["method"] == expected, target
    assert target["parameters"]["weight"]["dtype"] == (
        "torch.float8_e4m3fn" if args.variant == "fp8" else "torch.bfloat16"
    )
    if args.variant == "bf16":
        assert inventory["quantization"] is None
        assert not any("Fp8" in k for k in inventory["methods"])
    else:
        assert inventory["methods"]["Fp8LinearMethod"] == (
            206 if args.variant == "fp8" else 205
        )
        assert inventory["methods"]["Fp8MoEMethod"] == 48
    print(args.variant, inventory["methods"], flush=True)
    for case, payload, meta in make_cases(args.assets):
        kinds = ("natural",) if args.variant == "bf16" else ("replay", "natural")
        if args.repeat:
            if args.variant != "bf16":
                raise ValueError("Repeat check is only for the BF16 reference")
            kinds = ("repeat",)
        for kind in kinds:
            name = f"{case}_{kind}"
            if (folder / name).exists() or (folder / f"{name}.json").exists():
                raise ValueError(f"Case exists: {folder/name}; use a new directory")
            rid = f"logits-{args.variant}-{uuid.uuid4().hex}"
            control = {"case_id": name, "rid": rid, "max_steps": args.steps}
            reference = None
            if kind in {"replay", "repeat"}:
                reference = load_records(root / "bf16" / f"{case}_natural")
                ref_response = read_json(root / "bf16" / f"{case}_natural.json")
                assert ref_response["input_sha256"] == meta["input_sha256"]
                control["forced_token_ids"] = [
                    r["returned_token_id"] for r in reference
                ]
                control["max_steps"] = len(reference)
            request_payload = dict(
                payload,
                rid=rid,
                stream=False,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": control["max_steps"],
                    "stop": "<|im_end|>",
                },
            )
            write_json(folder / "control.json", control)
            try:
                response = requests.post(
                    args.url.rstrip("/") + "/generate",
                    json=request_payload,
                    timeout=600,
                )
                response.raise_for_status()
                output = response.json()
                finish = output["meta_info"].get("finish_reason")
                if isinstance(finish, dict) and finish.get("type") == "abort":
                    raise RuntimeError(output)
                if not output.get("text", "").strip():
                    raise ValueError(f"Empty generation: {output}")
                records = load_records(folder / name)
                assert all(r["rid"] == rid for r in records)
                assert len(records) == output["meta_info"]["completion_tokens"]
                if reference is not None:
                    assert len(records) == len(reference)
                    for i, (a, b) in enumerate(zip(reference, records)):
                        assert a["returned_token_id"] == b["returned_token_id"]
                        assert torch.equal(a["seq_lens"], b["seq_lens"])
                        assert torch.equal(a["input_ids"], b["input_ids"])
                write_json(
                    folder / f"{name}.json",
                    dict(
                        meta,
                        case_id=name,
                        variant=args.variant,
                        steps=len(records),
                        sampling_params=request_payload["sampling_params"],
                        response=output,
                    ),
                )
                print(
                    args.variant,
                    name,
                    f"steps={len(records)}",
                    output["text"],
                    flush=True,
                )
            finally:
                (folder / "control.json").unlink(missing_ok=True)


def distribution_metrics(ref_logits, cand_logits, token_id):
    if ref_logits.shape != cand_logits.shape or ref_logits.ndim != 1:
        raise ValueError("Expected matching full-vocabulary logit vectors")
    if not torch.isfinite(ref_logits).all() or not torch.isfinite(cand_logits).all():
        raise ValueError("Nonfinite logits")
    logp = ref_logits.double().log_softmax(-1)
    logq = cand_logits.double().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    logm = torch.logaddexp(logp, logq) - np.log(2)
    # BF16 logits can tie. Stable ordering uses ascending token ID for ties,
    # matching argmax at rank 1 and avoiding arbitrary topk tie permutations.
    ri = torch.argsort(logp, descending=True, stable=True)[:20]
    qi = torch.argsort(logq, descending=True, stable=True)[:20]
    rp, qp = p[ri], q[qi]
    ri, qi = ri.tolist(), qi.tolist()
    common = set(ri) & set(qi)
    return {
        "kl_bf16_to_candidate": max(0.0, (p * (logp - logq)).sum().item()),
        "kl_candidate_to_bf16": max(0.0, (q * (logq - logp)).sum().item()),
        "js_divergence": max(
            0.0, (0.5 * ((p * (logp - logm)).sum() + (q * (logq - logm)).sum())).item()
        ),
        "total_variation": (0.5 * (p - q).abs().sum()).item(),
        "max_probability_delta": (p - q).abs().max().item(),
        "top1_same": ri[0] == qi[0],
        "top20_overlap": len(common),
        "top20_exact_order": ri == qi,
        "common_rank_shift_mean": (
            float(np.mean([abs(ri.index(t) - qi.index(t)) for t in common]))
            if common
            else None
        ),
        "bf16_top20_mass": rp.sum().item(),
        "candidate_top20_mass": qp.sum().item(),
        "candidate_mass_on_bf16_top20": q[ri].sum().item(),
        "bf16_top1_top2_margin": (rp[0] - rp[1]).item(),
        "candidate_top1_top2_margin": (qp[0] - qp[1]).item(),
        "bf16_reference_token_probability": p[token_id].item(),
        "candidate_reference_token_probability": q[token_id].item(),
        "reference_token_nll_delta": (logp[token_id] - logq[token_id]).item(),
        "bf16_top20_ids": ri,
        "candidate_top20_ids": qi,
        "bf16_top20_probabilities": rp.tolist(),
        "candidate_top20_probabilities": qp.tolist(),
        "candidate_probability_on_bf16_top20": q[ri].tolist(),
        "bf16_probability_on_candidate_top20": p[qi].tolist(),
    }


def aggregate(rows):
    result = {"positions": len(rows)}
    for k in (
        "kl_bf16_to_candidate",
        "kl_candidate_to_bf16",
        "js_divergence",
        "total_variation",
        "max_probability_delta",
        "top20_overlap",
        "reference_token_nll_delta",
    ):
        vals = np.array([r[k] for r in rows])
        result[k] = {
            "mean": float(vals.mean()),
            "p50": float(np.median(vals)),
            "p95": float(np.quantile(vals, 0.95)),
            "max": float(vals.max()),
        }
    result["top1_agreement"] = sum(r["top1_same"] for r in rows) / len(rows)
    result["top20_exact_order_rate"] = sum(r["top20_exact_order"] for r in rows) / len(
        rows
    )
    return result


def analyze(args):
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = {
        "note": "Full-vocabulary temperature-1 probabilities on identical BF16 token prefixes; natural outputs separate.",
        "cases": {},
        "overall": {},
        "bf16_repeatability": {},
        "top20_tie_rule": "Descending raw logits, ascending token ID for equal logits",
    }
    all_rows = {v: [] for v in VARIANTS[1:]}
    ref_dirs = sorted((args.root / "bf16").glob("*_natural"))
    assert {p.name for p in ref_dirs} == {
        "image_describe_natural",
        "image_ocr_natural",
        "video_describe_natural",
    }
    for ref_dir in ref_dirs:
        case = ref_dir.name.removesuffix("_natural")
        reference = load_records(ref_dir)
        ref_meta = read_json(ref_dir.with_suffix(".json"))
        out = {
            "metadata": {k: v for k, v in ref_meta.items() if k != "response"},
            "reference_text": ref_meta["response"]["text"],
            "comparisons": {},
        }
        repeated = load_records(args.root / "bf16" / f"{case}_repeat")
        assert len(repeated) == len(reference)
        repeat_rows = [
            distribution_metrics(a["logits"], b["logits"], a["returned_token_id"])
            for a, b in zip(reference, repeated)
        ]
        report["bf16_repeatability"][case] = dict(
            aggregate(repeat_rows),
            exact_logit_equal_steps=sum(
                torch.equal(a["logits"], b["logits"])
                for a, b in zip(reference, repeated)
            ),
        )
        for variant in VARIANTS[1:]:
            records = load_records(args.root / variant / f"{case}_replay")
            assert len(records) == len(reference)
            replay_meta = read_json(args.root / variant / f"{case}_replay.json")
            assert replay_meta["input_sha256"] == ref_meta["input_sha256"]
            rows = []
            for i, (ref, cand) in enumerate(zip(reference, records)):
                assert ref["step"] == cand["step"] and torch.equal(
                    ref["seq_lens"], cand["seq_lens"]
                )
                assert ref["returned_token_id"] == cand["returned_token_id"]
                assert torch.equal(ref["input_ids"], cand["input_ids"])
                if i:
                    assert ref["input_ids"].tolist() == [
                        reference[i - 1]["returned_token_id"]
                    ]
                assert ref["natural_token_id"] == ref["logits"].argmax().item()
                assert cand["natural_token_id"] == cand["logits"].argmax().item()
                r = distribution_metrics(
                    ref["logits"], cand["logits"], ref["returned_token_id"]
                )
                r.update(
                    step=i,
                    reference_token_id=ref["returned_token_id"],
                    reference_token_text=tokenizer.decode([ref["returned_token_id"]]),
                )
                for name in ("bf16", "candidate"):
                    r[name + "_top20_text"] = [
                        tokenizer.decode([t]) for t in r[name + "_top20_ids"]
                    ]
                rows.append(r)
            natural = load_records(args.root / variant / f"{case}_natural")
            natural_meta = read_json(args.root / variant / f"{case}_natural.json")
            first_diff = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(reference, natural))
                    if a["returned_token_id"] != b["returned_token_id"]
                ),
                None,
            )
            if first_diff is None and len(natural) != len(reference):
                first_diff = min(len(natural), len(reference))
            out["comparisons"][variant] = {
                "summary": aggregate(rows),
                "positions": rows,
                "natural_text": natural_meta["response"]["text"],
                "natural_steps": len(natural),
                "natural_first_divergence_step": first_diff,
            }
            all_rows[variant].extend(rows)
        current = out["comparisons"]["fp8"]["positions"]
        skip = out["comparisons"]["fp8_skip_fc2"]["positions"]
        out["skip_vs_fp8"] = {
            "kl_closer_to_bf16_positions": sum(
                b["kl_bf16_to_candidate"] < a["kl_bf16_to_candidate"]
                for a, b in zip(current, skip)
            ),
            "top1_fixed": sum(
                not a["top1_same"] and b["top1_same"] for a, b in zip(current, skip)
            ),
            "top1_regressed": sum(
                a["top1_same"] and not b["top1_same"] for a, b in zip(current, skip)
            ),
        }
        report["cases"][case] = out
    for variant, rows in all_rows.items():
        report["overall"][variant] = aggregate(rows)
    write_json(args.root / "comparison.json", report)
    print(json.dumps(report["overall"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18004")
    p.add_argument("--assets", type=Path, default=Path("examples/assets"))
    p.add_argument("--steps", type=int, default=96)
    p.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat BF16 on its reference prefix to measure numerical noise",
    )
    p = sub.add_parser("analyze")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
