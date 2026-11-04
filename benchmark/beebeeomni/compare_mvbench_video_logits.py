"""Score MVBench options with prefill-only logits under two frame-sampling policies.

This mirrors VeOmni's deterministic ``get_seq_frames`` and image sizing
without loading the VeOmni runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

import eval_precision as base


DEFAULT_REFERENCE = Path(
    "outputs/benchmarks/20261102_qwen27b_projector_ratio4_veomni_format/mvbench.jsonl"
)


def veomni_smart_resize(
    height: int, width: int, ratio: int, factor: int, image_factor: int
) -> tuple[int, int]:
    """Mirror VeOmni smart_resize with the caller's exact factor."""
    min_pixels = 64 * image_factor**2
    max_pixels = 1980 * image_factor**2

    def round_by_factor(value: float) -> int:
        return int(round(value / factor)) * factor

    def ceil_by_factor(value: float) -> int:
        return int(np.ceil(value / factor)) * factor

    h = max(factor, round_by_factor(height))
    w = max(factor, round_by_factor(width))
    if h * w > max_pixels:
        scale = ((height * width) / max_pixels) ** 0.5
        h = int(height / scale / factor) * factor
        w = int(width / scale / factor) * factor
    elif h * w < min_pixels:
        scale = (min_pixels / (height * width)) ** 0.5
        h = ceil_by_factor(height * scale)
        w = ceil_by_factor(width * scale)

    min_side = int(np.ceil(np.sqrt(ratio))) * image_factor
    if h < min_side:
        scale = min_side / float(h)
        h = ceil_by_factor(min_side)
        w = ceil_by_factor(w * scale)
    if w < min_side:
        scale = min_side / float(w)
        w = ceil_by_factor(min_side)
        h = ceil_by_factor(h * scale)
    return h, w


def veomni_tokens_per_frame(
    height: int, width: int, ratio: int, image_factor: int
) -> tuple[int, tuple[int, int]]:
    # This checkpoint uses patch_size=16 and merge_size=2, then adaptive pooling
    # with scale=downsample_ratio and a Qwen25VL temporal pair tokenization.
    # calculate_video_each_frame_token uses image_patch_size * merge_size.
    resized_h, resized_w = veomni_smart_resize(
        height, width, ratio, factor=image_factor, image_factor=image_factor
    )
    grid_h, grid_w = resized_h / image_factor, resized_w / image_factor
    pool_h = max(1, int(np.round(grid_h / np.sqrt(ratio))))
    pool_w = max(1, int(np.round(grid_w / np.sqrt(ratio))))
    return int(np.ceil(pool_h * pool_w / 2) + 2), (resized_w, resized_h)


def veomni_frame_indices(
    path: Path, sample: dict, max_frames: int, max_image_tokens: int,
    ratio: int, image_factor: int,
) -> tuple[list[int], float, int, int, tuple[int, int]]:
    meta = iio.immeta(path, index=None)
    fps = meta.get("fps", 24.0)
    duration = meta.get("duration", 0)
    frame_count = int(duration * fps) if duration else 0
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Invalid video metadata: fps={fps}, duration={duration}")

    start = int(fps * sample["start"]) if sample.get("start") is not None else 0
    end = int(fps * sample["end"]) if sample.get("end") is not None else frame_count
    valid_count = max(0, end - start)
    if valid_count == 0:
        start, end = 0, frame_count
        valid_count = end - start

    # VeOmni first caps frames by finetune_sample_frames and the remaining
    # image-token budget. Its evaluator passes the video's source resolution
    # through the same sizing calculation as the actual image preprocessing.
    first_frame = Image.fromarray(iio.imread(path, index=0))
    each_token, resized_size = veomni_tokens_per_frame(
        first_frame.height, first_frame.width, ratio, image_factor
    )
    desired = min(max_frames, max_image_tokens // each_token)
    if desired <= 0:
        raise ValueError("No video frames fit in the text-adjusted token budget")
    segment = float((valid_count - 1) / desired)
    indices = sorted(
        {
            min(
                (int(segment * i) + int(segment * (i + 1))) // 2 + start,
                end,
            )
            for i in range(desired)
        }
    )
    if len(indices) % 2:
        indices.pop()
    return indices, fps, frame_count, each_token, resized_size


def read_frames(path: Path, indices: list[int], args) -> list[str]:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    frames = reader.get_batch(indices).asnumpy()
    resize_args = SimpleNamespace(ratio=args.ratio, size_factor=args.size_factor)
    encoded = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        # Apply one resize policy to both frame-sampling variants so their
        # logit difference isolates temporal sampling.
        if args.resize_mode == "benchmark":
            image = base.resize(image, resize_args)
        elif args.resize_mode == "veomni":
            # preprocess_image calls qwen25vl_image_preprocess with
            # size_factor=image_factor, then multiplies by ceil(sqrt(ratio)).
            factor = args.image_factor * int(np.ceil(np.sqrt(args.ratio)))
            h, w = veomni_smart_resize(
                image.height, image.width, args.ratio, factor, args.image_factor
            )
            image = image.resize((w, h), Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    del reader
    return encoded


def _vision_prefix(num_images: int, pair_wrappers: bool) -> str:
    """Render the same visual wrappers used by VeOmni's video evaluator.

    VeOmni enables ``split_video_vision_by_fps`` with ``sample_fps=2``.  Since
    the model consumes two video frames per temporal patch, each adjacent
    image pair is enclosed by its own vision_start/vision_end block.  Keeping
    this in the text prompt matters for both token positions and 3-D M-RoPE.
    """
    if num_images <= 0:
        return ""
    if num_images % 2:
        raise ValueError(f"Video image count must be even, got {num_images}")
    if pair_wrappers:
        return "".join(
            "<|vision_start|><image><image><|vision_end|>"
            for _ in range(num_images // 2)
        )
    return "<|vision_start|>" + "<image>" * num_images + "<|vision_end|>"


SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
ANSWER_PREFIX = "My best option: ("


def question_prompt(sample: dict) -> str:
    # Match process_image_text_eval, including its first-option-only newline.
    question = sample["question"]
    for i, candidate in enumerate(sample["candidates"]):
        question += ("\n" if i == 0 else "") + f"({chr(65 + i)}) {candidate}\n"
    return (
        "\n<|im_start|>user\n" + question + "<|im_end|>"
        "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def answer_and_option_ids(sample: dict, tokenizer) -> tuple[str, list[int]]:
    candidates = sample["candidates"]
    answer = chr(65 + candidates.index(sample["answer"]))
    target = tokenizer.encode(ANSWER_PREFIX + answer, add_special_tokens=False)[-1]
    # Match evaluate_callback._build_option_maps, including its (A fallback.
    for prefix in ("", "("):
        ids = [tokenizer.encode(prefix + chr(65 + i))[-1] for i in range(len(candidates))]
        if target in ids:
            return answer, ids
    raise ValueError(f"Unsupported option target token: {target}")


def expanded_prompt_length(prompt: str, images: list[str], args, tokenizer) -> int:
    # Mirror BeeBee's _encode_and_expand_text chunk boundaries exactly.
    text_len = 0
    bos_token_id = tokenizer.bos_token_id
    for chunk in prompt.split("<image>"):
        if not chunk:
            continue
        ids = tokenizer.encode(chunk)
        if text_len and ids and ids[0] == bos_token_id:
            ids = ids[1:]
        text_len += len(ids)
    vision_len = 0
    for encoded in images[::2]:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            grid_h, grid_w = image.height / args.image_factor, image.width / args.image_factor
        pool_h = max(1, int(np.round(grid_h / np.sqrt(args.ratio))))
        pool_w = max(1, int(np.round(grid_w / np.sqrt(args.ratio))))
        vision_len += pool_h * pool_w
    return text_len + vision_len


def build_payload(sample: dict, images: list[str], args, tokenizer) -> tuple[dict, dict]:
    answer, option_ids = answer_and_option_ids(sample, tokenizer)
    prompt = SYSTEM + _vision_prefix(len(images), args.pair_wrappers)
    prompt += question_prompt(sample) + ANSWER_PREFIX
    # This prefix ends where VeOmni reads logits[:, -2] after appending the
    # one-token answer. Prefill-only next-token scores are the same logits.
    prompt_tokens = expanded_prompt_length(prompt, images, args, tokenizer)
    payload = {
        "text": prompt,
        "image_data": images,
        "image_downsample_ratios": [args.ratio] * (len(images) // 2),
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "token_ids_logprob": option_ids,
        "stream": False,
    }
    audit = {
        "prompt": prompt,
        "expected_prompt_tokens": prompt_tokens,
        "option_token_ids": option_ids,
        "target": answer,
        "image_count": len(images),
        "frame_sha256": [hashlib.sha256(base64.b64decode(x)).hexdigest() for x in images],
        "processed_frame_sizes_wh": [list(Image.open(io.BytesIO(base64.b64decode(x))).size) for x in images[::2]],
    }
    return payload, audit


def request_logprobs(url: str, payload: dict, audit: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1500]}") from exc
    meta = result["meta_info"]
    if meta.get("completion_tokens") != 0 or result.get("output_ids"):
        raise RuntimeError(f"Expected prefill-only response, got {meta}")
    if meta["prompt_tokens"] != audit["expected_prompt_tokens"]:
        raise RuntimeError(f"Expanded length mismatch: {meta['prompt_tokens']} vs {audit['expected_prompt_tokens']}")
    scores = meta["output_token_ids_logprobs"]
    if len(scores) != 1:
        raise RuntimeError(f"Expected one prefill next-token score row, got {scores}")
    scores = scores[0]
    if [item[1] for item in scores] != audit["option_token_ids"]:
        raise RuntimeError(f"Candidate token IDs mismatch: {scores}")
    logprobs = np.array([item[0] for item in scores], dtype=np.float64)
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()
    prediction = chr(65 + int(logprobs.argmax()))
    return {
        **audit,
        "option_logprobs": logprobs.tolist(),
        "option_probs": probs.tolist(),
        "prediction": prediction,
        "correct": prediction == audit["target"],
        "completion_tokens": meta["completion_tokens"],
        "prompt_tokens": meta["prompt_tokens"],
        "scored_hidden_position": meta["prompt_tokens"] - 1,
        "veomni_equivalent_readout": "logits[0, -2] with one answer token appended",
        "cached_tokens": meta.get("cached_tokens"),
    }


def load_reference(path: Path) -> dict[int, dict]:
    return {
        int(row["sample_id"]): row
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("benchmark/beebeeomni"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/sglang_beebee_benchmark_cache"))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mvbench_video_logits.jsonl"))
    parser.add_argument("--url", default="http://127.0.0.1:18005")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--size-factor", type=int, default=32)
    parser.add_argument("--image-factor", type=int, default=32)
    parser.add_argument("--model-path", type=Path, default=Path(
        "/mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base_qwen38_projector"
    ))
    parser.add_argument("--model-max-length", type=int, default=12800)
    parser.add_argument(
        "--resize-mode", choices=("veomni", "benchmark"), default="veomni"
    )
    parser.add_argument(
        "--pair-wrappers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap each adjacent video frame pair in its own vision block (VeOmni default).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    samples = base.data_file(args.data / base.DATASETS["mvbench"][0])
    by_id = {int(sample["id"]): sample for sample in samples}
    references = load_reference(args.reference)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for sample_id in args.ids:
            sample = by_id[sample_id]
            reference = references[sample_id]
            ref_media = reference["media_info"][0]
            original = Path(ref_media["path"])
            clip = Path(ref_media["request_path"])
            sg_indices = ref_media["frame_indices"]
            answer, _ = answer_and_option_ids(sample, tokenizer)
            # VeOmni tokenizes system, query and answer independently.
            text_tokens = (
                len(tokenizer.encode(SYSTEM))
                + len(tokenizer.encode(question_prompt(sample)))
                + len(tokenizer.encode(ANSWER_PREFIX + answer, add_special_tokens=False))
            )
            max_image_tokens = args.model_max_length - text_tokens
            veo_indices, source_fps, source_frames, each_token, veomni_size = veomni_frame_indices(
                original, sample, args.frames, max_image_tokens, args.ratio,
                args.image_factor,
            )
            pair = {
                "sample_id": sample_id,
                "sglang_reference": {
                    "path": str(clip),
                    "frame_indices": sg_indices,
                    "images": read_frames(clip, sg_indices, args),
                },
                "veomni_sampling": {
                    "path": str(original),
                    "fps": source_fps,
                    "estimated_source_frames": source_frames,
                    "frame_indices": veo_indices,
                    "images": read_frames(original, veo_indices, args),
                },
            }
            row = {
                "sample_id": sample_id,
                "sglang_frame_indices": sg_indices,
                "veomni_frame_indices": veo_indices,
                "source_fps": source_fps,
                "source_frames_estimated": source_frames,
                "text_tokens_for_budget": text_tokens,
                "max_image_tokens": max_image_tokens,
                "veomni_estimated_tokens_per_frame": each_token,
                "veomni_budget_frame_size_wh": list(veomni_size),
                "image_factor": args.image_factor,
                "downsample_ratio": args.ratio,
                "model_path": str(args.model_path),
                "pair_wrappers": args.pair_wrappers,
                "image_transport": "lossless_png",
            }
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            **row,
                            "sglang_image_count": len(pair["sglang_reference"]["images"]),
                            "veomni_image_count": len(pair["veomni_sampling"]["images"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            for key in ("sglang_reference", "veomni_sampling"):
                payload, audit = build_payload(sample, pair[key]["images"], args, tokenizer)
                row[key + "_logprobs"] = request_logprobs(args.url, payload, audit)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"sample_id={sample_id}: SGLang={len(sg_indices)} frames, "
                f"VeOmni sampling={len(veo_indices)} frames",
                flush=True,
            )


if __name__ == "__main__":
    main()
