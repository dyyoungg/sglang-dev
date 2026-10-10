"""Paired, resumable BF16/FP8 business evaluation with identical media payloads.

Run from repository root; Python deps: aiohttp, jiwer==4.0.0, rapidfuzz, Pillow,
AOSS client, soundfile, numpy, decord. Example:
  PYTHONPATH=/tmp/sglang_beebee_eval_deps_mirror /usr/bin/python3 -u \
    benchmark/beebeeomni/eval_precision.py \
    --output fp8_analysis/business_0815_smoke --limit 1
  # Omit --limit for all samples. Restart the identical command to resume.
  # --retry-errors reruns incomplete/error pairs; successes are not duplicated.
  # LibriSpeech uses English WER with reference_protocol normalization:
  # --datasets librispeech_dev_clean librispeech_dev_other \
  #            librispeech_test_clean librispeech_test_other
  # Summaries store corpus_wer (total word errors / total reference words);
  # mean_score is the unweighted average of utterance WERs.

Two DP2/TP2 servers: bf16=http://127.0.0.1:18100, fp8=:18101 (4 GPUs each).
Prompt/scoring functions are frozen in reference_protocol.py from eval_sglang.py.
Original 64-frame, ratio=1, size_factor=32 resizing retained; JPEG encoding.
Default output budgets: ASR 1024, OCR generation 4096, MCQ 64 with ')' stop.
The MCQ stop only shortens continuation after the answer option. Both variants
use identical temperature=0, repetition_penalty=1.05, input, and token budgets.
No labels, old predictions, or scoring metadata are sent to either model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

import aiohttp
import jiwer
import numpy as np
import soundfile as sf
from decord import VideoReader, cpu
from PIL import Image
from rapidfuzz.distance import Levenshtein

import reference_protocol as protocol

DATASETS = {
    "aishell": ("aishell_test.aoss.json", "generate", "cer"),
    "wenet": ("wenet_test.aoss.json", "generate", "cer"),
    "online_badaudio": ("online_badaudio.aoss.json", "generate", "cer"),
    "librispeech_dev_clean": ("librispeech_dev_clean_new.aoss.json", "generate", "wer"),
    "librispeech_dev_other": ("librispeech_dev_other_new.aoss.json", "generate", "wer"),
    "librispeech_test_clean": ("librispeech_test_clean_new.aoss.json", "generate", "wer"),
    "librispeech_test_other": ("librispeech_test_other_new.aoss.json", "generate", "wer"),
    "mmstar": ("mmstar_test.aoss.json", "mcq", "accuracy"),
    "RealWorldQA": ("RealWorldQA_test.aoss.json", "mcq", "accuracy"),
    "ocr": ("ocr_eval.json", "mcq", "accuracy"),
    "ocr_generate": ("ocr_intense_val_generate.jsonl", "generate", "similarity"),
    "chartqa": ("chartqa_test_final.aoss.json", "mcq", "accuracy"),
    "mvbench": ("mvbench_test.json", "mcq", "accuracy"),
    "videomme": ("videomme_test.json", "mcq", "accuracy"),
    "videommmu": ("videommmu_test.json", "mcq", "accuracy"),
}
LOCAL = threading.local()
LOCKS = {}
LOCK_GUARD = threading.Lock()
# Keep the normalization order identical to reference_protocol.calculate_wer.
WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def data_file(path):
    with path.open() as f:
        return (
            [json.loads(line) for line in f if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(f)
        )


def get_client():
    if not hasattr(LOCAL, "client"):
        from aoss_client.client import Client

        LOCAL.client = Client(
            os.getenv("AOSS_FILE", "/mnt/afs/yangdeyu/aoss_ydy_game.conf")
        )
    return LOCAL.client


def cache_media(uri, cache):
    if not uri.startswith("s3://"):
        if not Path(uri).is_file():
            raise FileNotFoundError(uri)
        return Path(uri)
    key = hashlib.sha256(uri.encode()).hexdigest()
    path = cache / "media" / f"{key}{Path(uri).suffix}"
    with LOCK_GUARD:
        lock = LOCKS.setdefault(key, threading.Lock())
    with lock:
        if path.exists() and path.stat().st_size:
            return path
        raw = None
        for attempt in range(3):
            try:
                raw = get_client().Get(uri)
                if not raw:
                    raise ValueError("Empty AOSS object")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(raw)
        temp.replace(path)
    return path


def resize(image, args):
    image = protocol.to_rgb(image)
    h, w = protocol.smart_resize(
        image.height,
        image.width,
        factor=args.size_factor * math.ceil(math.sqrt(args.ratio)),
        min_pixels=64 * args.size_factor**2,
        max_pixels=1980 * args.size_factor**2,
        min_side=112,
    )
    return image.resize((w, h), Image.Resampling.BICUBIC)


def jpeg(image):
    f = io.BytesIO()
    image.save(f, format="JPEG")
    return base64.b64encode(f.getvalue()).decode("ascii")


def prepare(dataset, index, sample, args):
    start = time.time()
    _, task, metric = DATASETS[dataset]
    media, uri = protocol.detect_media_type(sample)
    if media == "unknown":
        raise ValueError(f"Unknown media for sample {index}")
    language = sample.get("language", "en")
    question, gt = protocol.prepare_question(sample, task, media, language)
    if task == "mcq" and (len(gt) != 1 or gt not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError(f"MCQ ground truth is not an option letter: {gt!r}")
    system = sample.get(
        "system", sample.get("system_prompt", "You are a helpful assistant.")
    )
    payload = {}
    info = {}
    images = []
    if media == "audio":
        path = cache_media(uri, args.cache)
        payload["audio_data"] = base64.b64encode(path.read_bytes()).decode("ascii")
        audio = sf.info(str(path))
        info = {"audio_seconds": audio.duration, "sample_rate": audio.samplerate}
    elif media == "image":
        for source in ([uri] if isinstance(uri, str) else uri):
            path = cache_media(source, args.cache)
            with Image.open(path) as im:
                im = resize(im, args)
            encoded = jpeg(im)
            images.extend([encoded, encoded])
            info.setdefault("sizes", []).append(list(im.size))
    else:
        path = cache_media(uri, args.cache)
        if path.suffix.lower() == ".gif":
            with Image.open(path) as im:
                count = im.n_frames
                indices = frame_indices(count, 4, sample, args.frames)
                for idx in indices:
                    im.seek(idx)
                    frame = resize(im.copy(), args)
                    images.append(jpeg(frame))
                    info.setdefault("sizes", []).append(list(frame.size))
        else:
            vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            count = len(vr)
            fps = vr.get_avg_fps()
            indices = frame_indices(count, fps, sample, args.frames)
            # Decode in small batches to avoid materializing 64 full-resolution frames.
            for offset in range(0, len(indices), 4):
                frames = vr.get_batch(indices[offset : offset + 4]).asnumpy()
                for frame in frames:
                    im = resize(Image.fromarray(frame), args)
                    images.append(jpeg(im))
                    info.setdefault("sizes", []).append(list(im.size))
            del vr
            info["fps"] = fps
        if len(images) % 2:
            images.pop()
            indices.pop()
            info["sizes"].pop()
        if len(images) < 2:
            raise ValueError("Fewer than 2 usable video frames")
        info.update(frame_indices=indices, total_frames=count)
    if images:
        payload["image_data"] = images
        payload["image_downsample_ratios"] = [args.ratio] * (len(images) // 2)
    prompt = protocol.build_prompt(question, media, task, len(images), language, system)
    if task == "mcq":
        prompt += "My best option: ("
    max_tokens = (
        args.mcq_tokens
        if task == "mcq"
        else args.asr_tokens if media == "audio" else args.ocr_tokens
    )
    sampling = {
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "max_new_tokens": max_tokens,
        "repetition_penalty": 1.05,
        "stop": ["<|im_end|>", ")"] if task == "mcq" else ["<|im_end|>"],
    }
    payload.update(text=prompt, sampling_params=sampling, stream=False)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    row = {
        "dataset": dataset,
        "index": index,
        "sample_id": sample.get("id", index),
        "media_type": media,
        "media_path": uri,
        "task": task,
        "metric": metric,
        "gt": gt,
        "question": question,
        "category": sample.get("category", "unknown"),
        "subtask": sample.get("category_abbr", sample.get("task", "")),
        "input_sha256": digest,
        "media_info": info,
        "prompt": prompt,
        "sampling_params": sampling,
        "prepare_seconds": time.time() - start,
    }
    return payload, row


def frame_indices(count, fps, sample, n):
    if count <= 0 or fps <= 0:
        raise ValueError("Invalid video metadata")
    start = (
        max(0, int(float(sample["start"]) * fps))
        if sample.get("start") is not None
        else 0
    )
    end = (
        min(count, int(float(sample["end"]) * fps))
        if sample.get("end") is not None
        else count
    )
    if end <= start:
        raise ValueError(f"Invalid video segment {start}:{end}/{count}")
    desired = min(n, end - start)
    segment = (end - start - 1) / desired
    return sorted(
        set(
            min((int(segment * i) + int(segment * (i + 1))) // 2 + start, end - 1)
            for i in range(desired)
        )
    )


def score(row, raw):
    pred = (
        protocol.extract_mcq_answer(raw.strip())
        if row["task"] == "mcq"
        else raw.strip().replace("The answer is ", "").replace(".", "").strip()
    )
    gt = row["gt"]
    metric = row["metric"]
    extra = {}
    if metric == "cer":
        ref = protocol._mixed_language_chars([gt])[0]
        hyp = protocol._mixed_language_chars([pred])[0]
        err = Levenshtein.distance(ref, hyp)
        # Keep standard CER as the primary score (lower is better), and also
        # report the training-style normalized edit similarity (higher is
        # better) using max(reference, hypothesis) as the denominator.
        value = err / len(ref) if ref else float(err)
        norm_distance = err / max(len(ref), len(hyp), 1)
        extra = {
            "edit_errors": err,
            "reference_units": len(ref),
            "normalized_edit_distance": norm_distance,
            "similarity": 1.0 - norm_distance,
        }
    elif metric == "wer":
        result = jiwer.process_words(
            gt,
            pred,
            reference_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM,
        )
        value = result.wer
        extra = {
            "edit_errors": result.substitutions + result.deletions + result.insertions,
            "reference_units": sum(len(words) for words in result.references),
            "reference_unit_type": "word",
            "substitutions": result.substitutions,
            "deletions": result.deletions,
            "insertions": result.insertions,
        }
    elif metric == "accuracy":
        value = float(gt.strip() == pred.strip())
    else:
        # Match VeOmni EvaluateCallback._eval_llm_batch exactly:
        # lowercase, retain only latin letters/digits/CJK/Hiragana/Katakana,
        # compare against the GT prefix of hypothesis length + five chars,
        # and report the Levenshtein complement as similarity (higher better).
        gt_cleaned = clean_training_text(gt)
        hyp_cleaned = clean_training_text(pred)
        gt_sliced = gt_cleaned[: len(hyp_cleaned) + 5]
        distance = Levenshtein.distance(gt_sliced, hyp_cleaned)
        similarity = 1.0 - (distance / max(len(gt_sliced), len(hyp_cleaned), 1))
        value = similarity
        extra = {
            "similarity": similarity,
            "normalized_edit_distance": 1.0 - similarity,
            "cleaned_gt_length": len(gt_cleaned),
            "cleaned_gt_scored_length": len(gt_sliced),
            "cleaned_pred_length": len(hyp_cleaned),
        }
    return dict(pred=pred, score=value, **extra)


def clean_training_text(s: str) -> str:
    """Copy VeOmni EvaluateCallback.clean_text for generation scoring."""
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", "", s)


async def request(session, url, payload, row):
    t = time.monotonic()
    for attempt in range(3):
        try:
            async with session.post(
                url.rstrip("/") + "/generate", json=payload
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {str(body)[:700]}")
            meta = body.get("meta_info", {})
            finish = meta.get("finish_reason", {})
            if isinstance(finish, dict) and finish.get("type") == "abort":
                raise RuntimeError(str(finish))
            if "text" not in body:
                raise RuntimeError(f"Missing generation: {str(body)[:500]}")
            raw = body["text"]
            return dict(
                status="success",
                raw_answer=raw,
                **score(row, raw),
                meta_info=meta,
                elapsed_seconds=time.monotonic() - t,
                attempts=attempt + 1,
            )
        except Exception as e:
            if attempt == 2:
                return dict(
                    status="error",
                    error=str(e)[:1000],
                    elapsed_seconds=time.monotonic() - t,
                    attempts=3,
                )
            await asyncio.sleep(1 + attempt)


def summary(rows, total):
    result = {
        "total": total,
        "recorded": len(rows),
        "paired_success": 0,
        "variants": {},
        "errors": [],
    }
    paired = [
        r
        for r in rows
        if all(
            r.get("outputs", {}).get(v, {}).get("status") == "success"
            for v in ["bf16", "fp8"]
        )
    ]
    result["paired_success"] = len(paired)
    result["same_prediction"] = sum(
        r["outputs"]["bf16"]["pred"] == r["outputs"]["fp8"]["pred"] for r in paired
    )
    for v in ["bf16", "fp8"]:
        out = [
            r["outputs"][v]
            for r in rows
            if r.get("outputs", {}).get(v, {}).get("status") == "success"
        ]
        po = [r["outputs"][v] for r in paired]
        st = {
            "success": len(out),
            "score_direction": (
                "lower_is_better"
                if out and "edit_errors" in out[0]
                else "higher_is_better"
            ),
            "mean_score": float(np.mean([x["score"] for x in out])) if out else None,
            "mean_similarity": (
                float(np.mean([x["similarity"] for x in out]))
                if out and "similarity" in out[0]
                else None
            ),
            "paired_mean_score": (
                float(np.mean([x["score"] for x in po])) if po else None
            ),
            "paired_mean_similarity": (
                float(np.mean([x["similarity"] for x in po]))
                if po and "similarity" in po[0]
                else None
            ),
            "length_stopped": sum(
                x.get("meta_info", {}).get("finish_reason", {}).get("type") == "length"
                for x in out
            ),
        }
        if out and "reference_units" in out[0]:
            corpus_metric = (
                "corpus_wer"
                if out[0].get("reference_unit_type") == "word"
                else "corpus_cer"
            )
            st[corpus_metric] = sum(x["edit_errors"] for x in out) / max(
                sum(x["reference_units"] for x in out), 1
            )
        result["variants"][v] = st
    if paired:
        delta = np.array(
            [
                r["outputs"]["fp8"]["score"] - r["outputs"]["bf16"]["score"]
                for r in paired
            ]
        )
        rng = np.random.default_rng(42)
        # Paired bootstrap for a descriptive interval, not a quality acceptance threshold.
        means = [
            float(rng.choice(delta, size=len(delta), replace=True).mean())
            for _ in range(1000)
        ]
        result["paired_delta_fp8_minus_bf16"] = float(delta.mean())
        result["paired_delta_bootstrap_95pct"] = np.quantile(
            means, [0.025, 0.975]
        ).tolist()
        result["fp8_score_higher"] = int((delta > 0).sum())
        result["fp8_score_lower"] = int((delta < 0).sum())
    paired_indices = {r["index"] for r in paired}
    for r in rows:
        if r["index"] not in paired_indices and len(result["errors"]) < 10:
            result["errors"].append(
                {
                    "index": r["index"],
                    "error": r.get("error"),
                    "outputs": r.get("outputs"),
                }
            )
    return result


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    datasets = {}
    manifest = {}
    for name in args.datasets:
        file = args.data / DATASETS[name][0]
        data = data_file(file)
        if args.limit:
            data = data[: args.limit]
        datasets[name] = data
        manifest[name] = {
            "path": str(file.resolve()),
            "samples": len(data),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    config = {
        "datasets": manifest,
        "bf16_url": args.bf16_url,
        "fp8_url": args.fp8_url,
        "frames": args.frames,
        "ratio": args.ratio,
        "size_factor": args.size_factor,
        "max_tokens": {
            "asr": args.asr_tokens,
            "mcq": args.mcq_tokens,
            "ocr": args.ocr_tokens,
        },
        "temperature": 0,
        "repetition_penalty": 1.05,
        "protocol_sha256": hashlib.sha256(
            Path(protocol.__file__).read_bytes()
        ).hexdigest(),
    }
    config_file = args.output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Resume configuration mismatch; use a new output directory")
    atomic_json(config_file, config)
    timeout = aiohttp.ClientTimeout(total=1800, connect=30)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=aiohttp.TCPConnector(
            limit=max(args.concurrency, args.image_concurrency, args.video_concurrency)
            * 2
            + 4
        ),
    ) as session:
        for name, data in datasets.items():
            path = args.output / f"{name}.jsonl"
            saved = {}
            if path.exists():
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            saved[r["index"]] = r
            pending = [
                (i, s)
                for i, s in enumerate(data)
                if i not in saved
                or (
                    args.retry_errors
                    and not all(
                        saved[i].get("outputs", {}).get(v, {}).get("status")
                        == "success"
                        for v in ["bf16", "fp8"]
                    )
                )
            ]
            print(f"{name}: {len(data)} samples, {len(pending)} pending", flush=True)
            queue = asyncio.Queue()
            for item in pending:
                queue.put_nowait(item)
            started = time.monotonic()
            completed = 0
            if name in ["mvbench", "videomme", "videommmu"]:
                concurrency = args.video_concurrency
            elif DATASETS[name][2] in {"cer", "wer"}:
                concurrency = args.concurrency
            else:
                concurrency = args.image_concurrency
            concurrency = min(concurrency, max(len(pending), 1))
            print(f"{name}: paired request concurrency={concurrency}", flush=True)

            async def worker():
                nonlocal completed
                while not queue.empty():
                    try:
                        i, s = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        payload, row = await asyncio.to_thread(
                            prepare, name, i, s, args
                        )
                        # Same payload object is sent to both precision services.
                        a, b = await asyncio.gather(
                            request(session, args.bf16_url, payload, row),
                            request(session, args.fp8_url, payload, row),
                        )
                        row["outputs"] = {"bf16": a, "fp8": b}
                        del payload
                    except Exception as e:
                        row = {
                            "dataset": name,
                            "index": i,
                            "status": "prepare_error",
                            "error": str(e)[:1000],
                        }
                    saved[i] = row
                    with path.open("a") as f:
                        f.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                    completed += 1
                    queue.task_done()
                    if completed % 25 == 0 or completed == len(pending):
                        ok = sum(
                            all(
                                r.get("outputs", {}).get(v, {}).get("status")
                                == "success"
                                for v in ["bf16", "fp8"]
                            )
                            for r in saved.values()
                        )
                        print(
                            f"{name}: done {len(saved)}/{len(data)}, paired_ok={ok}, elapsed={time.monotonic()-started:.1f}s",
                            flush=True,
                        )

            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

            async def heartbeat():
                while any(not t.done() for t in tasks):
                    await asyncio.sleep(30)
                    print(
                        f"{name}: active={sum(not t.done() for t in tasks)}, done={len(saved)}/{len(data)}, queued={queue.qsize()}",
                        flush=True,
                    )

            heart = asyncio.create_task(heartbeat())
            await asyncio.gather(*tasks)
            heart.cancel()
            result = summary(list(saved.values()), len(data))
            result.update(
                dataset=name,
                metric=DATASETS[name][2],
                elapsed_seconds=time.monotonic() - started,
            )
            atomic_json(args.output / f"{name}_summary.json", result)
            print(json.dumps(result, ensure_ascii=False)[:2500], flush=True)


if __name__ == "__main__":
    logging.getLogger("aoss_client").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("benchmark/beebeeomni"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cache", type=Path, default=Path("/tmp/sglang_beebee_benchmark_cache")
    )
    p.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    p.add_argument("--bf16-url", default="http://127.0.0.1:18100")
    p.add_argument("--fp8-url", default="http://127.0.0.1:18101")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--concurrency", type=int, default=32, help="Audio concurrency per precision"
    )
    p.add_argument("--image-concurrency", type=int, default=64)
    p.add_argument("--video-concurrency", type=int, default=4)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--ratio", type=int, default=1)
    p.add_argument("--size-factor", type=int, default=32)
    p.add_argument("--mcq-tokens", type=int, default=64)
    p.add_argument("--asr-tokens", type=int, default=1024)
    p.add_argument("--ocr-tokens", type=int, default=4096)
    p.add_argument("--retry-errors", action="store_true")
    asyncio.run(main(p.parse_args()))
