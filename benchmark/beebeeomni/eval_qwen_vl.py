"""Evaluate local image/video MCQ datasets through Qwen's native chat API.

Uses the frozen VeOmni question/answer protocol, but native Qwen media tokens
and chat template. No BeeBee frame duplication or downsample-ratio fields.
Run with /usr/bin/python3; dependencies match eval_precision.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import subprocess
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
from PIL import Image

import eval_precision as base

DATASETS = {
    k: v for k, v in base.DATASETS.items()
    if k in ("mmstar", "RealWorldQA", "ocr", "chartqa", "mvbench", "videomme", "videommmu")
}
INSTRUCTION = "\nAnswer with only the option letter (A, B, C, etc.). Do not explain."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_segment(path, sample, cache):
    start = max(0.0, float(sample.get("start") or 0))
    end = sample.get("end")
    if end is None and start == 0:
        return path
    key = hashlib.sha256(f"{path}:{start}:{end}:ffv1-v1".encode()).hexdigest()
    target = cache / "segments" / f"{key}.mkv"
    with base.LOCK_GUARD:
        lock = base.LOCKS.setdefault(key, base.threading.Lock())
    with lock:
        if target.exists() and target.stat().st_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp.mkv")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ss", str(start)]
        if end is not None:
            duration = float(end) - start
            if duration <= 0:
                raise ValueError(f"Invalid video segment {start}:{end}")
            cmd += ["-t", str(duration)]
        cmd += ["-map", "0:v:0", "-an", "-c:v", "ffv1", "-threads", "2", str(temp)]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-1000:])
        temp.replace(target)
    return target


def prepare(name, index, sample, args):
    started = time.monotonic()
    media, uri = base.protocol.detect_media_type(sample)
    question, gt = base.protocol.prepare_question(sample, "mcq", media, sample.get("language", "en"))
    if not isinstance(question, str):
        raise ValueError("Only single-turn MCQ samples are supported")
    if len(gt) != 1 or gt not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError(f"Invalid option ground truth: {gt!r}")
    content, media_info = [], []
    if media == "image":
        resize_args = SimpleNamespace(ratio=1, size_factor=32)
        for item in uri if isinstance(uri, list) else [uri]:
            path = base.cache_media(item, args.cache)
            with Image.open(path) as image:
                original = image.size
                image = base.resize(image, resize_args)
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base.jpeg(image)}})
                media_info.append({"path": str(path), "sha256": sha256(path), "original_size": original, "request_size": image.size})
    elif media == "video":
        path = base.cache_media(uri, args.cache)
        cropped = crop_segment(path, sample, args.cache)
        # Local paths are intentional: this runner and the server share a host.
        content.append({"type": "video_url", "video_url": {"url": str(cropped.resolve())}})
        media_info.append({"path": str(path), "sha256": sha256(path), "request_path": str(cropped), "start": sample.get("start"), "end": sample.get("end")})
    else:
        raise ValueError(f"Unsupported media type: {media}")
    content.append({"type": "text", "text": question + INSTRUCTION})
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": sample.get("system", "You are a helpful assistant.")},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1.05,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    row = {
        "dataset": name, "index": index, "sample_id": sample.get("id", index),
        "media_type": media, "uri": uri, "gt": gt, "question": question,
        "subtask": sample.get("category_abbr", sample.get("task", sample.get("category", ""))),
        "duration_group": sample.get("duration"), "media_info": media_info,
        "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "prepare_seconds": time.monotonic() - started,
    }
    return payload, row


async def request(session, payload, args):
    started = time.monotonic()
    async with session.post(args.url.rstrip("/") + "/v1/chat/completions", json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:1500]}")
        result = json.loads(body)
    choice = result["choices"][0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise RuntimeError(f"Unexpected finish_reason: {choice}")
    raw = choice["message"].get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Empty model response: {choice}")
    return {
        "status": "success", "raw_answer": raw,
        "prediction": base.protocol.extract_mcq_answer(raw.strip()),
        "elapsed_seconds": time.monotonic() - started,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage", {}),
    }


def summarize(rows, planned, full_count):
    good = [r for r in rows if r["output"]["status"] == "success"]
    correct = sum(r["output"]["score"] for r in good)
    latency = [r["output"]["elapsed_seconds"] for r in good]
    prepare_time = [r["prepare_seconds"] for r in good]
    result = {
        "source_samples": full_count, "planned": planned, "recorded": len(rows),
        "success": len(good), "errors": len(rows) - len(good), "correct": correct,
        "accuracy_successful": correct / len(good) if good else None,
        "accuracy_errors_as_wrong": correct / planned if planned else None,
        "coverage": len(good) / planned if planned else None,
        "invalid_predictions": sum(not r["output"]["prediction"] for r in good),
        "truncated_answers": sum(r["output"].get("finish_reason") == "length" for r in good),
        "request_latency_seconds": {"mean": float(np.mean(latency)), "p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))} if latency else {},
        "prepare_seconds_mean": float(np.mean(prepare_time)) if prepare_time else None,
        "prompt_tokens_mean": float(np.mean([r["output"]["usage"].get("prompt_tokens", 0) for r in good])) if good else None,
        "subtasks": {}, "duration_groups": {},
        "error_examples": [{"index": r["index"], "error": r["output"].get("error")} for r in rows if r["output"]["status"] != "success"][:10],
    }
    for field, dest in (("subtask", "subtasks"), ("duration_group", "duration_groups")):
        for key in sorted({str(r.get(field)) for r in good}):
            group = [r for r in good if str(r.get(field)) == key]
            result[dest][key] = {"samples": len(group), "accuracy": sum(r["output"]["score"] for r in group) / len(group)}
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    datasets = {name: base.data_file(args.data / DATASETS[name][0]) for name in args.datasets}
    selected = {}
    for name, samples in datasets.items():
        rng = random.Random(f"{args.seed}:{name}")
        selected[name] = sorted(rng.sample(range(len(samples)), min(args.limit or len(samples), len(samples))))
    config = {
        "model": args.model, "url": args.url, "seed": args.seed, "limit": args.limit,
        "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        "protocol": "native Qwen chat; enable_thinking=false; frozen VeOmni question and MCQ extraction",
        "instruction": INSTRUCTION, "image_preprocessing": "ratio=1, factor=32, 65536..2027520 pixels, RGB JPEG quality=75, one image per input",
        "video_preprocessing": "native server config recorded in server-info.json; segments cropped with lossless ffv1; no subtitles",
        "script_sha256": sha256(__file__), "reference_protocol_sha256": sha256(base.protocol.__file__),
        "datasets": {name: {"samples": len(data), "sha256": sha256(args.data / DATASETS[name][0]), "indices": selected[name]} for name, data in datasets.items()},
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration mismatch; use another output directory")
    base.atomic_json(config_path, config)
    timeout = aiohttp.ClientTimeout(total=900, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.url.rstrip("/") + "/get_server_info") as response:
            response.raise_for_status()
            base.atomic_json(args.output / "server-info.json", await response.json())
        for name, data in datasets.items():
            path = args.output / f"{name}.jsonl"
            saved = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row["index"] in selected[name]:
                        saved[row["index"]] = row
            queue = asyncio.Queue()
            for index in selected[name]:
                if index not in saved or args.retry_errors and saved[index]["output"]["status"] != "success":
                    queue.put_nowait(index)
            started = time.monotonic()
            print(f"{name}: {len(selected[name])}/{len(data)} selected, {queue.qsize()} pending", flush=True)

            async def worker():
                while not queue.empty():
                    index = queue.get_nowait()
                    row = {"dataset": name, "index": index}
                    try:
                        payload, row = await asyncio.to_thread(prepare, name, index, data[index], args)
                        output = await request(session, payload, args)
                        output["score"] = int(output["prediction"] == row["gt"])
                        row["output"] = output
                    except Exception as exc:
                        row["output"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
                    saved[index] = row
                    with path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    summary = summarize(list(saved.values()), len(selected[name]), len(data))
                    base.atomic_json(args.output / f"{name}_summary.json", summary)
                    if len(saved) % 10 == 0 or len(saved) == len(selected[name]) or row["output"]["status"] == "error":
                        print(f"{name}: {len(saved)}/{len(selected[name])}, correct={summary['correct']}, errors={summary['errors']}, elapsed={time.monotonic()-started:.1f}s", flush=True)
                        if row["output"]["status"] == "error":
                            print(row["output"]["error"], flush=True)

            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            print(json.dumps(summarize(list(saved.values()), len(selected[name]), len(data)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/sglang_beebee_benchmark_cache"))
    p.add_argument("--url", default="http://127.0.0.1:18005")
    p.add_argument("--model", default="/mnt/afs/share/Qwen3.8-27B")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--limit", type=int, default=100, help="Uniform random sample count per dataset; 0 for all")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--retry-errors", action="store_true")
    args = p.parse_args()
    if args.limit < 0 or args.concurrency < 1 or args.max_tokens < 1:
        p.error("limit must be nonnegative; concurrency and max-tokens must be positive")
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    asyncio.run(run(args))
