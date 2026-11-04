"""Evaluate the OCR generation set against one SGLang ``/generate`` server.

Image resizing and scoring are reused from ``eval_precision.py``. Prompts use
the model's native chat template and one copy of each image, as in the earlier
Qwen evaluation. No ground-truth text is sent to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer

import eval_precision as base


def prepare(index, sample, args, tokenizer):
    started = time.monotonic()
    images, media_info = [], []
    sources = sample["image_path"]
    for source in sources if isinstance(sources, list) else [sources]:
        path = base.cache_media(source, args.cache)
        with Image.open(path) as image:
            original_size = image.size
            image = base.resize(image, SimpleNamespace(ratio=args.ratio, size_factor=args.size_factor))
            images.append("data:image/jpeg;base64," + base.jpeg(image))
            media_info.append({"original_size": original_size, "request_size": image.size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    messages = [
        {"role": "system", "content": sample.get("system", "You are a helpful assistant.")},
        {"role": "user", "content": [{"type": "image"} for _ in images]
         + [{"type": "text", "text": sample["question"]}]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    sampling = {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 1.5, "repetition_penalty": 1.0,
                "max_new_tokens": args.ocr_tokens,
                "stop": ["<|im_end|>"]}
    payload = {"text": prompt, "image_data": images, "sampling_params": sampling, "stream": False}
    row = {"dataset": "ocr_generate", "index": index, "category": sample.get("category"),
           "task": "generate", "metric": "similarity", "gt": sample["answer"],
           "question": sample["question"], "media_path": sources, "media_info": media_info,
           "prompt": prompt, "sampling_params": sampling,
           "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
           "prepare_seconds": time.monotonic() - started}
    return payload, row


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def request(session, url, payload, row, timeout_retries=3):
    started = time.monotonic()
    for attempt in range(timeout_retries):
        try:
            async with session.post(url.rstrip("/") + "/generate", json=payload) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {text[:1000]}")
                body = json.loads(text)
            if "text" not in body:
                raise RuntimeError(f"Missing generation: {str(body)[:500]}")
            if body.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
                raise RuntimeError(str(body["meta_info"]["finish_reason"]))
            raw = body["text"]
            scored = base.score(row, raw)
            ref = base.clean_training_text(row["gt"])
            hyp = base.clean_training_text(scored["pred"])
            return {
                "status": "success",
                "raw_answer": raw,
                **scored,
                "full_gt_similarity": 1 - Levenshtein.distance(ref, hyp) / max(len(ref), len(hyp), 1),
                "cleaned_gt_coverage": min(len(hyp) / max(len(ref), 1), 1),
                "meta_info": body.get("meta_info", {}),
                "elapsed_seconds": time.monotonic() - started,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            if attempt + 1 == timeout_retries:
                return {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.monotonic() - started,
                    "attempts": timeout_retries,
                }
            await asyncio.sleep(1 + attempt)


def summarize(rows, total):
    good = [r for r in rows if r.get("output", {}).get("status") == "success"]
    scores = [r["output"]["similarity"] for r in good]
    latency = [r["output"]["elapsed_seconds"] for r in good]
    return {
        "total": total,
        "recorded": len(rows),
        "success": len(good),
        "errors": len(rows) - len(good),
        "mean_similarity": float(np.mean(scores)) if scores else None,
        "mean_full_gt_similarity": float(np.mean([r["output"]["full_gt_similarity"] for r in good])) if good else None,
        "mean_cleaned_gt_coverage": float(np.mean([r["output"]["cleaned_gt_coverage"] for r in good])) if good else None,
        "median_similarity": float(np.percentile(scores, 50)) if scores else None,
        "p05_similarity": float(np.percentile(scores, 5)) if scores else None,
        "p95_similarity": float(np.percentile(scores, 95)) if scores else None,
        "mean_normalized_edit_distance": (
            float(np.mean([r["output"]["normalized_edit_distance"] for r in good]))
            if good
            else None
        ),
        "request_latency_seconds": {
            "mean": float(np.mean(latency)),
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
        }
        if latency
        else {},
        "length_stopped": sum(
            r["output"].get("meta_info", {}).get("finish_reason", {}).get("type")
            == "length"
            for r in good
        ),
        "error_examples": [
            {"index": r["index"], "error": r.get("output", {}).get("error")}
            for r in rows
            if r.get("output", {}).get("status") != "success"
        ][:10],
    }


async def run(args):
    data = base.data_file(args.input)
    if args.limit:
        data = data[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "url": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "ocr_tokens": args.ocr_tokens,
        "ratio": args.ratio,
        "size_factor": args.size_factor,
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "scoring": "VeOmni clean_text + gt prefix (hypothesis length + 5) + normalized Levenshtein complement",
        "samples": len(data),
        "prompt_protocol": "native Qwen chat template; enable_thinking=false; one image per input",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output configuration differs; use a new directory")
    atomic_json(config_path, config)
    result_path = args.output / "results.jsonl"
    saved = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["index"]] = row
    pending = [
        i
        for i in range(len(data))
        if i not in saved
        or (args.retry_errors and saved[i].get("output", {}).get("status") != "success")
    ]
    queue = asyncio.Queue()
    for i in pending:
        queue.put_nowait(i)
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=30)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2 + 2)
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker():
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                row = {"dataset": "ocr_generate", "index": index}
                try:
                    payload, row = await asyncio.to_thread(
                        prepare, index, data[index], args, tokenizer
                    )
                    row["output"] = await request(session, args.url, payload, row)
                except Exception as exc:
                    row["output"] = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                saved[index] = row
                with result_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                queue.task_done()
                done = len(saved)
                if done % 10 == 0 or done == len(data) or row["output"]["status"] != "success":
                    print(
                        f"done {done}/{len(data)}, elapsed={time.monotonic()-started:.1f}s",
                        flush=True,
                    )

        await asyncio.gather(
            *(worker() for _ in range(min(args.concurrency, max(1, len(pending)))))
        )
    summary = summarize(list(saved.values()), len(data))
    summary["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("ocr_intense_val_generate.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/sglang_beebee_benchmark_cache"))
    parser.add_argument("--url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", default="/mnt/afs/share/Qwen3.8-27B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ocr-tokens", type=int, default=4096)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--size-factor", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete dataset")
    asyncio.run(run(parser.parse_args()))
