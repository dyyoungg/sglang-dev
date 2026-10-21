"""Evaluate the BeeBeeOmni pure-text multiple-choice set through SGLang.

The input contains a short multi-turn conversation for each question.  Roles
are converted from the dataset's ``Human``/``Assistant`` names to the native
OpenAI chat roles, and the model is asked for a non-thinking answer.  Results
are resumable and include per-category accuracy and request latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import numpy as np


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def answer_letter(text: str, candidates: list[str]) -> str | None:
    """Extract the first standalone candidate letter from a short answer."""
    if not isinstance(text, str):
        return None
    allowed = "".join(re.escape(x) for x in candidates)
    match = re.search(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", text.upper())
    return match.group(1) if match else None


def messages(sample: dict) -> list[dict]:
    result = []
    for item in sample["question"]:
        role = {"Human": "user", "Assistant": "assistant"}.get(item["role"])
        if role is None:
            raise ValueError(f"Unsupported role: {item['role']!r}")
        result.append({"role": role, "content": item["value"]})
    return result


def payload(sample: dict, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages(sample),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }


async def request(session, url: str, body: dict, expected_answer: str):
    started = time.monotonic()
    async with session.post(
        url.rstrip("/") + "/v1/chat/completions", json=body
    ) as response:
        text = await response.text()
        elapsed = time.monotonic() - started
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {text[:2000]}")
    data = json.loads(text)
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty response: {choice}")
    return {
        "status": "success",
        "raw_answer": content,
        "prediction": answer_letter(content, ["A", "B", "C", "D", "E"]),
        "correct": answer_letter(content, ["A", "B", "C", "D", "E"])
        == expected_answer,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def summarize(rows: list[dict], total: int) -> dict:
    good = [r for r in rows if r.get("output", {}).get("status") == "success"]
    correct = sum(bool(r["output"].get("correct")) for r in good)
    latencies = [r["output"]["elapsed_seconds"] for r in good]
    out = {
        "total": total,
        "recorded": len(rows),
        "success": len(good),
        "errors": len(rows) - len(good),
        "correct": correct,
        "accuracy_successful": correct / len(good) if good else None,
        "accuracy_errors_as_wrong": correct / total if total else None,
        "invalid_predictions": sum(not r["output"].get("prediction") for r in good),
        "latency_seconds": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
        if latencies
        else {},
        "categories": {},
        "error_examples": [
            {"index": r["index"], "error": r.get("output", {}).get("error")}
            for r in rows
            if r.get("output", {}).get("status") != "success"
        ][:10],
    }
    groups = defaultdict(list)
    for row in good:
        groups[row["category"]].append(row)
    for category, group in sorted(groups.items()):
        out["categories"][category] = {
            "samples": len(group),
            "correct": sum(bool(x["output"].get("correct")) for x in group),
            "accuracy": sum(bool(x["output"].get("correct")) for x in group)
            / len(group),
            "latency_mean_seconds": float(
                np.mean([x["output"]["elapsed_seconds"] for x in group])
            ),
        }
    return out


async def run(args):
    data = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "url": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "samples": len(data),
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
        or (args.retry_errors and saved[i]["output"].get("status") != "success")
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
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                sample = data[i]
                row = {"index": i, "category": sample.get("category", "")}
                try:
                    body = payload(sample, args.model, args.max_tokens)
                    output = await request(session, args.url, body, sample["answer"])
                    row["output"] = output
                except Exception as exc:
                    row["output"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                saved[i] = row
                with result_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                queue.task_done()
                done = len(saved)
                if done % 25 == 0 or done == len(data) or row["output"]["status"] != "success":
                    print(f"done {done}/{len(data)}, elapsed={time.monotonic()-started:.1f}s", flush=True)

        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, max(1, len(pending))))))
    summary = summarize(list(saved.values()), len(data))
    summary["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("text_eval.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", default="/mnt/afs/share/Qwen3.8-27B")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))
