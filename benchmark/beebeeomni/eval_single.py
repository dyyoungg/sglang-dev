"""Single-server business evaluation for one precision/configuration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import aiohttp

import eval_precision as base


def atomic_json(path, value):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def summarize(rows, dataset):
    out = [
        r["outputs"]["bf16"]
        for r in rows
        if r.get("outputs", {}).get("bf16", {}).get("status") == "success"
    ]
    errors = [r for r in rows if r.get("outputs", {}).get("bf16", {}).get("status") != "success"]
    scores = [x["score"] for x in out]
    result = {
        "total": len(rows),
        "recorded": len(rows),
        "success": len(out),
        "errors": len(errors),
        "error_examples": [
            {"index": r.get("index"), "error": r.get("outputs", {}).get("bf16", {}).get("error")}
            for r in errors[:10]
        ],
        "mean_score": sum(scores) / len(scores) if scores else None,
        "score_direction": "higher_is_better",
        "dataset": dataset,
        "metric": base.DATASETS[dataset][2],
        "subtasks": {},
    }
    groups = {}
    for r in rows:
        o = r.get("outputs", {}).get("bf16", {})
        if o.get("status") == "success":
            groups.setdefault(r.get("subtask", ""), []).append(o["score"])
    for subtask, values in sorted(groups.items()):
        result["subtasks"][subtask] = {
            "samples": len(values),
            "score": sum(values) / len(values),
        }
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=1800, connect=30)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2 + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for name in args.datasets:
            source = args.data / base.DATASETS[name][0]
            data = base.data_file(source)
            if args.limit:
                data = data[: args.limit]
            path = args.output / f"{name}.jsonl"
            saved = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        saved[row["index"]] = row
            pending = [(i, sample) for i, sample in enumerate(data) if i not in saved]
            queue = asyncio.Queue()
            for item in pending:
                queue.put_nowait(item)
            started = time.monotonic()
            print(f"{name}: {len(data)} samples, {len(pending)} pending", flush=True)

            async def worker():
                while True:
                    try:
                        i, sample = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        payload, row = await asyncio.to_thread(base.prepare, name, i, sample, args)
                        response = await base.request(session, args.url, payload, row)
                        row["outputs"] = {"bf16": response}
                    except Exception as exc:
                        row = {"dataset": name, "index": i, "outputs": {"bf16": {"status": "error", "error": str(exc)[:1000]}}}
                    saved[i] = row
                    with path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    queue.task_done()
                    if len(saved) % 25 == 0 or len(saved) == len(data):
                        ok = sum(r.get("outputs", {}).get("bf16", {}).get("status") == "success" for r in saved.values())
                        print(f"{name}: done {len(saved)}/{len(data)}, success={ok}, elapsed={time.monotonic()-started:.1f}s", flush=True)

            tasks = [asyncio.create_task(worker()) for _ in range(min(args.concurrency, max(1, len(pending))))]
            await asyncio.gather(*tasks)
            result = summarize(list(saved.values()), name)
            result["elapsed_seconds"] = time.monotonic() - started
            atomic_json(args.output / f"{name}_summary.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            atomic_json(args.output / "config.json", {
                "datasets": {n: {"path": str((args.data / base.DATASETS[n][0]).resolve()), "samples": len(base.data_file(args.data / base.DATASETS[n][0]))} for n in args.datasets},
                "url": args.url, "frames": args.frames, "ratio": args.ratio,
                "size_factor": args.size_factor, "video_concurrency": args.concurrency,
                "tp_size": 2, "dp_size": 4,
                "protocol_sha256": hashlib.sha256(Path(base.protocol.__file__).read_bytes()).hexdigest(),
            })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("benchmark/beebeeomni"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18102")
    p.add_argument("--datasets", nargs="+", choices=["mvbench", "videomme", "videommmu"], required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/sglang_beebee_benchmark_cache"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--ratio", type=int, default=12)
    p.add_argument("--size-factor", type=int, default=32)
    p.add_argument("--mcq-tokens", type=int, default=64)
    p.add_argument("--asr-tokens", type=int, default=1024)
    p.add_argument("--ocr-tokens", type=int, default=4096)
    asyncio.run(run(p.parse_args()))
