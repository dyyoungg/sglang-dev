"""Evaluate local image/video MCQ and audio ASR datasets with Qwen or BeeBee Omni models.

Official Qwen model (native chat template and video_url processing)::

    /usr/bin/python -u benchmark/beebeeomni/eval_qwen_vl.py \
        --output outputs/qwen_mvbench --model-type qwen \
        --model /mnt/afs/share/Qwen3.8-27B \
        --datasets mvbench --limit 0

Audio ASR (CER/WER)::

    /usr/bin/python -u benchmark/beebeeomni/eval_qwen_vl.py \
        --output outputs/beebee_aishell --model-type beebee \
        --model /mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base_qwen38_projector \
        --datasets aishell --think false

27B MCQ with the closed think prefix (the VeOmni ``eval_sglang.py`` equivalent
is ``--think false``; its default remains ``none``)::

    /usr/bin/python -u benchmark/beebeeomni/eval_qwen_vl.py \
        --output outputs/beebee_mvbench_think_false --model-type beebee \
        --model /mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base_qwen38_projector \
        --datasets mvbench --think false

BeeBee custom model (pairwise frames and VeOmni answer prefix)::

    /usr/bin/python -u benchmark/beebeeomni/eval_qwen_vl.py \
        --output outputs/beebee_mvbench_ratio4 --model-type beebee \
        --model /mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base_qwen38_projector \
        --datasets mvbench --limit 0 --downsample-ratio 4 \
        --veomni-resize --veomni-format --pair-vision-segments

BeeBee videos are sampled to at most 64 frames; adjacent frames form pairs
without duplicating frames. ``--pair-vision-segments`` wraps each pair in its
own ``<|vision_start|>...<|vision_end|>`` block. BeeBee prompts are assembled
manually. ``--think`` accepts ``none``, ``false`` (the 27B closed think block),
or ``true`` (an open think block). Official Qwen uses its native chat template
and ``--enable-thinking``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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
from decord import VideoReader, cpu

import eval_precision as base

VISUAL_DATASETS = {
    k: v for k, v in base.DATASETS.items()
    if k in ("mmstar", "RealWorldQA", "ocr", "chartqa", "mvbench", "videomme", "videommmu")
}
DATASETS = dict(VISUAL_DATASETS)
DATASETS.update({
    name: base.DATASETS[name]
    for name in (
        "aishell", "wenet", "online_badaudio",
        "librispeech_dev_clean", "librispeech_dev_other",
        "librispeech_test_clean", "librispeech_test_other",
    )
})
INSTRUCTION = "\nAnswer with only the option letter (A, B, C, etc.). Do not explain."
VEOMNI_PREFIX = "My best option: ("


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampling_params(args, index=None):
    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    for name in ("top_k", "min_p", "presence_penalty"):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    if args.sampling_seed is not None:
        params["seed"] = (
            args.sampling_seed
            if args.fixed_sampling_seed
            else args.sampling_seed + (index or 0)
        )
    return params


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


def extract_video_frames(path, sample, max_frames=64):
    """Sample frames with VeOmni's midpoint rule and segment bounds."""
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total = len(reader)
    fps = float(reader.get_avg_fps() or 2.0)
    start = max(0, int(float(sample.get("start") or 0) * fps))
    end = min(total, int(float(sample["end"]) * fps)) if sample.get("end") is not None else total
    if end <= start:
        start, end = 0, total
    valid = end - start
    count = min(max_frames, valid)
    if count % 2:
        count -= 1
    if count < 2:
        raise RuntimeError(f"Video segment has fewer than two decodable frames: {path}")
    segment = float((valid - 1) / count)
    indices = [
        min((int(segment * i) + int(segment * (i + 1))) // 2 + start, end - 1)
        for i in range(count)
    ]
    frames = [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]
    return frames, indices, fps


def prepare(name, index, sample, args):
    started = time.monotonic()
    media, uri = base.protocol.detect_media_type(sample)
    _, task, metric = DATASETS[name]
    question, gt = base.protocol.prepare_question(sample, task, media, sample.get("language", "en"))
    if not isinstance(question, str):
        raise ValueError("Only single-turn MCQ samples are supported")
    if task == "mcq" and (len(gt) != 1 or gt not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError(f"Invalid option ground truth: {gt!r}")
    system = sample.get("system", sample.get("system_prompt", "You are a helpful assistant."))
    content, media_info, pair_image_data = [], [], []
    resize_args = SimpleNamespace(
        ratio=args.downsample_ratio if args.model_type == "beebee" and args.veomni_resize else 1,
        size_factor=32,
        min_side=64 if args.model_type == "beebee" and args.veomni_resize else 112,
    )
    encoded_audio = None
    if media == "audio":
        path = base.cache_media(uri, args.cache)
        media_info.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        if args.model_type == "qwen":
            suffix = path.suffix.lower().lstrip(".")
            if suffix not in ("wav", "mp3"):
                converted = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                     "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                    capture_output=True, timeout=120, check=True,
                )
                audio_bytes = converted.stdout
                suffix = "wav"
            else:
                audio_bytes = path.read_bytes()
            encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
            content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/{suffix};base64,{encoded_audio}"}})
        else:
            encoded_audio = base64.b64encode(path.read_bytes()).decode("ascii")
    elif media == "image":
        for item in uri if isinstance(uri, list) else [uri]:
            path = base.cache_media(item, args.cache)
            with Image.open(path) as image:
                original = image.size
                image = base.resize(image, resize_args)
                encoded = "data:image/jpeg;base64," + base.jpeg(image)
                image_content = {"type": "image_url", "image_url": {"url": encoded}}
                if args.model_type == "beebee":
                    content.extend([image_content, dict(image_content)])
                    pair_image_data.extend([encoded.split(",", 1)[-1]] * 2)
                else:
                    content.append(image_content)
                media_info.append({"path": str(path), "sha256": sha256(path), "original_size": original, "request_size": image.size})
    elif media == "video":
        path = base.cache_media(uri, args.cache)
        cropped = crop_segment(path, sample, args.cache)
        if args.model_type == "beebee":
            frames, frame_indices, source_fps = extract_video_frames(path, sample, 64)
            for frame in frames:
                image = base.resize(frame, resize_args)
                encoded = "data:image/jpeg;base64," + base.jpeg(image)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
                pair_image_data.append(encoded.split(",", 1)[-1])
            media_info.append({
                "path": str(path), "sha256": sha256(path), "request_path": str(cropped),
                "start": sample.get("start"), "end": sample.get("end"),
                "frame_indices": frame_indices, "source_fps": source_fps,
                "transport": "up to 64 midpoint-sampled frames; adjacent frames form image pairs",
            })
        else:
            # Qwen's official processor receives the video and applies its native template.
            content.append({"type": "video_url", "video_url": {"url": str(cropped.resolve())}})
            media_info.append({"path": str(path), "sha256": sha256(path), "request_path": str(cropped), "start": sample.get("start"), "end": sample.get("end")})
    else:
        raise ValueError(f"Unsupported media type: {media}")
    question_text = question + ("" if args.veomni_format or task == "generate" else INSTRUCTION)
    if args.model_type == "beebee" and args.think == "true" and task == "mcq":
        question_text += '\nEnd your response with "my best option is (X)", replacing X with the correct option letter.'
    content.append({"type": "text", "text": question_text})
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        **sampling_params(args, index),
        "max_tokens": args.asr_tokens if media == "audio" else args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "stream": False,
    }
    if args.model_type == "beebee":
        image_count = len(pair_image_data)
        if args.pair_vision_segments:
            if image_count % 2:
                raise ValueError(f"Pair vision segments require an even image count, got {image_count}")
            vision = "".join(
                "<|vision_start|><image><image><|vision_end|>"
                for _ in range(image_count // 2)
            )
        else:
            vision = "<|vision_start|>" + "<image>" * image_count + "<|vision_end|>" if image_count else ""
        audio_prefix = "<|audio_start|><audio><|audio_end|>\n" if media == "audio" else ""
        think_prefix = {
            "none": "",
            "false": "<think>\n\n</think>\n\n",
            "true": "<think>\n",
        }[args.think]
        answer_prefix = VEOMNI_PREFIX if task == "mcq" and args.veomni_format and args.think != "true" else ""
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>"
            f"{vision}\n<|im_start|>user\n{audio_prefix}{question_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{think_prefix}{answer_prefix}"
        )
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": (
                    args.asr_tokens if media == "audio"
                    else 1 if args.veomni_format and args.think != "true"
                    else args.max_tokens
                ),
                **sampling_params(args, index),
                "stop": [")", "<|im_end|>"] if args.veomni_format and args.think != "true" and task == "mcq" else "<|im_end|>",
            },
            "stream": True,
        }
        if media == "audio":
            payload["audio_data"] = encoded_audio
        else:
            payload["image_data"] = pair_image_data or None
            payload["image_downsample_ratios"] = [args.downsample_ratio] * (image_count // 2)
    row = {
        "dataset": name, "index": index, "sample_id": sample.get("id", index),
        "media_type": media, "uri": uri, "gt": gt, "question": question,
        "task": task, "metric": metric,
        "subtask": sample.get("category_abbr", sample.get("task", sample.get("category", ""))),
        "duration_group": sample.get("duration"), "media_info": media_info,
        "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "prepare_seconds": time.monotonic() - started,
    }
    return payload, row


async def request(session, payload, args):
    started = time.monotonic()
    if args.model_type == "beebee":
        async with session.post(args.url.rstrip("/") + "/generate", json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1500]}")
            full = ""
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                event = json.loads(line[5:].strip())
                if "error" in event:
                    raise RuntimeError(f"SSE server error: {event['error']}")
                if "text" in event:
                    full = event["text"]
        if not full.strip():
            raise ValueError("Empty model response")
        return {
            "status": "success", "raw_answer": full,
            "final_answer": full.rsplit("</think>", 1)[-1].strip(),
            "elapsed_seconds": time.monotonic() - started,
            "finish_reason": "stop", "usage": {},
        }
    async with session.post(args.url.rstrip("/") + "/v1/chat/completions", json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:1500]}")
        result = json.loads(body)
    choice = result["choices"][0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise RuntimeError(f"Unexpected finish_reason: {choice}")
    message = choice["message"]
    raw = message.get("content") or ""
    if not isinstance(raw, str):
        raise ValueError(f"Invalid model response: {choice}")
    if not args.enable_thinking and not raw.strip():
        raise ValueError(f"Empty model response: {choice}")
    answer = raw
    reasoning = message.get("reasoning_content")
    reasoning_complete = None
    if args.enable_thinking:
        if reasoning is not None:
            reasoning_complete = bool(answer.strip())
        elif "</think>" in raw:
            reasoning, answer = raw.rsplit("</think>", 1)
            reasoning_complete = True
        else:
            reasoning, answer = raw, ""
            reasoning_complete = False
    return {
        "status": "success", "raw_answer": raw,
        "final_answer": answer.strip(),
        "reasoning_content": reasoning,
        "reasoning_complete": reasoning_complete,
        "elapsed_seconds": time.monotonic() - started,
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage", {}),
    }


def summarize(rows, planned, full_count):
    good = [r for r in rows if r["output"]["status"] == "success"]
    accuracy_rows = [r for r in good if r.get("metric") == "accuracy"]
    is_asr = bool(good) and all(r.get("metric") in ("cer", "wer") for r in good)
    metric_values = {
        metric: float(np.mean([r["output"]["score"] for r in good if r.get("metric") == metric]))
        for metric in sorted({r.get("metric") for r in good})
    }
    correct = sum(r["output"]["score"] for r in accuracy_rows)
    latency = [r["output"]["elapsed_seconds"] for r in good]
    prepare_time = [r["prepare_seconds"] for r in good]
    result = {
        "source_samples": full_count, "planned": planned, "recorded": len(rows),
        "success": len(good), "errors": len(rows) - len(good), "correct": correct,
        "accuracy_successful": correct / len(accuracy_rows) if accuracy_rows else None,
        "accuracy_errors_as_wrong": correct / planned if accuracy_rows and planned else None,
        "metrics": metric_values,
        "coverage": len(good) / planned if planned else None,
        "invalid_predictions": sum(not r["output"]["prediction"] for r in good),
        "truncated_answers": sum(r["output"].get("finish_reason") == "length" for r in good),
        "incomplete_reasoning": sum(r["output"].get("reasoning_complete") is False for r in good),
        "completion_tokens_mean": float(np.mean([r["output"]["usage"].get("completion_tokens", 0) for r in good])) if good else None,
        "request_latency_seconds": {"mean": float(np.mean(latency)), "p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))} if latency else {},
        "prepare_seconds_mean": float(np.mean(prepare_time)) if prepare_time else None,
        "prompt_tokens_mean": float(np.mean([r["output"]["usage"].get("prompt_tokens", 0) for r in good])) if good else None,
        "subtasks": {}, "duration_groups": {},
        "error_examples": [{"index": r["index"], "error": r["output"].get("error")} for r in rows if r["output"]["status"] != "success"][:10],
    }
    if is_asr:
        result.update({
            f"corpus_{metric}": (
                sum(r["output"]["edit_errors"] for r in good if r["metric"] == metric)
                / sum(r["output"]["reference_units"] for r in good if r["metric"] == metric)
            )
            for metric in metric_values
            if sum(r["output"]["reference_units"] for r in good if r["metric"] == metric)
        })
    for field, dest in (("subtask", "subtasks"), ("duration_group", "duration_groups")):
        for key in sorted({str(r.get(field)) for r in good}):
            group = [r for r in good if str(r.get(field)) == key]
            result[dest][key] = {
                "samples": len(group),
                "metrics": {
                    metric: float(np.mean([r["output"]["score"] for r in group if r.get("metric") == metric]))
                    for metric in sorted({r.get("metric") for r in group})
                },
            }
            if all(r["metric"] == "accuracy" for r in group):
                result[dest][key]["accuracy"] = result[dest][key]["metrics"]["accuracy"]
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
        "asr_tokens": args.asr_tokens,
        "sampling_params": sampling_params(args),
        "sampling_seed_rule": (
            "fixed seed"
            if args.sampling_seed is not None and args.fixed_sampling_seed
            else "base seed + dataset index"
            if args.sampling_seed is not None
            else None
        ),
        "model_type": args.model_type,
        "pair_vision_segments": args.pair_vision_segments,
        "downsample_ratio": args.downsample_ratio if args.model_type == "beebee" else None,
        "veomni_format": args.veomni_format,
        "veomni_resize": args.veomni_resize,
        "think": args.think,
        "enable_thinking": args.enable_thinking,
        "protocol": (
            f"BeeBee native /generate; think={args.think}; "
            + ("VeOmni My best option: ( prefix" if args.veomni_format else "letter-only instruction")
        ) if args.model_type == "beebee" else
            f"official Qwen /v1/chat/completions template; enable_thinking={str(args.enable_thinking).lower()}",
        "instruction": (
            VEOMNI_PREFIX if args.veomni_format else "letter-only instruction"
        ) if args.model_type == "beebee" else INSTRUCTION,
        "image_preprocessing": (
            f"ratio={args.downsample_ratio if args.veomni_resize else 1}, factor=32, "
            "RGB JPEG quality=75, each still image duplicated into a pair"
        ) if args.model_type == "beebee" else
            "ratio=1, factor=32, 65536..2027520 pixels, RGB JPEG quality=75, one image per input",
        "video_preprocessing": (
            f"VeOmni midpoint sampling up to 64 frames, adjacent frames paired without duplication, "
            f"ratio={args.downsample_ratio}, pair_vision_segments={args.pair_vision_segments}"
        ) if args.model_type == "beebee" else
            "native Qwen video_url processing; segments cropped with lossless ffv1; no subtitles",
        "audio_transport": (
            "BeeBee /generate audio_data: original file bytes, base64"
            if args.model_type == "beebee" else
            "Qwen chat audio_url: original WAV/MP3 or FLAC converted to 16 kHz mono WAV"
        ),
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
                        answer = output.get("final_answer", output["raw_answer"])
                        scored = base.score(row, answer)
                        output.update(scored)
                        output["prediction"] = scored["pred"]
                        row["output"] = output
                    except Exception as exc:
                        row["output"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}
                    saved[index] = row
                    with path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    summary = summarize(list(saved.values()), len(selected[name]), len(data))
                    base.atomic_json(args.output / f"{name}_summary.json", summary)
                    if len(saved) % 10 == 0 or len(saved) == len(selected[name]) or row["output"]["status"] == "error":
                        print(f"{name}: {len(saved)}/{len(selected[name])}, metrics={summary['metrics']}, errors={summary['errors']}, elapsed={time.monotonic()-started:.1f}s", flush=True)
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
    p.add_argument("--model-type", choices=("qwen", "beebee"), default="qwen",
                   help="qwen uses the official chat template; beebee uses pairwise images and a manually assembled prompt")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(VISUAL_DATASETS))
    p.add_argument("--limit", type=int, default=100, help="Uniform random sample count per dataset; 0 for all")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--asr-tokens", type=int, default=1024,
                   help="Maximum generated tokens for audio transcription")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-p", type=float)
    p.add_argument("--presence-penalty", type=float)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--sampling-seed", type=int)
    p.add_argument("--fixed-sampling-seed", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--think", choices=("none", "false", "true"), default="false",
                   help="BeeBee prompt think mode: none, closed empty block (false), or open block (true)")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--pair-vision-segments", action="store_true",
                   help="BeeBee only: put each adjacent frame/image pair in its own vision segment")
    p.add_argument("--downsample-ratio", type=int, default=1,
                   help="BeeBee image downsample ratio, applied once per adjacent image pair")
    p.add_argument("--veomni-format", action="store_true",
                   help="BeeBee only: use VeOmni's My best option: ( prefix and stop after the option")
    p.add_argument("--veomni-resize", action="store_true",
                   help="BeeBee only: use ratio-aware image resizing with factor 32")
    args = p.parse_args()
    if args.limit < 0 or args.concurrency < 1 or args.max_tokens < 1 or args.asr_tokens < 1 or args.downsample_ratio < 1:
        p.error("limit must be nonnegative; concurrency, max-tokens and asr-tokens must be positive")
    if args.model_type == "qwen" and (args.pair_vision_segments or args.veomni_format or args.veomni_resize):
        p.error("pair-vision-segments, veomni-format and veomni-resize require --model-type beebee")
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    asyncio.run(run(args))
