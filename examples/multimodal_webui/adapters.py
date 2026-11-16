"""OpenAI chat and native BeeBee /generate adapters."""

import asyncio
import json
import logging
import os
import re
import uuid
from functools import lru_cache

import httpx

from .media import prepare_media

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def tokenizer_for(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )


def split_reasoning(raw, thinking=False):
    """Preserve all reasoning spans, including empty initial / reopened spans."""
    inside = thinking
    answer, blocks = [], []
    buffer = ""
    for part in re.split(r"(<think>|</think>)", raw):
        if part == "<think>":
            if inside:
                buffer += part
            else:
                answer.append(buffer)
                buffer = ""
                inside = True
        elif part == "</think>":
            if inside:
                blocks.append({"text": buffer, "complete": True})
                buffer = ""
                inside = False
            else:
                buffer += part
        else:
            buffer += part
    if inside:
        blocks.append({"text": buffer, "complete": False})
    else:
        answer.append(buffer)
    return {
        "content": "".join(answer).strip(),
        "reasoning_blocks": blocks,
        "reasoning": "\n\n".join(
            b["text"].strip() for b in blocks if b["text"].strip()
        ),
        "raw": raw,
    }


def build_payload(model, chat, store):
    settings = chat.settings
    messages, images, audios, ratios, insertions = [], [], [], [], []
    if settings.system.strip():
        messages.append({"role": "system", "content": settings.system})
    beebee = model["adapter"] == "beebee"
    for message in chat.messages:
        if message.role == "assistant":
            messages.append({"role": "assistant", "content": message.content})
            continue
        media = prepare_media(store, message.attachments, settings)
        content, vision, audio_tags = [], [], []
        media_index = 0
        while media_index < len(media):
            item = media[media_index]
            if (
                beebee
                and settings.video_layout in ("fps_groups", "single_segment")
                and "timestamp" in item
            ):
                frames = []
                video_id = item.get("video_id")
                group = item.get("video_group")
                while (
                    media_index < len(media)
                    and "timestamp" in media[media_index]
                    and (
                        media[media_index].get("video_id") == video_id
                        and (
                            settings.video_layout == "single_segment"
                            or media[media_index].get("video_group") == group
                        )
                    )
                ):
                    frames.append(media[media_index])
                    media_index += 1
                images.extend(frame["data"] for frame in frames)
                ratios.extend([settings.downsample] * len(frames))
                vision.append("<|vision_start|>" + "<image>" * len(frames) + "<|vision_end|>")
                continue
            if item["kind"] == "image":
                if beebee:
                    images.extend([item["data"], item["data"]])
                    ratios.append(settings.downsample)
                    stamp = f"{item['timestamp']:.2f}s" if "timestamp" in item else ""
                    vision.append(
                        stamp + "<|vision_start|><image><image><|vision_end|>"
                    )
                else:
                    if "timestamp" in item:
                        content.append(
                            {
                                "type": "text",
                                "text": f"Video frame at {item['timestamp']:.2f}s:",
                            }
                        )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + item["data"]
                            },
                        }
                    )
            else:
                if beebee:
                    audios.append(item["data"])
                    audio_tags.append("<audio>")
                else:
                    content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": item["data"], "format": "wav"},
                        }
                    )
            media_index += 1
        if beebee:
            # A unique sentinel locates this user turn after native rendering.
            # This also works when no explicit system message was supplied.
            marker = f"WEBUI_MEDIA_{uuid.uuid4().hex}_"
            insertions.append((marker, "".join(vision)))
            messages.append(
                {
                    "role": "user",
                    "content": marker + "".join(audio_tags) + message.content,
                }
            )
        else:
            content.append({"type": "text", "text": message.content})
            messages.append(
                {"role": "user", "content": content if media else message.content}
            )
    if beebee:
        tokenizer = tokenizer_for(model["tokenizer_path"])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.thinking,
        )
        for marker, vision in insertions:
            needle = "<|im_start|>user\n" + marker
            if prompt.count(needle) != 1:
                raise ValueError(
                    "此 BeeBee tokenizer 的模板不兼容，请检查 tokenizer_path"
                )
            # Keep media outside the user turn, matching the trained protocol.
            if vision:
                if "\n" + needle in prompt:
                    needle = "\n" + needle
                prompt = prompt.replace(needle, vision + "\n<|im_start|>user\n", 1)
            else:
                prompt = prompt.replace(marker, "", 1)
        payload = {
            "text": prompt,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": 20,
                "presence_penalty": settings.presence_penalty,
                "stop": ["<|im_end|>"],
            },
        }
        if images:
            payload.update(image_data=images, image_downsample_ratios=ratios)
        if audios:
            payload["audio_data"] = audios
        return "/generate", payload
    return "/v1/chat/completions", {
        "model": model["model"],
        "messages": messages,
        "stream": True,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "presence_penalty": settings.presence_penalty,
        "chat_template_kwargs": {"enable_thinking": settings.thinking},
    }


async def sse_events(response):
    """SSE supports comments, multiple data lines, and arbitrary TCP chunking."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


async def chat_events(model, chat, store):
    yield {"type": "status", "message": "正在准备媒体与对话…"}
    try:
        endpoint, payload = await asyncio.to_thread(build_payload, model, chat, store)
        key = os.getenv(model.get("api_key_env", ""), "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        yield {"type": "status", "message": "模型正在生成…"}
        raw, reasoning, usage, finish = "", "", {}, None
        beebee = model["adapter"] == "beebee"
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(300, connect=5), trust_env=False
            ) as client,
            client.stream(
                "POST", model["url"] + endpoint, json=payload, headers=headers
            ) as response,
        ):
            if not response.is_success:
                detail = (await response.aread()).decode(errors="replace")[:1000]
                raise ValueError(f"模型服务返回 HTTP {response.status_code}：{detail}")
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ValueError("模型服务没有返回 SSE 流，请确认服务支持 stream=true")
            received = False
            terminated = False
            async for event in sse_events(response):
                if event == "[DONE]":
                    terminated = True
                    break
                chunk = json.loads(event)
                if "error" in chunk:
                    raise ValueError(str(chunk["error"])[:1000])
                received = True
                if beebee:
                    raw = chunk.get("text", raw)  # SGLang sends cumulative text.
                    meta = chunk.get("meta_info", {})
                    usage = {
                        k: meta[k]
                        for k in ("prompt_tokens", "completion_tokens")
                        if k in meta
                    } or usage
                    reason = meta.get("finish_reason") or {}
                    finish = reason.get("type") or finish
                    if finish == "abort":
                        raise ValueError(str(reason))
                    output = split_reasoning(raw, chat.settings.thinking)
                else:
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    raw += delta.get("content") or ""
                    reasoning += (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    finish = choice.get("finish_reason") or finish
                    usage = chunk.get("usage") or usage
                    output = split_reasoning(raw)
                    output["raw_reasoning"] = reasoning
                    if reasoning:
                        output["reasoning"] = reasoning + (
                            "\n\n" + output["reasoning"] if output["reasoning"] else ""
                        )
                yield {
                    "type": "snapshot",
                    **output,
                    "usage": usage,
                    "finish_reason": finish,
                }
            if not received:
                raise ValueError("模型服务返回了空数据流")
            if not terminated and not finish:
                raise ValueError("模型数据流提前结束，已保留收到的内容，请重试")
        yield {"type": "done", "usage": usage, "finish_reason": finish}
    except asyncio.CancelledError:
        raise  # Closing the downstream stream also closes the upstream connection.
    except Exception as exc:
        log.exception("Chat request failed")
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
