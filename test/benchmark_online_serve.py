"""Benchmark multi-turn stateful serving throughput with Prefix Caching (SGLang).

真实线上多轮场景模拟：
1. 每个用户维护独立的持久化会话状态（文本历史、图片历史列表、音频历史列表）。
2. 每一轮次的 Stage 1 (Warmup) 追加图片建立 Cache。
3. Stage 2 (Interactive) 模拟动态多模态输入（文本/音频按分布出现）。
4. Token 预算精确滑窗机制：图像超标按间隔丢弃，文本超标按轮次剔除。
5. 修复：彻底解决随机文本生成导致的 <image> 占位符不对齐问题。

复现随机采样：两次运行使用相同参数和 --seed 42（默认）。
每用户的行为、输入文本、模拟回复、媒体使用独立随机流；媒体在线程提交前分配种子。
仍为闭环压测：响应耗时决定下一轮开始时间，实际输出长度影响历史上下文，
--active-time 内完成的轮数可能不同。固定种子不等同于固定请求轨迹回放。
"""
import argparse
import asyncio
import json
import random
import time
import io
import base64
import math
from typing import AsyncGenerator, List, Tuple, Optional, Union, Dict

import aiohttp
import numpy as np
from PIL import Image
import soundfile as sf
from transformers import AutoTokenizer
import concurrent.futures

REQUEST_LATENCY: List[Tuple[str, int, int, float, float]] = []


def log_request_token_budget(
    req_type, prompt, images, audios, text_tokens, args, recent_images_count=0
):
    """Validate and print the client-side token budget immediately before sending."""
    image_tokens = [int(img["tokens"]) for img in images]
    audio_tokens = [int(audio["tokens"]) for audio in audios]
    image_sum = sum(image_tokens)
    audio_sum = sum(audio_tokens)
    total_tokens = int(text_tokens) + image_sum + audio_sum
    recent_images_count = min(max(0, int(recent_images_count)), len(images))
    retained_history_images = len(images) - recent_images_count
    if image_sum > args.image_token_bucket:
        raise AssertionError(
            f"{req_type} image budget exceeded before send: "
            f"{image_sum}>{args.image_token_bucket} ({len(images)} images)"
        )
    if prompt.count("<image>") != len(images):
        raise AssertionError(
            f"{req_type} image tag/asset mismatch before send: "
            f"tags={prompt.count('<image>')} assets={len(images)}"
        )
    if prompt.count("<audio>") != len(audios):
        raise AssertionError(
            f"{req_type} audio tag/asset mismatch before send: "
            f"tags={prompt.count('<audio>')} assets={len(audios)}"
        )
    print(
        f"[CLIENT TOKEN COUNT] type={req_type} "
        f"images={len(images)}  (sum={image_sum}), "
        f"audios={len(audios)}  (sum={audio_sum}), "
        f"text_tokens={int(text_tokens)}, total_input_tokens={total_tokens}, "
        f"recent_images={recent_images_count}, history_images={retained_history_images}, "
        f"downsample_ratio={args.downsample_ratio}, "
        f"image_bucket={args.image_token_bucket}",
        flush=True,
    )

def get_tokenizer(tokenizer_name: str, *args, **kwargs):
    kwargs["use_fast"] = True
    return AutoTokenizer.from_pretrained(tokenizer_name, *args, **kwargs)

def gen_random_input_text(target_token_len: int, tokenizer, rng=None) -> str:
    """生成随机文本，并实施严格的特殊占位符清洗"""
    if target_token_len <= 0: return ""
    rng = rng or random
    random_ids = [rng.randint(512, 8192) for _ in range(target_token_len)]
    text = tokenizer.decode(random_ids)
    
    # 核心修复 1：严格防止随机生成的文本恰好碰撞出系统的关键控制标签
    # 否则会导致 SGLang 在解析 Prompt 和 Media 资产时长度对不上！
    text = text.replace("<image>", "") \
               .replace("<audio>", "") \
               .replace("<|im_end|>", "") \
               .replace("<|im_start|>", "")
    return text

def get_adaptive_pool_size(M, N, scale=16):
    r = 1 / math.sqrt(scale)
    return max(1, int(np.round(M * r))), max(1, int(np.round(N * r)))

def get_image_token_count(image_width: int, image_height: int, patch_size, downsample_ratio) -> int:
    h, w = get_adaptive_pool_size(image_height // patch_size // 2, image_width // patch_size // 2, scale=downsample_ratio)
    return (h * w) // 2

def compute_image_resolution(target_tokens: int, patch_size: int, downsample_ratio: int,
                             aspect_ratio: float = 16 / 9) -> Tuple[int, int]:
    """从目标 token 数 + patch_size/downsample_ratio 反推图像分辨率 (width, height)"""
    r = 1 / math.sqrt(downsample_ratio)
    hw_pool = target_tokens * 2  # h_pool * w_pool
    h_pool = max(1, round(math.sqrt(hw_pool / aspect_ratio)))
    w_pool = max(1, round(hw_pool / h_pool))

    # 反推 adaptive pool 前的 M, N
    M = max(1, round(h_pool / r))
    N = max(1, round(w_pool / r))

    height = M * patch_size * 2
    width = N * patch_size * 2

    actual = get_image_token_count(width, height, patch_size, downsample_ratio)
    if actual != target_tokens:
        print(f"[WARN] 目标 {target_tokens} tokens, 实际 {actual} tokens (分辨率 {width}x{height}, patch={patch_size}, ds={downsample_ratio})")
    return width, height

def get_audio_token_count(audio_sec: float) -> int:
    audio_downsample_ratio = 10
    data_len = int(16000 * audio_sec)
    return (math.ceil(data_len / 320) + audio_downsample_ratio) // audio_downsample_ratio

def apply_eviction_policies(
    history_text: str, 
    history_images: List[Dict], 
    history_audios: List[Dict], 
    recent_images_count: int, 
    tokenizer,
    args
) -> Tuple[str, List[Dict], List[Dict]]:
    
    # --- 1. 图像预算淘汰
    # This is a hard request/history budget. Images are paired, and the newest
    # turn is protected while the oldest historical pairs are evicted first.
    image_count = history_text.count("<image>")
    if image_count != len(history_images):
        raise ValueError(
            "image tag/asset mismatch before eviction: "
            f"tags={image_count}, assets={len(history_images)}"
        )
    if len(history_images) % 2:
        raise ValueError(
            f"image assets must stay paired, got odd count={len(history_images)}"
        )

    budget = max(0, args.image_token_bucket)
    keep = [True] * len(history_images)
    total_img_tokens = sum(img["tokens"] for img in history_images)
    recent_images_count = max(0, int(recent_images_count))
    recent_images_count -= recent_images_count % 2
    recent_start = max(0, len(history_images) - recent_images_count)

    # Keep the current turn whenever it fits. Evict the oldest historical
    # pairs first, so new images always replace old images in the sliding
    # window rather than being discarded by the budget pass.
    recent_tokens = sum(img["tokens"] for img in history_images[recent_start:])
    if recent_tokens <= budget:
        drop_order = list(range(0, recent_start, 2))
    else:
        # If the current turn alone exceeds the bucket, trimming it is the
        # only possible fallback; normally this branch is not expected.
        print(
            f"[WARN] current image turn exceeds image bucket: "
            f"{recent_tokens}>{budget}; trimming current-turn images",
            flush=True,
        )
        drop_order = list(range(0, recent_start, 2)) + list(
            range(recent_start, len(history_images), 2)
        )
    for i in drop_order:
        if total_img_tokens <= budget:
            break
        # Each processor grid corresponds to two consecutive image assets.
        pair_tokens = history_images[i]["tokens"] + history_images[i + 1]["tokens"]
        keep[i] = keep[i + 1] = False
        total_img_tokens -= pair_tokens

    if recent_tokens <= budget and not all(keep[recent_start:]):
        raise AssertionError("current-turn images cannot be evicted")

    if not all(keep):
        parts = history_text.split("<image>")
        new_parts = [parts[0]]
        new_images = []
        for i, image in enumerate(history_images):
            if keep[i]:
                new_images.append(image)
                new_parts.append("<image>" + parts[i + 1])
            else:
                new_parts.append(parts[i + 1])
        history_images = new_images
        history_text = "".join(new_parts)

    # Verify the hard bound and multimodal alignment after pruning.
    final_img_tokens = sum(img["tokens"] for img in history_images)
    if final_img_tokens > budget:
        raise AssertionError(
            f"image token budget enforcement failed: {final_img_tokens}>{budget}"
        )

    # --- 2. 文本预算淘汰 (基于 im_start 安全拆分) ---
    # 排除 system prompt 的 token 数，只统计对话轮次的文本预算
    blocks = history_text.split("<|im_start|>")
    # blocks[0] 是空串, blocks[1] 是 system block，从 blocks[2:] 开始是对话轮次
    conversation_text = "<|im_start|>".join([""] + blocks[2:]) if len(blocks) > 2 else ""
    base_text_tokens = len(tokenizer.encode(conversation_text.replace("<image>", "").replace("<audio>", "")))
    total_text_tokens = base_text_tokens + sum(a["tokens"] for a in history_audios)

    if total_text_tokens > args.text_token_bucket:
        while total_text_tokens > args.text_token_bucket // 2 and len(blocks) > 2:
            # 弹出最早的对话块 (blocks[0]是空字符串, blocks[1]是system prompt, 所以弹 blocks[2])
            dropped_block = "<|im_start|>" + blocks.pop(2)
            
            img_c = dropped_block.count("<image>")
            aud_c = dropped_block.count("<audio>")
            
            history_images = history_images[img_c:]
            
            dropped_audio_tokens = sum(a["tokens"] for a in history_audios[:aud_c])
            history_audios = history_audios[aud_c:]
            
            dropped_text_tokens = len(tokenizer.encode(dropped_block.replace("<image>", "").replace("<audio>", "")))
            total_text_tokens -= (dropped_text_tokens + dropped_audio_tokens)
            
        history_text = "<|im_start|>".join(blocks)
        
    return history_text, history_images, history_audios


class DynamicDataPool:
    def __init__(self, patch_size: int, downsample_ratio: int, max_workers=16):
        print(f"Initializing Dynamic Data Generator (On-the-fly mode with {max_workers} threads)...")
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def _gen_image_sync(self, width: int, height: int, seed: int) -> dict:
        rng = random.Random(seed)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        img = Image.new("RGB", (width, height), color=color)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        tok_count = get_image_token_count(width, height, self.patch_size, self.downsample_ratio)
        return {"b64": b64_str, "tokens": tok_count}

    def _gen_audio_sync(self, duration: float, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        audio_data = rng.uniform(-1, 1, int(16000 * duration)).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 16000, format='WAV')
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        tok_count = get_audio_token_count(duration)
        return {"b64": b64_str, "tokens": tok_count, "duration": duration}

    async def get_random_images_data(
        self, n: int, small_ratio: float, small_res: tuple, large_res: tuple, rng=None
    ) -> List[dict]:
        """按照 6:2 的周期规律排列生成图像序列 (6小 -> 2大 -> 6小...)"""
        n = (n // 2) * 2
        if n <= 0: 
            return []
            
        loop = asyncio.get_running_loop()
        tasks = []
        rng = rng or random
        
        for i in range(n):
            if i % 8 < 6:
                tasks.append(
                    loop.run_in_executor(
                        self.executor, self._gen_image_sync, *small_res, rng.getrandbits(64)
                    )
                )
            else:
                tasks.append(
                    loop.run_in_executor(
                        self.executor, self._gen_image_sync, *large_res, rng.getrandbits(64)
                    )
                )
            
        results = await asyncio.gather(*tasks)
        return results

    async def get_audios_matching_duration(self, target_duration: float, n: int, rng=None) -> List[dict]:
        if n <= 0: 
            return []
        loop = asyncio.get_running_loop()
        rng = rng or random
        tasks = [
            loop.run_in_executor(
                self.executor, self._gen_audio_sync, target_duration, rng.getrandbits(64)
            )
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

async def send_request_sglang(
    req_type,
    api_url,
    prompt,
    prompt_len,
    images_b64_list,
    audios_b64_list,
    max_output_token,
    image_downsample_ratio,
) -> int:
    # --- 新增：发送前的严格一致性预检 (Pre-flight Check) ---
    img_tag_count = prompt.count("<image>")
    audio_tag_count = prompt.count("<audio>")
    
    img_list_len = len(images_b64_list) if images_b64_list else 0
    audio_list_len = len(audios_b64_list) if audios_b64_list else 0
    
    if img_tag_count != img_list_len or audio_tag_count != audio_list_len:
        error_msg = (
            f"\n[FATAL ERROR - {req_type.upper()}] 模态标签与物理资产数量不匹配！\n"
            f"  - <image> 标签数量: {img_tag_count}, 图片数组长度: {img_list_len}\n"
            f"  - <audio> 标签数量: {audio_tag_count}, 音频数组长度: {audio_list_len}\n"
        )
        print(error_msg)
        raise ValueError(error_msg)
    # -------------------------------------------------------------

    headers = {"User-Agent": "Test Client"}
    pload = {"text": prompt, 
             "sampling_params": {"max_new_tokens": max_output_token, "temperature": 0.1}, 
             "stream": True}
    
    if images_b64_list: 
        pload["image_data"] = images_b64_list
        # BeeBee groups images in pairs and expects one ratio per pair.
        pload["image_downsample_ratios"] = [
            image_downsample_ratio
        ] * ((len(images_b64_list) + 1) // 2)
    if audios_b64_list:
        pload["audio_data"] = audios_b64_list

    request_start_time = time.time()
    first_token_time, output_token_len = 0, 0
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3*3600)) as session:
        try:
            async with session.post(api_url, headers=headers, json=pload) as response:
                async for line in response.content:
                    if line:
                        data = line.decode().strip()
                        if data.startswith("data:"):
                            if first_token_time == 0: first_token_time = time.time() - request_start_time
                            line = data[5:].strip()
                            if "[DONE]" in line: break
                            try:
                                json_data = json.loads(line)
                                if "meta_info" in json_data and "completion_tokens" in json_data["meta_info"]:
                                    output_token_len = json_data["meta_info"]["completion_tokens"]
                            except: pass
        except Exception as e:
            print(f"Request failed: {e}")
                                           
    request_latency = time.time() - request_start_time
    if req_type == "interactive":
        print(f"[{req_type.upper()}] PromptLen: {prompt_len}, OutputLen: {output_token_len}, TTFT: {first_token_time:.3f}s")
    REQUEST_LATENCY.append((req_type, prompt_len, output_token_len, request_latency, first_token_time))
    return output_token_len


async def simulated_user_session(
    user_id: int,
    api_url: str,
    tokenizer,
    data_pool: DynamicDataPool,
    args: argparse.Namespace,
    user_rng: random.Random,
    user_np_rng: np.random.Generator,
):
    session_start_time = time.time()
    # Separate streams keep generated output length and thread scheduling from
    # changing subsequent behavior samples or new input media/text.
    text_rng = random.Random(f"{args.seed}:{user_id}:text")
    reply_rng = random.Random(f"{args.seed}:{user_id}:reply")
    media_rng = random.Random(f"{args.seed}:{user_id}:media")
    
    system_prompt_body = "You are a helpful assistant.\n"
    if args.system_prompt_tokens > 0:
        system_prompt_body += gen_random_input_text(
            args.system_prompt_tokens, tokenizer, text_rng
        )
    history_text = f"<|im_start|>system\n{system_prompt_body}<|im_end|>\n"
    history_images = [] 
    history_audios = [] 
    
    user_type_rand = user_rng.random()
    # user_silence_scale = 5.0 if user_type_rand < 0.2 else (60.0 if user_type_rand > 0.8 else 15.0) # 60% 15s, 20% 5s, 20%60s
    user_silence_scale = 8.0
    print(
        f"[USER {user_id}] seed={args.seed + user_id}, "
        f"silence_scale={user_silence_scale}s", flush=True
    )
    small_res = (args.small_image_width, args.small_image_height)
    large_res = (args.large_image_width, args.large_image_height)
    
    while time.time() - session_start_time < args.active_time:
        user_silence_time = user_np_rng.exponential(scale=user_silence_scale)
        system_patience_time = max(2.0, user_np_rng.normal(loc=12.0, scale=3.0))  # 12 +- 3s
        
        if system_patience_time < user_silence_time:
            # --- 系统主动 ---
            await asyncio.sleep(system_patience_time)

            if args.request_mode == "text-only":
                proactive_image_count = 0
                turn_images = []
            else:
                proactive_image_count = (args.proactive_image_count // 2) * 2
                t1 = time.perf_counter()
                turn_images = await data_pool.get_random_images_data(
                    proactive_image_count, args.small_image_ratio, small_res, large_res, rng=media_rng
                )
                # print(f"get {len(turn_images)} random image time:", time.perf_counter() - t1)

            turn_text = ("<image>" * proactive_image_count) + "\n<|im_start|>assistant\n"
            current_prompt = history_text + turn_text

            current_images = history_images + turn_images
            current_audios = history_audios
            current_prompt, current_images, current_audios = apply_eviction_policies(
                current_prompt,
                current_images,
                current_audios,
                recent_images_count=proactive_image_count,
                tokenizer=tokenizer,
                args=args,
            )
            
            text_tokens = len(tokenizer([current_prompt.replace("<image>", "").replace("<audio>", "")]).input_ids[0])
            total_tokens = text_tokens + sum(img["tokens"] for img in current_images) + sum(a["tokens"] for a in current_audios)
            
            max_out = min(
                args.max_output_token,
                max(8, int(user_np_rng.gamma(shape=args.gamma_shape, scale=args.gamma_scale))),
            )
            log_request_token_budget(
                "proactive",
                current_prompt,
                current_images,
                current_audios,
                text_tokens,
                args,
                recent_images_count=proactive_image_count,
            )
            
            actual_out_len = await send_request_sglang(
                "proactive", api_url, current_prompt, total_tokens, 
                [img["b64"] for img in current_images],
                [a["b64"] for a in current_audios],
                max_output_token=max_out,
                image_downsample_ratio=args.downsample_ratio,
            )
            
            assistant_reply = gen_random_input_text(actual_out_len, tokenizer, reply_rng)
            history_text = current_prompt + assistant_reply + "<|im_end|>\n"
            history_images = current_images
            
            history_text, history_images, history_audios = apply_eviction_policies(
                history_text, history_images, history_audios, proactive_image_count, tokenizer, args
            )
            continue

        else:
            # --- 用户主动交互 ---
            await asyncio.sleep(user_silence_time)

            # [Stage 1: Warmup]
            if args.request_mode == "text-only":
                warmup_images_count = 0
                turn_warmup_images = []
            else:
                warmup_images_count = (args.warmup_images // 2) * 2
                t1 = time.perf_counter()
                turn_warmup_images = await data_pool.get_random_images_data(
                    warmup_images_count, args.small_image_ratio, small_res, large_res, rng=media_rng
                )
                # print(f"get {len(turn_warmup_images)} random image time:", time.perf_counter() - t1)

            stage_1_text = history_text + f"<|im_start|>user\n" + ("<image>" * warmup_images_count) + "\n"
            stage_1_images = history_images + turn_warmup_images
            stage_1_text, stage_1_images, stage_1_audios = apply_eviction_policies(
                stage_1_text,
                stage_1_images,
                history_audios,
                recent_images_count=warmup_images_count,
                tokenizer=tokenizer,
                args=args,
            )
            text_tokens_s1 = len(tokenizer([stage_1_text.replace("<image>", "").replace("<audio>", "")]).input_ids[0])
            total_tokens_s1 = text_tokens_s1 + sum(img["tokens"] for img in stage_1_images) + sum(a["tokens"] for a in stage_1_audios)

            log_request_token_budget(
                "warmup",
                stage_1_text,
                stage_1_images,
                stage_1_audios,
                text_tokens_s1,
                args,
                recent_images_count=warmup_images_count,
            )

            await send_request_sglang(
                "warmup", api_url, stage_1_text, total_tokens_s1,
                [img["b64"] for img in stage_1_images],
                [a["b64"] for a in stage_1_audios],
                max_output_token=1,
                image_downsample_ratio=args.downsample_ratio,
            )

            # [Stage 2: 采样]
            speak_duration = min(
                max(2.0, user_np_rng.lognormal(mean=args.speak_mean, sigma=args.speak_sigma)),
                30.0,
            )
            await asyncio.sleep(speak_duration)

            if args.request_mode == "text-only":
                active_image_count = 0
                turn_active_images = []
                turn_num_audios = 0
                user_spoken_text = ""
                target_text_len = user_rng.randint(args.prompt_len_min, args.prompt_len_max)
                user_spoken_text = gen_random_input_text(target_text_len, tokenizer, text_rng)
            else:
                active_image_count = min(args.max_active_images, max(2, int(speak_duration * args.image_sample_rate)))
                active_image_count = (active_image_count // 2) * 2

                t1 = time.perf_counter()
                turn_active_images = await data_pool.get_random_images_data(
                    active_image_count, args.small_image_ratio, small_res, large_res, rng=media_rng
                )
                # print(f"get {len(turn_active_images)} random image time:", time.perf_counter() - t1)

                modality_rand = user_rng.random()
                turn_num_audios = 0
                user_spoken_text = ""

                if args.request_mode == "multimodal-only":
                    # multimodal-only 模式下强制带音频，不发纯文本
                    turn_num_audios = args.num_audios
                elif modality_rand < 0.3:
                    target_text_len = user_rng.randint(args.prompt_len_min, args.prompt_len_max)
                    user_spoken_text = gen_random_input_text(target_text_len, tokenizer, text_rng)
                else:
                    turn_num_audios = args.num_audios

            turn_active_audios = await data_pool.get_audios_matching_duration(
                speak_duration, turn_num_audios, rng=media_rng
            )

            stage_2_append_tags = ("<image>" * active_image_count) + ("<audio>" * turn_num_audios)

            if user_spoken_text:
                stage_2_text = stage_1_text + stage_2_append_tags + "\n" + user_spoken_text + "<|im_end|>\n<|im_start|>assistant\n"
            else:
                stage_2_text = stage_1_text + stage_2_append_tags + "<|im_end|>\n<|im_start|>assistant\n"

            # print("stage2 text", stage_2_text[:500])
            stage_2_images = stage_1_images + turn_active_images
            stage_2_audios = stage_1_audios + turn_active_audios
            stage_2_text, stage_2_images, stage_2_audios = apply_eviction_policies(
                stage_2_text,
                stage_2_images,
                stage_2_audios,
                recent_images_count=active_image_count + warmup_images_count,
                tokenizer=tokenizer,
                args=args,
            )
            text_tokens_s2 = len(tokenizer([stage_2_text.replace("<image>", "").replace("<audio>", "")]).input_ids[0])
            total_tokens_s2 = text_tokens_s2 + sum(img["tokens"] for img in stage_2_images) + sum(a["tokens"] for a in stage_2_audios)

            log_request_token_budget(
                "interactive",
                stage_2_text,
                stage_2_images,
                stage_2_audios,
                text_tokens_s2,
                args,
                recent_images_count=warmup_images_count + active_image_count,
            )

            max_out = min(
                args.max_output_token,
                max(10, int(user_np_rng.gamma(shape=args.gamma_shape, scale=args.gamma_scale))),
            )
            actual_out_len = await send_request_sglang(
                "interactive", api_url, stage_2_text, total_tokens_s2,
                [img["b64"] for img in stage_2_images],
                [a["b64"] for a in stage_2_audios],
                max_output_token=max_out,
                image_downsample_ratio=args.downsample_ratio,
            )

            assistant_reply = gen_random_input_text(actual_out_len, tokenizer, reply_rng)
            history_text = stage_2_text + assistant_reply + "<|im_end|>\n"
            history_images = stage_2_images
            history_audios = stage_2_audios

            history_text, history_images, history_audios = apply_eviction_policies(
                history_text, history_images, history_audios,
                recent_images_count=(warmup_images_count + active_image_count),
                tokenizer=tokenizer, args=args
            )
            

async def run_benchmark(args):
    REQUEST_LATENCY.clear()
    # Local RNGs avoid cross-user scheduling dependence. Real arrival times
    # still depend on response latency in this closed-loop benchmark.
    startup_rng = np.random.default_rng(args.seed)
    api_url = f"http://{args.host}:{args.port}/generate_stream"
    tokenizer = get_tokenizer(args.tokenizer, "fast")

    # 从 token 数动态计算图像分辨率
    args.small_image_width, args.small_image_height = compute_image_resolution(
        args.small_image_tokens, args.patch_size, args.downsample_ratio, args.image_aspect_ratio)
    args.large_image_width, args.large_image_height = compute_image_resolution(
        args.large_image_tokens, args.patch_size, args.downsample_ratio, args.image_aspect_ratio)

    data_pool = DynamicDataPool(patch_size=args.patch_size, downsample_ratio=args.downsample_ratio, max_workers=16)

    print(f"\n--- Starting Multi-Turn Stateful Benchmark ---")
    print(f"Backend: SGLang")
    print(f"Request Mode: {args.request_mode}")
    print(f"Concurrent Users: {args.num_users}")
    print(f"Random Seed: {args.seed} (closed-loop; arrival times are response-dependent)")
    print(f"Test Duration: {args.active_time} seconds")
    print(f"Max Context Limit: {args.max_context_len} tokens")
    print(f"Patch Size: {args.patch_size}, Downsample Ratio: {args.downsample_ratio}")
    print(f"Small Image: {args.small_image_width}x{args.small_image_height} ({args.small_image_tokens} tokens)")
    print(f"Large Image: {args.large_image_width}x{args.large_image_height} ({args.large_image_tokens} tokens)")
   
    
    benchmark_start_time = time.time()
    
    tasks = []
    for i in range(args.num_users):
        user_rng = random.Random(args.seed + i)
        user_np_rng = np.random.default_rng(args.seed + i)
        task = asyncio.create_task(
            simulated_user_session(
                i, api_url, tokenizer, data_pool, args, user_rng, user_np_rng
            )
        )
        tasks.append(task)
        await asyncio.sleep(startup_rng.exponential(1.0))
        
    await asyncio.gather(*tasks)
    
    benchmark_time = time.time() - benchmark_start_time
    print(f"\n{'='*50}")
    print(f"Benchmark finished in {benchmark_time:.2f} s")
    
    interactive_reqs = [r for r in REQUEST_LATENCY if r[0] == "interactive"]
    if not interactive_reqs:
        print("No interactive requests completed.")
        return
        
    print("\n--- [Stage 2] Multi-Turn Interactive Generation Stats ---")
    interactive_ttft = [r[4] for r in interactive_reqs if r[4] > 0]
    interactive_tpot = [(r[3] - r[4]) / r[2] for r in interactive_reqs if r[2] > 1 and r[4] > 0]
    
    print(f"Completed interactive turns: {len(interactive_reqs)}")
    if interactive_ttft:
        print(f"Average TTFT: {np.mean(interactive_ttft):.4f} s")
        print("TTFT Quantiles:")
        for q, val in zip([0.5, 0.90, 0.99], np.quantile(interactive_ttft, [0.5, 0.90, 0.99])):
            print(f"  P{int(q*100)}: {val:.4f} s")
            
    if interactive_tpot:
        avg_tpot = np.mean(interactive_tpot)
        print(f"\nAverage TPOT (Time Per Output Token): {avg_tpot * 1000:.2f} ms")
        print(f"Equivalent Output Speed: {1 / avg_tpot:.1f} tokens/s")

def main():
    parser = argparse.ArgumentParser(description="Multi-Turn Stateful Benchmark for MLLM (SGLang)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", type=str, required=True)
    
    # Session 宏观参数
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="非负随机种子；固定每用户的随机采样，不固定实际请求到达时间",
    )
    parser.add_argument("--active-time", type=float, default=360)
    parser.add_argument("--max-context-len", type=int, default=8192)
    
    # 预算与淘汰机制参数
    parser.add_argument("--image-token-bucket", type=int, default=3500)
    parser.add_argument("--text-token-bucket", type=int, default=3000)
    
    # 图像生成与采样参数（分辨率由 token 数 + patch_size/downsample_ratio 动态计算）
    parser.add_argument("--small-image-tokens", type=int, default=9, help="每张小图的目标 token 数")
    parser.add_argument("--large-image-tokens", type=int, default=42, help="每张大图的目标 token 数")
    parser.add_argument("--image-aspect-ratio", type=float, default=16/9, help="图像宽高比")
    parser.add_argument("--small-image-ratio", type=float, default=0.75)
    
    parser.add_argument("--warmup-images", type=int, default=36)
    parser.add_argument("--proactive-image-count", type=int, default=20)
    parser.add_argument("--image-sample-rate", type=float, default=4.0, help="每秒语音采样几张图")
    parser.add_argument("--max-active-images", type=int, default=16, help="Stage 2 抽图上限")

    # 视觉编码器参数
    parser.add_argument("--patch-size", type=int, default=14, help="视觉编码器 patch size")
    parser.add_argument("--downsample-ratio", type=int, default=16, help="视觉编码器下采样比例")
    
    # 音频与对数正态分布参数
    parser.add_argument("--num-audios", type=int, default=1)
    parser.add_argument("--speak-mean", type=float, default=1.7)
    parser.add_argument("--speak-sigma", type=float, default=0.5)
    
    # 文本生成参数
    parser.add_argument("--prompt-len-min", type=int, default=5)
    parser.add_argument("--prompt-len-max", type=int, default=64)
    parser.add_argument("--max_output_token", type=int, default=64)
    parser.add_argument("--gamma-shape", type=float, default=2.0) # 均值30，众数15个字
    parser.add_argument("--gamma-scale", type=float, default=15.0)

    # System Prompt 参数
    parser.add_argument("--system-prompt-tokens", type=int, default=10000,
                        help="每个用户会话的 system prompt 固定填充 token 数（模拟真实长 system prompt）")

    # 请求模式控制（用于排查显存泄漏）
    parser.add_argument("--request-mode", type=str, default="all",
                        choices=["all", "text-only", "multimodal-only"],
                        help="控制请求类型: all=混合(默认), text-only=纯文本无图无音频, multimodal-only=每轮都带图/音频")
    
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    asyncio.run(run_benchmark(args))

if __name__ == "__main__":
    main()
