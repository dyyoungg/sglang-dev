"""Benchmark online serving throughput with Fake Data and Length Distribution.

On the server side, run your backend (vLLM, lightllm, sglang).

On the client side, run:
    python benchmark_serve.py \
        --backend lightllm \
        --tokenizer <your_model_path> \
        --prompt-len-min 512 \
        --prompt-len-max 2048 \
        --num-images 2 \
        --request-rate 1.0
"""
import argparse
import asyncio
import json
import random
import time
import io
import base64
import math
from typing import AsyncGenerator, List, Tuple, Optional, Union

import aiohttp
import numpy as np
from PIL import Image
import soundfile as sf
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# (prompt len, output len, latency, first_token_time)
REQUEST_LATENCY: List[Tuple[int, int, float, float]] = []


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    *args,
    **kwargs,
):
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args, **kwargs)
    except TypeError as e:
        raise RuntimeError("Failed to load the tokenizer.") from e

    return tokenizer


def gen_random_input_text(target_token_len: int, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> str:
    """Generate fake text by randomly sampling token IDs and decoding."""
    if target_token_len <= 0:
        return ""
    
    random_ids = [random.randint(512, 8192) for _ in range(target_token_len)]
    random_text = tokenizer.decode(random_ids)
    return random_text


def get_adaptive_pool_size(M, N, scale=16):
    r = 1 / math.sqrt(scale)
    Mh = max(1, int(np.round(M * r)))
    Nw = max(1, int(np.round(N * r)))
    return Mh, Nw

def generate_fake_requests(
    num_requests: int,
    tokenizer: PreTrainedTokenizer,
    prompt_len_min: int,
    prompt_len_max: int,
    num_images: int,
    image_width: int,
    image_height: int,
    num_audios: int,
    audio_sec: float,   
) -> List[Tuple[str, int, List[Image.Image], List[np.ndarray]]]:
    """Generates synthetic requests with distributed text lengths, images, and audio."""
    
    sampled_requests = []
    
    assert num_images % 2 == 0, "image nums must be divided by 2."

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    user_format = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
   
    for _ in range(num_requests):
        images_list = []
        for _ in range(num_images):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", (image_width, image_height), color=color)
            images_list.append(img)
            
        audios_list = []
        for _ in range(num_audios):
            sample_rate = 16000
            audio_data = np.random.uniform(-1, 1, int(sample_rate * audio_sec)).astype(np.float32)
            audios_list.append(audio_data)

        target_text_len = random.randint(prompt_len_min, prompt_len_max)
        base_text = gen_random_input_text(target_text_len, tokenizer)
        
    
        media_tags =  ("<image>" * num_images) +  ("<audio>" * num_audios)
        full_content = media_tags + "\n" + base_text
        question = system_prompt + user_format.format(content=full_content)

        text_token_ids = tokenizer([question.replace("<image>", "").replace("<audio>", "")]).input_ids[0]
        
        # Audio tokens estimation (based on chunking logic)
        audio_downsample_ratio = 10
        audio_token_nums = 0
        if len(audios_list) > 0:
            audio_token_nums = sum([(math.ceil(len(data) / 320) + audio_downsample_ratio) // audio_downsample_ratio for data in audios_list])
        h, w = get_adaptive_pool_size(image_height//14//2, image_width//14//2, scale=16)
        image_tokens = h*w* num_images // 2
        total_prompt_len = len(text_token_ids) + image_tokens + audio_token_nums
        
        sampled_requests.append((question, total_prompt_len, images_list, audios_list))

    print("Generate random fake data finish.")
    print(f"Total Requests: {len(sampled_requests)}")
    if sampled_requests:
        avg_len = np.mean([req[1] for req in sampled_requests])
        print(f"Average Total Prompt Token Length: {avg_len:.1f} (Min: {prompt_len_min}, Max: {prompt_len_max})")
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, List[Image.Image], List[np.ndarray]]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, List[Image.Image], List[np.ndarray]], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    model_dir: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    best_of: int,
    use_beam_search: bool,
    images_list: Optional[List[Image.Image]] = [],
    audios_list: Optional[List[np.ndarray]] = [],
    max_output_token: int = 1024
) -> None:
    
    if backend in ["vllm", "sglang"]:
        headers = {"User-Agent": "Test Client"}
        multi_modal_data = {}
        image_data = []
        if len(images_list):
            image_data = []
            for img in images_list:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data.append(img_base64)
            
            multi_modal_data["image_data"] = image_data 
            
        if len(audios_list):
            audio_data = []
            for audio in audios_list:
                buffer = io.BytesIO()
                sf.write(buffer, audio, 16000, format='WAV')
                audio_bs64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                audio_data.append(audio_bs64)
                
            multi_modal_data["audio_data"] = audio_data

        pload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_output_token},
            "stream": True,
        }
        
        if multi_modal_data:
            pload.update(multi_modal_data)
        
        
    elif backend == "lightllm":
        headers = {'Content-Type': 'application/json'}
        default_sampling_params = {
            "max_new_tokens": max_output_token,
            "stop_sequences": ["<|im_end|>", " <|im_end|>"],
            "do_sample": True,
            "repetition_penalty": 1.05,
            "temperature": 0.1,
        }
        images = []
        if len(images_list):
            for img in images_list:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({"type": "base64", "data": img_base64})
                
        audios = []
        if len(audios_list):
            for audio in audios_list:
                buffer = io.BytesIO()
                sf.write(buffer, audio, 16000, format='WAV')
                audio_bs64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                audios.append({"type": "base64", "data": audio_bs64})

        pload = {
            "inputs": prompt,
            "parameters": default_sampling_params,
        }
        pload.update({"multimodal_params": {
                "images": images,
                "audios": audios
        }})
    else:
        raise ValueError(f"Unknown backend: {backend}")

    request_start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    
    stream = True
    first_token_time = 0
    output_token_len = 0
    output = ""
    
    if not stream:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                
                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)
                request_end_time = time.time()
        
                if backend == "lightllm":
                    output_token_len = output.get("count_output_tokens", 0)
                elif backend in ["lmdeploy", "vllm", "sglang"]:
                    output_token_len = output.get("usage", {}).get("completion_tokens", 0)
                    
                print("## output token length:", output_token_len)
                print("#"*30)
                if "error" not in output:
                    break
    else:
        if backend not in ["sglang", "vllm"]:
            api_url = api_url + "_stream"
            
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, headers={'Accept': 'text/event-stream'}, json=pload) as response:
                async for line in response.content:
                    if line:
                        data = line.decode().strip()
                        if data.startswith("data:"):
                            if first_token_time == 0:
                                first_token_time = time.time() - request_start_time
                            line = data[5:].strip()
                        
                            if backend == "lightllm":
                                try:
                                    json_data = json.loads(line)
                                    if json_data.get("finished", False):
                                        request_end_time = time.time()
                                        output_token_len = json_data.get("token", {}).get("count_output_tokens", 0)
                                except:
                                    pass
                            elif backend == "sglang":
                                if "[DONE]" in line:
                                    request_end_time = time.time()
                                    break
                                try:
                                    json_data = json.loads(line)
                                    if "meta_info" in json_data and "completion_tokens" in json_data["meta_info"]:
                                         output_token_len = json_data["meta_info"]["completion_tokens"]
                                except:
                                    pass
                                           
        request_end_time = time.time()                      
        print(f"## Finished request. Output tokens: {output_token_len}") # 可视情况关闭打印避免刷屏
        
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_token_len, request_latency, first_token_time))


async def benchmark(
    backend: str,
    model_dir: str,
    api_url: str,
    input_requests: List[Tuple[str, int, List[Image.Image], List[np.ndarray]]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    max_output_token: int = 1024
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, images, audios = request
        task = asyncio.create_task(send_request(backend, 
                                                model_dir, 
                                                api_url, 
                                                prompt,
                                                prompt_len, 
                                                best_of, 
                                                use_beam_search, 
                                                images, 
                                                audios,
                                                max_output_token))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"

    tokenizer = get_tokenizer(args.tokenizer, "fast")
    
    input_requests = generate_fake_requests(
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        prompt_len_min=args.prompt_len_min,
        prompt_len_max=args.prompt_len_max,
        num_images=args.num_images,
        image_width=args.image_width,
        image_height=args.image_height,
        num_audios=args.num_audios,
        audio_sec=args.audio_sec,
    )

    benchmark_start_time = time.time()
    asyncio.run(benchmark(args.backend, 
                          args.tokenizer,
                          api_url, 
                          input_requests, 
                          args.best_of,
                          args.use_beam_search, 
                          args.request_rate, 
                          args.max_output_token))
    
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"\n{'='*40}")
    print(f"Total time: {benchmark_time:.4f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.4f} requests/s")

    if not REQUEST_LATENCY:
        print("No requests completed.")
        return

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _ in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")

    avg_output_length = np.mean([output_len for _, output_len, _, _ in REQUEST_LATENCY])
    print(f"Average output length: {avg_output_length:.4f} tokens")

    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len) if (prompt_len + output_len) > 0 else 0
        for prompt_len, output_len, latency, _ in REQUEST_LATENCY
    ])
    print(f"Average latency per total token: {avg_per_token_latency:.4f} s")
    
    avg_per_output_token_latency = np.mean([
        latency / output_len if output_len > 0 else 0
        for _, output_len, latency, _ in REQUEST_LATENCY
    ])
    print(f"Average latency per output token: {avg_per_output_token_latency:.4f} s")
    
    valid_first_times = [first_time for _, _, _, first_time in REQUEST_LATENCY if first_time > 0]
    if valid_first_times:
        avg_firsttoken_time = np.mean(valid_first_times)
        print(f"Average first token time: {avg_firsttoken_time:.4f} s")

        quantiles = [0.5, 0.75, 0.90, 0.95]
        first_token_times_np = np.array(valid_first_times)
        quantile_values = np.quantile(first_token_times_np, quantiles)
        print("First token time quantiles:")
        for q, val in zip(quantiles, quantile_values):
            print(f"  {int(q*100)} percentile: {val:.4f} s")
    else:
        print("No valid first token time data to compute.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput with distributed Fake Data.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "lightllm", "sglang"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", type=str, required=True, help="Name or path of the tokenizer.")
    
    parser.add_argument("--prompt-len-min", type=int, default=512, help="Minimum text prompt token length.")
    parser.add_argument("--prompt-len-max", type=int, default=2048, help="Maximum text prompt token length.")
    
    parser.add_argument("--num-images", type=int, default=1, help="Number of fake images per request.")
    parser.add_argument("--image-width", type=int, default=644, help="Width of the fake image.")
    parser.add_argument("--image-height", type=int, default=364, help="Height of the fake image.")
    parser.add_argument("--num-audios", type=int, default=0, help="Number of fake audios per request.")
    parser.add_argument("--audio-sec", type=float, default=2.0, help="Duration of fake audio in seconds.")
    
    parser.add_argument("--max_output_token", type=int, default=128)
    parser.add_argument("--template_type", type=str, default="llava")
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. (inf = concurrent burst)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)