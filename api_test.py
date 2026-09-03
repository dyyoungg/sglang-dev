import io
import os
import re
from tqdm import tqdm
import random
import time
import aiohttp
import base64
import json
from pathlib import Path
from typing import List, Optional, AsyncGenerator, Union, Tuple
import asyncio
from PIL import Image
import numpy as np
import cv2
import tempfile
from concurrent.futures import ThreadPoolExecutor
import av
import imageio.v3 as iio
import math
# ceph_client = CepthClient("/mnt/afs/yangdeyu/aoss_ydy_game.conf")


def get_video_frames(video_path, total_sample_frames, start_time=None, end_time=None, method="imageio", format=""):
    start_frame = 0

    if "s3://" not in video_path:
        if os.path.exists(video_path):
            video_file = video_path
        else:
            print("video path does not exist", video_path)
            return
    else:
        # video_file = ceph_client.Get(video_path)
        pass

    def get_fps(PIL_Image_object):
        """Returns the average framerate of a PIL Image object"""
        PIL_Image_object.seek(0)
        frames = 0
        while True:
            try:
                frames += 1
                PIL_Image_object.seek(PIL_Image_object.tell() + 1)
            except EOFError:
                return frames
        return None

    if method == "pyav":
        video_io = io.BytesIO(video_file)
        container = av.open(video_io)
        framerate = container.streams.video[0].average_rate  # get the frame rate
        time_base = container.streams.video[0].time_base  # get the time base
        frame_per_time_base = 1 / (framerate * time_base)
        duration = container.duration / 1000000
        frame_count = int(duration * framerate)
        end_frame = frame_count
        if start_time != None:
            start_frame = int(framerate * start_time)
        if end_time != None:
            end_frame = int(framerate * end_time)
        frame_count = end_frame - start_frame
        if frame_count == 0:
            start_frame = 0
            end_frame = frame_count
            frame_count = end_frame - start_frame
    elif method == "imageio":
        if isinstance(video_file, bytes):
            video_io = io.BytesIO(video_file)
        else:
            video_io = video_file
        if format == "gif":
            try:
                video = iio.imread(video_io, index=None)
                frame_count = video.shape[0]
            except:
                if isinstance(video_file, bytes):
                    video_io = io.BytesIO(video_file)
                else:
                    video_io = video_file
                gif_obj = Image.open(video_io)
                frame_count = get_fps(gif_obj)
                video = iio.imiter(video_file, plugin="pyav", thread_count=1)
            end_frame = frame_count
            framerate = 4
        else:
            container = av.open(video_io)
            meta_data = iio.immeta(video_file, index=None)
            if "duration" not in meta_data:
                if container.duration is not None:
                    meta_data["duration"] = container.duration / 1000000
                else:
                    meta_data["duration"] = container.duration
            container.close()
            video = iio.imiter(video_file, plugin="pyav", thread_count=1)
            frame_count = int(meta_data["duration"] * meta_data["fps"])
            end_frame = frame_count
            framerate = meta_data["fps"]
            if start_time != None:
                start_frame = int(meta_data["fps"] * start_time)
            if end_time != None:
                end_frame = int(meta_data["fps"] * end_time)
            frame_count = end_frame - start_frame
            if frame_count == 0:
                start_frame = 0
                end_frame = frame_count
                frame_count = end_frame - start_frame

    def get_seq_frames(total_num_frames, desired_num_frames, start_frame, end_frame, framerate):
        seg_size = float(total_num_frames - 1) / desired_num_frames
        seq = []
        for i in range(desired_num_frames):
            # Calculate the start and end indices of each segment
            start = int(seg_size * i)
            end = int(seg_size * (i + 1))
            # Append the middle index of the segment to the list
            index = (start + end) // 2 + start_frame
            if index < end_frame:
                seq.append(index)
            else:
                seq.append(end_frame)
        seq = list(set(seq))
        seq.sort()
        return seq

    frame_seq = get_seq_frames(frame_count, min(total_sample_frames, frame_count), start_frame, end_frame, framerate)
    raw_img_list = []

    if method == "pyav":
        for frame_number in frame_seq:
            target_time = frame_number / framerate
            target_frame = int(target_time / time_base)
            container.seek(target_frame, backward=True, stream=container.streams.video[0])
            while True:
                frame = next(container.decode(video=0))
                if frame.pts + frame_per_time_base > target_frame:
                    break
            image = frame.to_image()
            raw_img_list.append(image.convert("RGB"))
    elif method == "imageio":
        img_index = 0
        for idx, image in enumerate(video):
            if idx == frame_seq[img_index]:
                image = Image.fromarray(image)
                raw_img_list.append(image.convert("RGB"))
                img_index += 1
                if img_index == len(frame_seq):
                    break

    return raw_img_list, frame_count


def load_and_resize_image(image, target_size=(644, 364)) -> Image.Image:
    resized_image = image.resize(target_size, Image.BICUBIC)
    return resized_image


class MultiModalClient:
    def __init__(self, url: str, default_sampling_params: dict, logger=None):
        self.url = url
        self.default_sampling_params = default_sampling_params
        self._logger = logger or print

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        # SGLang 可以直接处理原始的 base64 字符串
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def encode_audio_to_base64(audio: Union[np.ndarray, str, bytes]) -> str:
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio_data = f.read()
        elif isinstance(audio, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, audio)
            audio_data = buffer.getvalue()
        elif isinstance(audio, bytes):
            audio_data = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        return base64.b64encode(audio_data).decode("utf-8")

    async def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        audios: Optional[List[Union[np.ndarray, str, bytes]]] = None,
        target_sizes: Optional[List[Tuple]] = None,
        image_downsample_ratios: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:

        image_data_list = []
        if images:
            for i, image in enumerate(images):
                if target_sizes is not None:
                    image = load_and_resize_image(image, target_sizes[i])
                else:
                    image = load_and_resize_image(image, target_size=(644, 364))
                img_b64 = self.encode_image_to_base64(image)
                image_data_list.append(img_b64)

        audio_data_list = []
        if audios:
            for audio in audios:
                audio_b64 = self.encode_audio_to_base64(audio)
                audio_data_list.append(audio_b64)

        # 适配 SGLang 的 Payload 结构
        payload = {
            "text": prompt,
            "sampling_params": self.default_sampling_params,
            "stream": True,
        }

        # SGLang 支持单图传入 string，多图传入 list
        if image_data_list:
            payload["image_data"] = image_data_list if len(image_data_list) > 1 else image_data_list[0]

        if audio_data_list:
            payload["audio_data"] = audio_data_list if len(audio_data_list) > 1 else audio_data_list[0]

        # 动态压缩率
        if image_downsample_ratios is not None:
            payload["image_downsample_ratios"] = image_downsample_ratios

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers={"Accept": "text/event-stream"}, json=payload) as response:
                    if response.status != 200:
                        err_msg = await response.text()
                        raise Exception(f"Request failed with status {response.status}: {err_msg}")

                    prev_len = 0
                    async for line in response.content:
                        if line:
                            line_dec = line.decode('utf-8').strip()
                            if line_dec.startswith("data:"):
                                if line_dec == "data: [DONE]":
                                    break
                                
                                # 解析 SGLang 返回的数据块
                                json_data = json.loads(line_dec[5:].strip())
                                full_text = json_data.get("text", "")
                                
                                # 计算增量文本并更新指针
                                new_text = full_text[prev_len:]
                                if new_text:
                                    yield new_text
                                    prev_len = len(full_text)
        except Exception as e:
            self._logger(f"Error during request: {e}")


def construct_prompt(query, image_num, audio_nums=0, system_prompt="You are a helpful AI assistant."):

    system_prompt_format = "<|im_start|>system\n{}<|im_end|>"
    user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    assistant_format = "{content}<|im_end|>"  # llm 回复完需要拼上这个

    system = system_prompt_format.format(system_prompt)

    if audio_nums == 0:
        user_prompt = user_format.format(content=query)
    else:
        user_prompt = user_format.format(content="<audio>" * audio_nums)  # audio token
    if image_num > 0:
        image_prompt = "<|vision_start|>" + "<image>" * image_num + "<|vision_end|>"
    else:
        image_prompt = ""
    total_prompt = system + image_prompt + user_prompt
    return total_prompt


def build_inputs_qwen2(prompt_list, system_prompt: str = "You are a helpful assistant."):
    prompt = ""
    system_prompt_format = "<|im_start|>system\n{content}<|im_end|>\n"

    user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    assistant_format = "{content}<|im_end|>"

    system_prompt = system_prompt_format.format(content=system_prompt)
    prompt += system_prompt

    for message in prompt_list:
        content = message["value"]
        if message["role"].lower() in ["human", "user"]:
            prompt += user_format.format(content=content)
        elif message["role"].lower() in ["assistant", "gpt"]:
            prompt += assistant_format.format(content=content)
        else:
            pass
    return prompt


def extract_frames_from_video(video_path, max_frames=8):
    """
    从本地或 S3 视频路径中抽取图像帧（最多 max_frames 张）。
    支持本地路径或 S3（通过 AOSS 读取）。
    """

    # client = CepthClient("/mnt/afs/yangdeyu/aoss_ydy_game.conf")

    is_temp_file = False

    # if "s3://" in video_path:
    #     video_bytes = client.Get(video_path)
    #     # 写入临时文件
    #     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #     temp_file.write(video_bytes)
    #     temp_file.flush()
    #     temp_file.close()
    #     video_path = temp_file.name
    #     is_temp_file = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    frames = []
    count = 0
    success = True
    while success and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        if success:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(image)
            count += frame_interval
    cap.release()

    # 清理临时文件
    if is_temp_file:
        os.remove(video_path)

    return frames

    
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

IMAGE_FACTOR=32
MIN_PIXELS_SEQ = 64
MIN_PIXELS_SEQ = 64  # 64 token
MAX_PIXELS_SEQ = 1980 # 1980 token
IMAGE_MIN_SIDE = IMAGE_FACTOR * 4 * 1

def smart_resize(
    height: int, 
    width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS_SEQ * IMAGE_FACTOR**2, 
    max_pixels: int = MAX_PIXELS_SEQ * IMAGE_FACTOR**2,
    min_side: int = IMAGE_MIN_SIDE
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # possible_resolution = [[644,364], [336,336], [448, 448], [364, 644], [560, 168], [168, 560]]
    # width, height = select_best_resolution((width, height), possible_resolution)
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_side > 0:
        
        if h_bar < min_side:
            beta = min_side / float(h_bar)
            h_bar = ceil_by_factor(min_side, factor)
            w_bar = ceil_by_factor(w_bar * beta, factor)
        
        if w_bar < min_side:
            beta = min_side / float(w_bar)
            w_bar = ceil_by_factor(min_side, factor)
            h_bar = ceil_by_factor(h_bar * beta, factor)
    
    return h_bar, w_bar

async def main():

    client = MultiModalClient(
        url="http://127.0.0.1:18004/generate_stream",
        default_sampling_params={
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "frequency_penalty": 1.05,
            "stop": "<|im_end|>",
        },
    )

    image = Image.open("/mnt/afs/yangdeyu/dependency/sglang/ScreenShot_2026-07-28_212000_993.png")
    width, height = image.size

    # ═══════════════════════════════════════════════════════
    # 测试1: 不传 image_downsample_ratios（回退到 config 默认值）
    # ═══════════════════════════════════════════════════════
    print("=== 测试1: 默认 downsample_ratio ===")
    mm_downsample_ratio = 12
    resized_height, resized_width = smart_resize(
        height, width,
        factor=IMAGE_FACTOR * math.ceil(math.sqrt(mm_downsample_ratio)),
        min_pixels=MIN_PIXELS_SEQ * IMAGE_FACTOR**2,
        max_pixels=MAX_PIXELS_SEQ * IMAGE_FACTOR**2,
        min_side=IMAGE_MIN_SIDE,
    )
    resized_image = image.resize((resized_width, resized_height), resample=Image.BICUBIC)
    images = [resized_image, resized_image]
    prompt = construct_prompt(query="识别这张图的所有文字", image_num=len(images))
    target_sizes = [(resized_width, resized_height)] * 256

    t1 = time.time()
    first_time = 0
    async for token in client.generate(prompt, images=images, target_sizes=target_sizes):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    print(f"\n[默认ratio] TTFT={first_time:.3f}s\n")

    # ═══════════════════════════════════════════════════════
    # 测试2: 传入固定 image_downsample_ratios=[4]
    #   一对图 → 一个 ratio 值
    # ═══════════════════════════════════════════════════════
    print("=== 测试2: downsample_ratio=4 (高压缩, 少token) ===")
    t1 = time.time()
    first_time = 0
    async for token in client.generate(
        prompt, images=images, target_sizes=target_sizes,
        image_downsample_ratios=[4],
    ):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    print(f"\n[ratio=4] TTFT={first_time:.3f}s\n")

    # ═══════════════════════════════════════════════════════
    # 测试3: 传入 image_downsample_ratios=[1] (低压缩, 多token)
    # ═══════════════════════════════════════════════════════
    print("=== 测试3: downsample_ratio=1 (无压缩, 最多token) ===")
    t1 = time.time()
    first_time = 0
    async for token in client.generate(
        prompt, images=images, target_sizes=target_sizes,
        image_downsample_ratios=[1],
    ):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    print(f"\n[ratio=1] TTFT={first_time:.3f}s\n")

    # ═══════════════════════════════════════════════════════
    # 测试4: 多对图 + 不同 ratio
    #   4张图 → 2对 → ratios=[1, 16]
    # ═══════════════════════════════════════════════════════
    print("=== 测试4: 多对图不同ratio [4, 12] ===")
    images_multi = [resized_image] * 4  # 4张图 → 2对
    prompt_multi = construct_prompt(query="描述这些图片", image_num=len(images_multi))
    t1 = time.time()
    first_time = 0
    async for token in client.generate(
        prompt_multi, images=images_multi, target_sizes=target_sizes,
        image_downsample_ratios=[4, 12],
    ):
        if first_time == 0:
            first_time = time.time() - t1
        print(token, end="", flush=True)
    print(f"\n[ratios=[1,16]] TTFT={first_time:.3f}s\n")


asyncio.run(main())