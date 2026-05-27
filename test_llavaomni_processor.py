import asyncio
import cProfile
import io
import pstats
import sys
import time
import types
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock
from io import BytesIO
import base64
import soundfile as sf

import numpy as np
import torch
from PIL import Image
from enum import Enum, auto
# ==========================================
# 0. Pre-import stubs（必须在 import llavaomni 之前完成）
#
# 真实加载：Qwen25VLImageProcessorOptimized（性能测试主体，不 stub）
# Stub：其余 sglang 内部依赖、whisper.audio
# ==========================================

def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# -- whisper.audio stub --------------------------------------------------------
_stub("whisper")
_wa = _stub("whisper.audio")
_wa.pad_or_trim = lambda arr, length=480000: (
    np.asarray(arr, np.float32)[:length]
    if len(arr) >= length
    else np.pad(np.asarray(arr, np.float32), (0, length - len(arr)))
)
_wa.log_mel_spectrogram = lambda wav, n_mels=128: torch.zeros(n_mels, 3000)

# -- sglang.srt.managers.schedule_batch stub -----------------------------------
class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]

class _MultimodalDataItem:
    def __init__(self, **kw): self.__dict__.update(kw)

class _MultimodalProcessorOutput:
    def __init__(self, **kw): self.__dict__.update(kw)

class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()

_sb = _stub("sglang.srt.managers.schedule_batch")
_sb.Modality = Modality
_sb.MultimodalDataItem = _MultimodalDataItem
_sb.MultimodalProcessorOutput = _MultimodalProcessorOutput
_sb.MultimodalInputFormat= MultimodalInputFormat


# sglang.srt.multimodal.image_processor_opt —— 不 stub，走真实包
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "llavaomni", pathlib.Path(__file__).parent / "llavaomni.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BeeBeeLlavaQwen2Processor = _mod.LlavaOmniQwen2Processor


# ==========================================
# 1. Mock 依赖数据结构
# ==========================================
@dataclass
class MockRequestObj:
    audio_data: Optional[List[np.ndarray]] = None

def generate_mock_data(num_image_pairs: int = 8, num_audio_chunks: int = 2):
    """生成测试用的图像、音频和 Prompt。"""
    print(f"🔧 生成 Mock 数据 (图像对: {num_image_pairs}, 音频段: {num_audio_chunks})")
 
    # 两种分辨率交替出现，覆盖分桶逻辑的两个 bucket
    # 前一半 pair 用低分辨率，后一半用高分辨率
    RESOLUTIONS = [(644, 364), (1288, 728)]
    images = []
    
    print("   正在生成并编码 Base64 图像...")
    for pair_idx in range(num_image_pairs):
        w, h = RESOLUTIONS[pair_idx % len(RESOLUTIONS)]
        for _ in range(2):  # 每对 2 张
            # 1. 生成随机 PIL 图像
            img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
            
            # 2. 存入内存 Buffer (使用 JPEG 加快编码速度)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            
            # 3. 转换为 Base64 字符串
            img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # SGLang 兼容这种纯 base64，如果你的系统强制要求 data URI，可以解开下面这行的注释
            img_b64_str = f"data:image/jpeg;base64,{img_b64_str}"
            
            images.append(img_b64_str)
            
    res_summary = ", ".join(f"{w}×{h}" for w, h in RESOLUTIONS)
    print(f"   图像分辨率交替: {res_summary}")
 
    print("   正在生成并编码 Base64 音频...")
    audios = []
    for _ in range(num_audio_chunks):
        # 1. 生成随机波形数据 (16kHz, 5秒)
        # 注意：为了保存为正常的音频，我们将其缩放到 -1.0 到 1.0 之间
        audio_arr = np.random.uniform(-1.0, 1.0, 16000 * 5).astype(np.float32)
        
        # 2. 写入内存 Buffer (封装为标准 WAV 格式)
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio_arr, samplerate=16000, format='WAV', subtype='FLOAT')
        
        # 3. 转换为标准的 Data URI Base64 字符串
        raw_audio_b64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
        audio_b64_str = f"data:audio/wav;base64,{raw_audio_b64}"
        
        audios.append(audio_b64_str)
 
    # Prompt：每对图像用两个连续 <image>，音频用 <audio>
    parts = ["这是测试 Prompt，请看以下图片："]
    for _ in range(num_image_pairs):
        parts.append("<image><image>这部分图片展示了什么？")
    parts.append("结合以下录音：")
    for _ in range(num_audio_chunks):
        parts.append("<audio>")
    parts.append("请给出最终分析。")
 
    return images, "".join(parts), MockRequestObj(audio_data=audios)


async def setup_and_warmup(model_path: str):
    print(f"🚀 加载 Qwen25VLImageProcessorOptimized from: {model_path}")

    # tokenizer stub（不需要真实权重）
    from transformers import AutoTokenizer
   
    tok = AutoTokenizer.from_pretrained("/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/checkpoints/omni_models/1107_llava_omni_qwen25vl_14B_st4_4k")

    _processor_stub = MagicMock()
    _processor_stub.tokenizer = tok

    hf_config = MagicMock()
    enc_cfg = MagicMock()
    vis_cfg = MagicMock()
    aud_cfg = MagicMock()

    # 2. 核心修复：把它们真正“挂载”到彼此身上，而不是赋值为空字典
    hf_config.encoder_config = enc_cfg
    enc_cfg.image_config = vis_cfg
    enc_cfg.audio_config = aud_cfg

    # 3. 填充具体参数
    enc_cfg.model_path = model_path  # ← 真实路径
    vis_cfg.spatial_merge_size = 2
    vis_cfg.image_downsample_size = 16
    aud_cfg.audio_downsample_ratio = 10
    aud_cfg.audio_frame_length = 320

    class MockServerArgs:
        keep_mm_feature_on_device = False
        mm_process_config={}
        skip_tokenizer_init=False

    processor = BeeBeeLlavaQwen2Processor(hf_config, MockServerArgs(), _processor_stub, transport_mode=None)

    img_proc_cls = type(processor._processor.image_processor).__name__
    print(f"   image_processor → {img_proc_cls}")
    if "Optimized" not in img_proc_cls:
        print("   ⚠️  未加载到 Optimized processor，请检查 MODEL_PATH 与 sglang 安装")

    # 预热：排除框架冷启动、CUDA context 初始化等开销
    print("🔥 预热中...")
    imgs, txt, req = generate_mock_data(num_image_pairs=1, num_audio_chunks=1)
    await processor.process_mm_data_async(
        image_data=imgs, input_text=txt, request_obj=req
    )
    print("✅ 预热完成\n" + "=" * 50)
    return processor



async def profile_processor(processor, num_image_pairs: int, num_audio_chunks: int):
    images, input_text, request_obj = generate_mock_data(num_image_pairs, num_audio_chunks)
    print(f"⏱️  开始测试 (prompt 长度: {len(input_text)} 字符)")

    pr = cProfile.Profile()
    pr.enable()
    t_start = time.perf_counter()

    output = await processor.process_mm_data_async(
        image_data=images,
        input_text=input_text,
        request_obj=request_obj,
    )

    t_end = time.perf_counter()
    pr.disable()

    print(f"\n🎯 process_mm_data_async 总耗时: {(t_end - t_start) * 1000:.2f} ms")
    print(f"📦 MultimodalDataItem 数量:      {len(output.mm_items)}")
    print(f"📝 展开后 Input IDs 长度:        {len(output.input_ids)}")

    print("\n🔍 Top 20 耗时函数 (按 cumtime 排序):")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    import os

    MODEL_PATH        = os.environ.get("MODEL_PATH", "/mnt/afs/share/qwen25_vl_encoder")
    TEST_IMAGE_PAIRS  = 64   
    TEST_AUDIO_CHUNKS = 64

    loop = asyncio.get_event_loop()
    processor = loop.run_until_complete(setup_and_warmup(MODEL_PATH))
    loop.run_until_complete(profile_processor(processor, TEST_IMAGE_PAIRS, TEST_AUDIO_CHUNKS))