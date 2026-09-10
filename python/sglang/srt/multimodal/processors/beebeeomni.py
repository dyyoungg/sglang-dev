# coding=utf-8
"""
SGLang multimodal processor for BeeBeeOmniForConditionalGeneration.
Modality pipeline
-----------------
  Image  : HF Qwen2.5-VL image_processor  →  pixel patches + grid_thw
           (1 Grid -> 2 Images) Token count is divided by 2 and paired.

  Audio  : LightLLM Chunked whisper.audio → mel spectrogram [num_chunks, mel_bins, T]
           Token count: sum of ((chunk_len // 320 + downsample - 1) // downsample)

  Video  : Removed (Inference is Image-only)

Position IDs: standard 1-D sequential (no mrope)
"""

import math
import re
import time
from typing import List, Optional, Tuple, Union, Dict
import io
import base64
import cv2
from urllib.parse import unquote, urlparse
import requests
import asyncio
from collections import defaultdict


import numpy as np
import torch
from PIL import Image
from whisper.audio import pad_or_trim, log_mel_spectrogram
import librosa
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import torchaudio.functional as F_audio

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
    MultimodalInputFormat
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
    MultimodalSpecialTokens,
    BaseMultiModalProcessorOutput
)
from sglang.srt.multimodal.image_processor_opt import Qwen25VLImageProcessorOptimized
from sglang.srt.utils.cuda_ipc_transport_utils import (
    CudaIpcTensorTransportProxy,
   
)
from sglang.utils import logger
from sglang.srt.server_args import get_global_server_args
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.models.beebee_audio_encoders import _get_feat_extract_output_lengths

# ── Audio constants (Aligned with LightLLM / Whisper) ─────────────
WHISPER_SAMPLING_RATE = 16000   
WHISPER_N_MEL_BINS    = 128      
WHISPER_HOP_LENGTH    = 320      
WHISPER_MAX_LENGTH    = 480000   # 30 * 16000
MIN_AUDIO_LEN         = 4000     # default min audio len


def compute_image_num_tokens_dynamic(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    downsample_ratios: Union[int, List[int]],
) -> List[int]:
    """
    专门针对图像的计算逻辑：每两张图合并为一个 grid_thw。
    计算出总 Token 数后，平分为两半，连续返回两次，以满足 Prompt 中的两个连续 <image> 标签。

    downsample_ratios: 单个 int（所有图共用）或 List[int]（每个 grid_thw 对应一个 ratio）。
    """
    if isinstance(downsample_ratios, (int, float)):
        downsample_ratios = [int(downsample_ratios)] * len(grid_thw)
    counts = []

    for thw, ratio in zip(grid_thw, downsample_ratios):
        r = 1.0 / math.sqrt(ratio)
        t  = int(thw[0].item())
        h_patches = int(thw[1].item())
        w_patches = int(thw[2].item())

        M = h_patches / spatial_merge_size
        N = w_patches / spatial_merge_size

        Mh  = max(1, int(np.round(M * r)))
        Nw  = max(1, int(np.round(N * r)))

        total_tokens = t * (Mh * Nw)
        half_tokens = total_tokens // 2

        counts.append(half_tokens)
        counts.append(total_tokens - half_tokens)
    return counts


def compute_audio_num_tokens(
    raw_waveform_lengths_list: List[List[int]],
    audio_frame_length: int,
    audio_downsample_ratio: int,
) -> List[int]:
    """
    对于每个切片 chunk：
      feature_len = chunk_len // audio_frame_length
      token_num = (feature_len + downsample - 1) // downsample
    单条音频的总 token 数为所有 chunk_tokens 的总和。
    """
    counts = []
    for chunk_lens in raw_waveform_lengths_list:
        total_tokens = 0
        for length in chunk_lens:
            feature_len = length // audio_frame_length
            token_num = (feature_len + audio_downsample_ratio - 1) // audio_downsample_ratio
            total_tokens += token_num
        counts.append(total_tokens)
    return counts


def compute_audio_num_tokens_qwen3(
    raw_waveform_lengths_list: List[List[int]],
    audio_downsample_ratio: int,
    n_window: int = 50,
) -> List[int]:
    """
    Qwen3 audio encoder token count:
      mel_frames = waveform_len // 160 (Whisper mel hop_length)
      encoder_tokens = _get_feat_extract_output_lengths(mel_frames, n_window)
      projector_tokens = ceil(encoder_tokens / downsample_ratio)
    """
    MEL_HOP_LENGTH = 160
    counts = []
    for chunk_lens in raw_waveform_lengths_list:
        total_tokens = 0
        for length in chunk_lens:
            mel_frames = length // MEL_HOP_LENGTH
            encoder_tokens = _get_feat_extract_output_lengths(
                torch.tensor([mel_frames]), n_window
            ).item()
            proj_tokens = (encoder_tokens + audio_downsample_ratio - 1) // audio_downsample_ratio
            total_tokens += proj_tokens
        counts.append(total_tokens)
    return counts


def fast_load_image_to_numpy(image_file) -> np.ndarray:

    if hasattr(image_file, "url"):
        image_file = image_file.url

    img_bytes = None
    if isinstance(image_file, bytes):
        img_bytes = image_file
    elif isinstance(image_file, str):
        if image_file.startswith("data:"):
            # 剥离 data:image/jpeg;base64,
            img_bytes = base64.b64decode(image_file.split(",")[1])
        elif image_file.startswith(("http://", "https://")):
            img_bytes = requests.get(image_file).content
        elif image_file.startswith("file://"):
            with open(unquote(urlparse(image_file).path), "rb") as f:
                img_bytes = f.read()
        else:
            try:
                img_bytes = base64.b64decode(image_file)
            except Exception:
                with open(image_file, "rb") as f:
                    img_bytes = f.read()
    elif hasattr(image_file, "convert"): 
        if image_file.mode != "RGB":
            image_file = image_file.convert("RGB")
        return np.asarray(image_file)

    if not img_bytes:
        raise ValueError(f"Invalid image input type: {type(image_file)}")

    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv2 is not None:
            return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except ImportError:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)
    

class BeeBeeOmniProcessor(SGLangBaseProcessor):

    from sglang.srt.models.beebee_omni import BeeBeeOmniForConditionalGeneration
    from sglang.srt.models.beebee_omni_moe import BeeBeeMoEOmniForConditionalGeneration

    models = [BeeBeeOmniForConditionalGeneration,
              BeeBeeMoEOmniForConditionalGeneration]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor
        vis_cfg = getattr(hf_config, "vision_config", None) 
        
        self._spatial_merge_size: int = getattr(vis_cfg, "spatial_merge_size", 2)
        self.image_downsample_ratio: int = getattr(vis_cfg, "image_downsample_ratio", 16)

        aud_cfg = getattr(hf_config, "audio_config", None)
        self.audio_downsample_ratio: int = getattr(aud_cfg, "audio_downsample_ratio", 10)
        self.audio_frame_length: int = getattr(aud_cfg, "audio_frame_length", 320)
        self.audio_encoder_type: str = getattr(aud_cfg, "model_type", "beebee_audio_model")
        if self.audio_encoder_type == "beebee_qwen3_audio_model":
            self.qwen3_n_window: int = getattr(aud_cfg, "n_window", 50)

        image_token = "<|image_pad|>"
        audio_token = "<|vision_pad|>"
        self.image_token_str = "<image>"
        self.audio_token_str = "<audio>"
        self.image_token_id =  self._tokenizer.convert_tokens_to_ids(image_token)
        self.audio_token_id =  self._tokenizer.convert_tokens_to_ids(audio_token)
        print("image token:", self.image_token_id, "audio token:", self.audio_token_id)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            image_token_id=self.image_token_id,
            image_token_regex=re.compile(rf"{re.escape(self.image_token_str)}"),
            audio_token=audio_token,
            audio_token_id=self.audio_token_id,
            audio_token_regex=re.compile(rf"{re.escape(self.audio_token_str)}"),
        ).build(_processor)

        try:
            model_path = getattr(vis_cfg, "model_path", None) or server_args.model_path
            optimized_image_processor = Qwen25VLImageProcessorOptimized.from_pretrained(model_path)
            self._processor.image_processor = optimized_image_processor
            logger.info("🚀 Successfully injected Qwen25VLImageProcessorOptimized (numexpr + torch)!")
        except Exception as e:
            logger.warning(f"Failed to load optimized image processor, using default. Error: {e}")

        # Processor-side pixel cache: caches preprocess output (pixel_values)
        # keyed by raw image hash. Eliminates decode + preprocess for repeated images.
        from sglang.srt.environ import envs as _envs
        _cache_mb = _envs.SGLANG_PROCESSOR_CACHE_SIZE_MB.get()
        if _cache_mb > 0:
            from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
            self._pixel_cache = MultiModalStaticCache(_cache_mb * 1024 * 1024)
            logger.info(f"Processor pixel cache enabled: {_cache_mb}MB")
        else:
            self._pixel_cache = None

        self._patch_size = getattr(
            self._processor.image_processor, "patch_size", 14
        )

    def _compute_audio_tokens(self, raw_waveform_lengths: List[List[int]]) -> List[int]:
        """Dispatch to correct token-count formula based on audio encoder type."""
        if self.audio_encoder_type == "beebee_qwen3_audio_model":
            return compute_audio_num_tokens_qwen3(
                raw_waveform_lengths, self.audio_downsample_ratio, self.qwen3_n_window
            )
        return compute_audio_num_tokens(
            raw_waveform_lengths, self.audio_frame_length, self.audio_downsample_ratio
        )

    @classmethod
    def _omni_fast_load_task(cls, args):
        """
        线程池任务路由：
        - 图像：走直接 Array 的极速路线。
        - 音频/视频/预处理字典：走原生的 _load_single_item。
        """
        data, modality, frame_limit, audio_sr, discard_alpha = args
        
        if isinstance(data, dict):
            data_format = data.get("format")
            if data_format in (
                MultimodalInputFormat.PROCESSOR_OUTPUT.name,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
                "processor_output",
                "precomputed_embedding",
            ):
                return data

        if modality == Modality.IMAGE:
            try:
                return fast_load_image_to_numpy(data)
            except Exception as e:
                logger.error(f"Image fast load failed: {e}")
                raise RuntimeError(f"Error while loading image data: {e}")

        return cls._load_single_item(
            data, 
            modality, 
            frame_count_limit=frame_limit, 
            audio_sample_rate=audio_sr, 
            discard_alpha_channel=discard_alpha
        )
    
    def _process_single_audio(self, audio) -> Tuple[torch.Tensor, List[int]]:
        """
        1. Torchaudio 高速重采样
        2. F.pad + view 实现零拷贝切片
        3. Batch 模式调用 log_mel_spectrogram
        """
        if isinstance(audio, dict):
            waveform = audio["array"]
            sr = audio.get("sampling_rate", WHISPER_SAMPLING_RATE)
        else:
            waveform = audio
            sr = WHISPER_SAMPLING_RATE

       
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

   
        if sr != WHISPER_SAMPLING_RATE:
            try:
                waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=WHISPER_SAMPLING_RATE)
            except ImportError:
                logger.warning_once("torchaudio is missing; falling back to slow librosa resample.")
                waveform_np = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=WHISPER_SAMPLING_RATE)
                waveform = torch.from_numpy(waveform_np)

        seq_len = waveform.shape[0]

        if seq_len < MIN_AUDIO_LEN:
            waveform = F.pad(waveform, (0, MIN_AUDIO_LEN - seq_len), mode="constant", value=0.0)
            seq_len = MIN_AUDIO_LEN

        num_chunks = (seq_len + WHISPER_MAX_LENGTH - 1) // WHISPER_MAX_LENGTH
        pad_len = num_chunks * WHISPER_MAX_LENGTH - seq_len
        
        if pad_len > 0:
            waveform = F.pad(waveform, (0, pad_len), mode="constant", value=0.0)
        
        waveform_chunks = waveform.view(num_chunks, WHISPER_MAX_LENGTH)

        chunk_lens = []
        for i in range(num_chunks):
            start_idx = i * WHISPER_MAX_LENGTH
            end_idx = min(start_idx + WHISPER_MAX_LENGTH, seq_len)
            c_len = max(end_idx - start_idx, MIN_AUDIO_LEN)
            chunk_lens.append(c_len)

        mels = log_mel_spectrogram(waveform_chunks, n_mels=WHISPER_N_MEL_BINS)

        return mels, chunk_lens

    def _process_audio_data(self, audio_data: List) -> Tuple[List[torch.Tensor], List[List[int]]]:
        t_audio_start = time.perf_counter()
        mel_list: List[torch.Tensor] = []
        chunk_lengths_list: List[List[int]] = []
        num_audios = len(audio_data)
        if num_audios > 1:
            results = list(self.io_executor.map(self._process_single_audio, audio_data))
            for mels, chunk_lens in results:
                mel_list.append(mels)
                chunk_lengths_list.append(chunk_lens)
        elif num_audios == 1:
            mels, chunk_lens = self._process_single_audio(audio_data[0])
            mel_list.append(mels)
            chunk_lengths_list.append(chunk_lens)
            logger.debug(f"Audio processing cost: {(time.perf_counter() - t_audio_start)*1000:.2f} ms")

        return mel_list, chunk_lengths_list

    def _encode_and_expand_text(
        self,
        input_text: Union[str, List[int]],
        image_num_tokens: List[int], 
        audio_num_tokens: List[int], 
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Modality]]:
       
        expanded:      List[int]              = []
        offsets:       List[Tuple[int, int]]  = []
        modality_list: List[Modality]         = []

        pattern = f"({re.escape(self.image_token_str)}|{re.escape(self.audio_token_str)})"
        
        chunks = re.split(pattern, input_text)
        tokenizer = self._processor.tokenizer
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        
        img_idx = 0
        aud_idx = 0
        idx = 0
        while idx < len(chunks):
           
            text = chunks[idx]
            if text:
                ids = tokenizer.encode(text)
                if len(expanded) == 0:
                    expanded.extend(ids)
                else:
                    if len(ids) > 0 and ids[0] == bos_token_id:
                        ids = ids[1:]
                    expanded.extend(ids)
            
            idx += 1
            
            if idx < len(chunks):
                token = chunks[idx]
                if token == self.image_token_str:
                    n = image_num_tokens[img_idx]
                    img_idx += 1
                    start = len(expanded)
                   
                    expanded.extend([self.image_token_id] * n)
                    end = len(expanded) - 1
                    offsets.append((start, end))
                    modality_list.append(Modality.IMAGE)
                    
                elif token == self.audio_token_str:
                    n = audio_num_tokens[aud_idx]
                    aud_idx += 1
                    start = len(expanded)
                    expanded.extend([self.audio_token_id] * n)
                    end = len(expanded) - 1
                    offsets.append((start, end))
                    modality_list.append(Modality.AUDIO)
                
                idx += 1

        return expanded, offsets, modality_list
    
    def _process_and_collect_mm_items(
        self,
        images: List,
        **kwargs,
    ) -> Tuple[List, None, dict]:
        """
        逻辑：
        1. 每两张图作为一个处理单元 (Pair)。
        2. 将 Pair 按 (Width, Height) 分桶。
        3. 对每个桶调用优化后的 image_processor 批量处理 (利用 numexpr 和 torch 算子)。
        4. 拆解批量结果，并按照原始 Pair 顺序重排。
        5. 返回 pixel_values 列表，其中每个元素对应一个 grid_thw。
        """
        hf_ret = {}
        if not images:
            return [], None, hf_ret

        t_vision_start = time.perf_counter()
        total_imgs = len(images)
        pairs_metadata = []
      
        for i in range(0, total_imgs, 2):
            curr_pair = [images[i], images[i+1]] if i+1 < total_imgs else [images[i], images[i]]
            pair_idx = i // 2
            h, w = curr_pair[0].shape[:2]

            pairs_metadata.append({
                "idx": pair_idx,
                "images": curr_pair,
                "size": (w, h),
            })

        buckets = defaultdict(list)
        for p in pairs_metadata:
            buckets[p["size"]].append(p)

        ordered_pixel_values = [None] * len(pairs_metadata)
        ordered_grids = [None] * len(pairs_metadata)

        for size, bucket_pairs in buckets.items():
            # 合并当前尺寸下所有的图片 (N 对 * 2 = 2N 张图)
            batch_images = []
            for p in bucket_pairs:
                batch_images.extend(p["images"])
            t_pre = time.perf_counter()
            preprocess_kwargs = dict(return_tensors="pt", **kwargs)
            if isinstance(self._processor.image_processor, Qwen25VLImageProcessorOptimized):
                preprocess_kwargs["patch_reshape_method"] = "torch"
            image_outputs = self._processor.image_processor.preprocess(
                batch_images,
                **preprocess_kwargs,
            )
            t_post = time.perf_counter()
            # logger.info(
            #     f"Image preprocess: size={size}, "
            #     f"num_imgs={len(batch_images)}, "
            #     f"time={(t_post - t_pre)*1000:.2f}ms"
            # )
            bucket_features = image_outputs.get("pixel_values") # Tensor: [P_total, D]
            bucket_thw = image_outputs.get("grid_thw")         # Tensor: [T_total, 3]

          
            grid_h, grid_w = int(bucket_thw[0, 1]), int(bucket_thw[0, 2])
            stride = grid_h * grid_w

            for i, p in enumerate(bucket_pairs):
                start_patch = i * stride
                end_patch = (i + 1) * stride
            
                pair_feature = bucket_features[start_patch:end_patch]
                pair_grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
                
                # 放入原始顺序对应的槽位
                ordered_pixel_values[p["idx"]] = pair_feature
                ordered_grids[p["idx"]] = pair_grid

        hf_ret["pixel_values"] = ordered_pixel_values 
        hf_ret["image_grid_thw"] = torch.cat(ordered_grids, dim=0)

        logger.debug(f"Optimized Image Preprocess cost: {(time.perf_counter() - t_vision_start)*1000:.2f} ms, num_imgs={len(batch_images)}")

        return [], None, hf_ret
    

    def _apply_cuda_ipc_protection(self, mm_items: List["MultimodalDataItem"]):
        """为所有准备跨进程发送的特征套上 Proxy 壳，防止显存泄漏"""
        try:
            from sglang.srt.environ import envs
            SGL_USE_CUDA_IPC = envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get()
            _IPC_POOL_HANDLE_CACHE = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()
        except ImportError:
            SGL_USE_CUDA_IPC = hasattr(self, "cudaipc_mmfeature_pool") and self.cudaipc_mmfeature_pool is not None
            _IPC_POOL_HANDLE_CACHE = True

        if not SGL_USE_CUDA_IPC or not hasattr(self, "cudaipc_mmfeature_pool"):
            return mm_items

        for item in mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                sync_flag, available_slice, byte_offset = (
                    self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(item.feature)
                )
                if isinstance(available_slice, torch.Tensor):
                    available_slice.copy_(item.feature.view(torch.int8).view(-1), non_blocking=True)
                    item.feature = CudaIpcTensorTransportProxy(
                        data=available_slice, 
                        info_data=item.feature, 
                        sync_buffer_meta=sync_flag,
                        pool_ipc_handle=(self.cudaipc_mmfeature_pool._pool_ipc_handle if _IPC_POOL_HANDLE_CACHE else None),
                        pool_byte_offset=byte_offset, 
                        pool_device_index=self.cudaipc_mmfeature_pool._pool_device_index,
                    )
                elif not getattr(self.server_args, "keep_mm_feature_on_device", False):
                    item.feature = item.feature.cpu()

            elif (
                    isinstance(item.precomputed_embeddings, torch.Tensor)
                    and item.precomputed_embeddings.is_cuda
                ):

                sync_flag, available_slice, byte_offset = (
                    self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                        item.precomputed_embeddings
                    )
                )
                if isinstance(available_slice, torch.Tensor):
                    available_slice.copy_(
                        item.precomputed_embeddings.view(torch.int8).view(-1),
                        non_blocking=True,
                    )
                    item.precomputed_embeddings = CudaIpcTensorTransportProxy(
                        data=available_slice,
                        info_data=item.precomputed_embeddings,
                        sync_buffer_meta=sync_flag,
                        pool_ipc_handle=(
                            self.cudaipc_mmfeature_pool._pool_ipc_handle
                            if _IPC_POOL_HANDLE_CACHE
                            else None
                        ),
                        pool_byte_offset=byte_offset,
                        pool_device_index=self.cudaipc_mmfeature_pool._pool_device_index,
                    )
                elif not getattr(self.server_args, "keep_mm_feature_on_device", False):
                    item.precomputed_embeddings = item.precomputed_embeddings.cpu()
        return mm_items

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    )-> BaseMultiModalProcessorOutput:
        
        SGLangBaseProcessor.validate_mm_data(image_data, video_data, audio_data)
        if isinstance(prompt, list) and return_text:
            prompt_str = self._tokenizer.decode(prompt)
        else:
            prompt_str = prompt
        assert isinstance(prompt, str)
      
        tasks = []
        if image_data:
            tasks.extend([(d, Modality.IMAGE, None, audio_sample_rate, discard_alpha_channel) for d in image_data])
    
        if audio_data:
            tasks.extend([(d, Modality.AUDIO, None, audio_sample_rate, discard_alpha_channel) for d in audio_data])

        
        
        results = list(self.io_executor.map(self.__class__._omni_fast_load_task, tasks)) if tasks else []

        images, videos, audios = [], [], []
        idx = 0
        if image_data:
            images = results[idx : idx + len(image_data)]
            idx += len(image_data)
        if audio_data:
            audios = results[idx : idx + len(audio_data)]

        return BaseMultiModalProcessorOutput(
            images=images, 
            audios=audios,
            videos=videos,
            input_text=prompt_str,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """EPD path: build MultimodalProcessorOutput from precomputed embeddings.

        Called by encode_receiver after the encoder server has produced
        embeddings. We need to:
        1. Compute per-image / per-audio token counts from grid metadata.
        2. Expand the prompt text with _encode_and_expand_text.
        3. Slice the flat embedding tensors and wrap them as mm_items.
        """
        img_grid_thw = kwargs.get("img_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        per_pair_ratios = kwargs.get("image_downsample_ratios", None)

        # ── Image token counts (paired: each grid_thw → 2 <image> tokens) ──
        image_num_tokens = []
        if img_grid_thw is not None and len(img_grid_thw) > 0:
            num_pairs = len(img_grid_thw)
            if per_pair_ratios is None:
                per_pair_ratios = [self.image_downsample_ratio] * num_pairs
            else:
                per_pair_ratios = [int(r) for r in per_pair_ratios]
                if len(per_pair_ratios) < num_pairs:
                    per_pair_ratios.extend(
                        [self.image_downsample_ratio] * (num_pairs - len(per_pair_ratios))
                    )
            image_num_tokens = compute_image_num_tokens_dynamic(
                img_grid_thw, self._spatial_merge_size, per_pair_ratios
            )

        # ── Audio token counts (one per audio) ──
        audio_num_tokens = []
        if audio_feature_lens is not None and len(audio_feature_lens) > 0:
            audio_num_tokens = [int(x.item()) for x in audio_feature_lens]

        # ── Expand prompt → input_ids with placeholder tokens ──
        input_ids, offsets, modality_list = self._encode_and_expand_text(
            prompt, image_num_tokens, audio_num_tokens
        )

        # ── Build mm_items from precomputed embeddings ──
        mm_items = []
        image_offsets = [
            off for m, off in zip(modality_list, offsets) if m == Modality.IMAGE
        ]
        audio_offsets = [
            off for m, off in zip(modality_list, offsets) if m == Modality.AUDIO
        ]

        # Image: every 2 consecutive <image> offsets belong to one grid_thw (one pair)
        if img_grid_thw is not None and len(img_grid_thw) > 0:
            img_embedding = embeddings.get(Modality.IMAGE)
            img_consumed = 0
            for i in range(len(img_grid_thw)):
                off1 = image_offsets[i * 2]
                off2 = image_offsets[i * 2 + 1]
                num_tokens = off2[1] - off1[0] + 1
                embedding_slice = img_embedding[img_consumed : img_consumed + num_tokens]
                img_consumed += num_tokens
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        offsets=[(off1[0], off2[1])],
                        precomputed_embeddings=embedding_slice,
                    )
                )

        # Audio: one embedding slice per audio
        if audio_feature_lens is not None and len(audio_feature_lens) > 0:
            aud_embedding = embeddings.get(Modality.AUDIO)
            aud_consumed = 0
            for i, off in enumerate(audio_offsets):
                num_tokens = off[1] - off[0] + 1
                embedding_slice = aud_embedding[aud_consumed : aud_consumed + num_tokens]
                aud_consumed += num_tokens
                mm_items.append(
                    MultimodalDataItem(
                        modality=Modality.AUDIO,
                        offsets=[off],
                        precomputed_embeddings=embedding_slice,
                    )
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=None,
            im_end_id=None,
            im_token_id=self.image_token_id,
            video_token_id=None,
            audio_token_id=self.audio_token_id,
            mrope_positions=None,
            mrope_position_delta=None,
        )

    # ── Processor pixel cache helpers ────────────────────────────────────

    @staticmethod
    def _hash_raw_data(item) -> int:
        """Hash raw input data (base64 string / bytes / URL) directly without decoding.

        For base64/bytes: hash as-is (cheap, no decode needed).
        For URLs: hash the URL string itself (same URL → same content assumption).
        """
        import hashlib
        hasher = hashlib.sha256()
        if isinstance(item, bytes):
            hasher.update(item)
        elif isinstance(item, str):
            hasher.update(item.encode("utf-8"))
        else:
            hasher.update(repr(item).encode("utf-8"))
        return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)

    def _compute_pair_hashes(self, image_data: List) -> List[int]:
        """Compute pair-wise hashes directly from raw image_data (no decode).

        Assumes len(image_data) is even (padding done at entry).
        """
        import hashlib

        pair_hashes = []
        for i in range(0, len(image_data), 2):
            hasher = hashlib.sha256()
            item0, item1 = image_data[i], image_data[i + 1]
            if isinstance(item0, bytes):
                hasher.update(item0)
            else:
                hasher.update(str(item0).encode("utf-8"))
            if isinstance(item1, bytes):
                hasher.update(item1)
            else:
                hasher.update(str(item1).encode("utf-8"))
            pair_hashes.append(
                int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)
            )
        return pair_hashes

    def _compute_audio_hashes(self, audio_data: List) -> List[int]:
        """Compute per-audio hashes directly from raw audio_data (no decode)."""
        return [self._hash_raw_data(item) for item in audio_data]

    # ── Cached version of process_mm_data_async ──────────────────────────

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ) -> Dict:
        """Multimodal processor with pixel cache.

        Pipeline:
          1. Hash raw data + cache lookup (no decode, instant)
          2. For misses: parallel load (fast_load_image_to_numpy / audio load)
          3. Parallel compute (image preprocess || audio process)
        """
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        # ═══ Front-load Batch Padding ═══
        if image_data and len(image_data) % 2 != 0:
            image_data = list(image_data) + [image_data[-1]]

        # ═══ Per-pair downsample ratios ═══
        per_pair_ratios = getattr(request_obj, 'image_downsample_ratios', None)
        num_pairs = len(image_data) // 2 if image_data else 0
        if per_pair_ratios is None:
            per_pair_ratios = [self.image_downsample_ratio] * num_pairs
        else:
            per_pair_ratios = [int(r) for r in per_pair_ratios]
            if len(per_pair_ratios) < num_pairs:
                per_pair_ratios.extend(
                    [self.image_downsample_ratio] * (num_pairs - len(per_pair_ratios))
                )

        # ══════════════════════════════════════════════════════════════════
        # STAGE 1: Hash + cache lookup (no decode, pure CPU, < 2ms)
        # ══════════════════════════════════════════════════════════════════
        pixel_values_list = None
        image_grid_thw = None
        pair_hashes = None
        image_num_tokens = []
        img_miss_pair_indices = []
        img_hit_pixels = {}
        img_hit_grids = {}

        if image_data:
            t_hash_start = time.perf_counter()
            pair_hashes = self._compute_pair_hashes(image_data)
            for i, h in enumerate(pair_hashes):
                cached = self._pixel_cache.get_single(h)
                if cached is not None:
                    img_hit_pixels[i] = cached.embedding
                    img_hit_grids[i] = cached.grid_thw
                else:
                    img_miss_pair_indices.append(i)
            t_hash_end = time.perf_counter()

            num_pairs = len(pair_hashes)
            logger.info(
                f"[PIXEL CACHE] {num_pairs} image pairs: "
                f"{num_pairs - len(img_miss_pair_indices)} hits, "
                f"{len(img_miss_pair_indices)} misses, "
                f"hash+lookup={( t_hash_end - t_hash_start)*1000:.2f}ms"
            )

        audio_num_tokens = []
        mel_list = []
        raw_waveform_lengths = []
        audio_hashes = None
        aud_miss_indices = []
        aud_hit = {}

        if audio_data:
            audio_hashes = self._compute_audio_hashes(audio_data)
            for i, h in enumerate(audio_hashes):
                cached = self._pixel_cache.get_single(h)
                if cached is not None:
                    aud_hit[i] = (cached.embedding, getattr(cached, "chunk_lens", None))
                else:
                    aud_miss_indices.append(i)

            logger.info(
                f"[AUDIO CACHE] {len(audio_hashes)} audios: "
                f"{len(audio_hashes) - len(aud_miss_indices)} hits, "
                f"{len(aud_miss_indices)} misses"
            )

        # ══════════════════════════════════════════════════════════════════
        # STAGE 2: Parallel load misses (reuse load_mm_data's io_executor.map)
        #   load_mm_data internally uses _omni_fast_load_task with thread pool
        #   for both images (fast_load_image_to_numpy) and audios (_load_single_item)
        # ══════════════════════════════════════════════════════════════════
        t_load_start = time.perf_counter()
        miss_images = None
        miss_audios = None

        # Build miss-only data lists
        miss_img_data = []
        if img_miss_pair_indices:
            for idx in img_miss_pair_indices:
                i = idx * 2
                miss_img_data.append(image_data[i])
                miss_img_data.append(image_data[i + 1])

        miss_aud_data = [audio_data[i] for i in aud_miss_indices] if aud_miss_indices else []

        # Single load_mm_data call handles both modalities with internal parallelism
        if miss_img_data or miss_aud_data:
            def _load_misses():
                return self.load_mm_data(
                    prompt=input_text,
                    image_data=miss_img_data or None,
                    audio_data=miss_aud_data or None,
                    multimodal_tokens=self.mm_tokens,
                )

            base_output = await loop.run_in_executor(self.io_executor, _load_misses)
            miss_images = base_output.images if base_output.images else None
            miss_audios = base_output.audios if base_output.audios else None

        t_load_end = time.perf_counter()

        # ══════════════════════════════════════════════════════════════════
        # STAGE 3: Parallel compute (image preprocess || audio process)
        # ══════════════════════════════════════════════════════════════════
        def _process_images():
            if miss_images:
                _, _, ret = self._process_and_collect_mm_items(images=miss_images)
                return ret
            return None

        def _process_audios():
            if miss_audios:
                return self._process_audio_data(miss_audios)
            return None

        image_task = loop.run_in_executor(self.io_executor, _process_images)
        audio_task = loop.run_in_executor(self.io_executor, _process_audios)
        img_result, aud_result = await asyncio.gather(image_task, audio_task)

        t_compute_end = time.perf_counter()

        # ── Store image misses in cache ──
        if img_result is not None:
            miss_pvs = img_result["pixel_values"]
            miss_grid_thw = img_result["image_grid_thw"]
            for i, idx in enumerate(img_miss_pair_indices):
                pv = miss_pvs[i]
                grid = miss_grid_thw[i]
                entry = EmbeddingResult(embedding=pv)
                entry.grid_thw = grid
                self._pixel_cache.set(pair_hashes[idx], entry)
                img_hit_pixels[idx] = pv
                img_hit_grids[idx] = grid

        # ── Store audio misses in cache ──
        if aud_result is not None:
            miss_mels, miss_chunk_lens = aud_result
            for i, idx in enumerate(aud_miss_indices):
                mel = miss_mels[i]
                chunk_lens = miss_chunk_lens[i]
                entry = EmbeddingResult(embedding=mel)
                entry.chunk_lens = chunk_lens
                self._pixel_cache.set(audio_hashes[idx], entry)
                aud_hit[idx] = (mel, chunk_lens)

        if img_miss_pair_indices or aud_miss_indices:
            logger.info(
                f"[CACHE] miss: img={len(img_miss_pair_indices)} pairs, "
                f"aud={len(aud_miss_indices)} items, "
                f"load={( t_load_end - t_load_start)*1000:.2f}ms, "
                f"compute={( t_compute_end - t_load_end)*1000:.2f}ms, "
                f"total time={( t_compute_end - t_load_start)*1000:.2f}ms",
            )

        # ══════════════════════════════════════════════════════════════════
        # Assemble results
        # ══════════════════════════════════════════════════════════════════
        if image_data:
            num_pairs = len(pair_hashes)
            pixel_values_list = [img_hit_pixels[i] for i in range(num_pairs)]
            image_grid_thw = torch.stack([img_hit_grids[i] for i in range(num_pairs)])
            image_num_tokens = compute_image_num_tokens_dynamic(
                image_grid_thw, self._spatial_merge_size, per_pair_ratios
            )

        if audio_data:
            for i in range(len(audio_hashes)):
                mel, chunk_lens = aud_hit[i]
                mel_list.append(mel)
                raw_waveform_lengths.append(chunk_lens)
            audio_num_tokens = self._compute_audio_tokens(raw_waveform_lengths)

        # === Build token sequence and mm_items ===
        logger.info(
            f"[TOKEN COUNT] image_num_tokens={image_num_tokens} (sum={sum(image_num_tokens)}), "
            f"audio_num_tokens={audio_num_tokens} (sum={sum(audio_num_tokens)}), "
        )
        expanded_ids, offsets, modality_list = self._encode_and_expand_text(
            input_text, image_num_tokens, audio_num_tokens
        )

        mm_items: List[MultimodalDataItem] = []
        image_offsets = [
            off for m, off in zip(modality_list, offsets) if m == Modality.IMAGE
        ]
        audio_offsets = [
            off for m, off in zip(modality_list, offsets) if m == Modality.AUDIO
        ]

        if image_grid_thw is not None and len(image_grid_thw) > 0:
            for i in range(len(image_grid_thw)):
                thw = image_grid_thw[i].unsqueeze(0)
                off1 = image_offsets[i * 2]
                off2 = image_offsets[i * 2 + 1]
                new_offset = (off1[0], off2[1])
                feat = pixel_values_list[i] if pixel_values_list is not None else None

                mm_items.append(MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[new_offset],
                    feature=feat,
                    hash=pair_hashes[i],
                    model_specific_data={
                        "image_grid_thw": thw,
                        "downsample_ratio": per_pair_ratios[i],
                    },
                ))

        for i, mel in enumerate(mel_list):
            lengths = raw_waveform_lengths[i]
            off = audio_offsets[i]
            mm_items.append(MultimodalDataItem(
                modality=Modality.AUDIO,
                offsets=[off],
                feature=mel,
                hash=audio_hashes[i] if audio_hashes else None,
                model_specific_data={"audio_length": lengths},
            ))

        mm_items = self._apply_cuda_ipc_protection(mm_items)

        t_total = time.perf_counter() - t0
        logger.info(
            f"[BeeBeeLlavaProcessor] process_mm_data_async: "
            f"{1e3*t_total:.1f}ms total"
        )

        return MultimodalProcessorOutput(
            input_ids=expanded_ids,
            mm_items=mm_items,
            im_start_id=None,
            im_end_id=None,
            im_token_id=self.image_token_id,
            video_token_id=None,
            audio_token_id=self.audio_token_id,
            mrope_positions=None,
            mrope_position_delta=None,
        )

    async def process_mm_data_async_origin(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ) -> Dict:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        # ═══ Per-pair downsample ratios ═══
        num_pairs = len(image_data) // 2 if image_data else 0
        per_pair_ratios = getattr(request_obj, 'image_downsample_ratios', None)
        if per_pair_ratios is None:
            per_pair_ratios = [self.image_downsample_ratio] * num_pairs
        else:
            per_pair_ratios = [int(r) for r in per_pair_ratios]
            if len(per_pair_ratios) < num_pairs:
                per_pair_ratios.extend(
                    [self.image_downsample_ratio] * (num_pairs - len(per_pair_ratios))
                )

        def _sync_load_data():
            return self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                audio_data=audio_data,
                multimodal_tokens=self.mm_tokens,
            )
        base_output = await loop.run_in_executor(self.io_executor, _sync_load_data)

        raw_images = [item for m, item in base_output.organize_results() if m == Modality.IMAGE]
       
        def _sync_process_images():
            if raw_images:
                _, _, ret = self._process_and_collect_mm_items(images=raw_images)
                return ret
            return {}

        def _sync_process_audios():
            if base_output.audios:
                return self._process_audio_data(base_output.audios)
            return [], []

        image_task = loop.run_in_executor(self.io_executor, _sync_process_images)
        audio_task = loop.run_in_executor(self.io_executor, _sync_process_audios)


        hf_ret, (mel_list, raw_waveform_lengths) = await asyncio.gather(image_task, audio_task)
        
        image_grid_thw: Optional[torch.Tensor] = hf_ret.get("image_grid_thw")
        pixel_values: Optional[torch.Tensor] = hf_ret.get("pixel_values")

        image_num_tokens = []
        if image_grid_thw is not None and len(image_grid_thw) > 0:
            image_num_tokens = compute_image_num_tokens_dynamic(
                image_grid_thw, self._spatial_merge_size, per_pair_ratios
            )
       
        audio_num_tokens = []
        if base_output.audios:
            # mel_list 为 [num_chunks, 128, 3000] 的列表
            audio_num_tokens = self._compute_audio_tokens(raw_waveform_lengths)

        # 正则切分、编码并展开占位符
        expanded_ids, offsets, modality_list = self._encode_and_expand_text(
            input_text, image_num_tokens, audio_num_tokens
        )

       
        mm_items: List[MultimodalDataItem] = []
        image_offsets = [off for m, off in zip(modality_list, offsets) if m == Modality.IMAGE]
        audio_offsets = [off for m, off in zip(modality_list, offsets) if m == Modality.AUDIO]

        # ── 图像：1个底层特征 -> 绑定给 2个文本区间的 Offsets ──
        if image_grid_thw is not None and len(image_grid_thw) > 0:
            for i in range(len(image_grid_thw)):
                thw = image_grid_thw[i].unsqueeze(0)
                off1 = image_offsets[i * 2]
                off2 = image_offsets[i * 2 + 1]
                new_offset = (off1[0], off2[1])
                feat = pixel_values[i] if pixel_values is not None else None
                
                mm_items.append(MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[new_offset],
                    feature=feat,
                    hash=None,
                    model_specific_data={
                        "image_grid_thw": thw,
                        "downsample_ratio": per_pair_ratios[i],
                    },
                ))

        for i, mel in enumerate(mel_list):
            lengths = raw_waveform_lengths[i]
            off = audio_offsets[i]
            
            mm_items.append(MultimodalDataItem(
                modality=Modality.AUDIO, 
                offsets=[off],
                feature=mel, 
                hash=None,
                model_specific_data={"audio_length": lengths},
            ))

        # 全局加上 IPC Proxy 保护，避免传给 GPU 进程时 OOM
        mm_items = self._apply_cuda_ipc_protection(mm_items)

        t_total = time.perf_counter() - t0
        logger.debug(f"[BeeBeeLlavaProcessor Perf] Process completed in {1e3*t_total:.1f}ms")

        return MultimodalProcessorOutput(
            input_ids=expanded_ids,
            mm_items=mm_items,
            im_start_id=None,
            im_end_id=None,
            im_token_id=self.image_token_id,
            video_token_id=None,
            audio_token_id=self.audio_token_id,
            mrope_positions=None,
            mrope_position_delta=None,
        )