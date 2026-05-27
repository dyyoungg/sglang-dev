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

# ── Audio constants (Aligned with LightLLM / Whisper) ─────────────
WHISPER_SAMPLING_RATE = 16000   
WHISPER_N_MEL_BINS    = 128      
WHISPER_HOP_LENGTH    = 320      
WHISPER_MAX_LENGTH    = 480000   # 30 * 16000
MIN_AUDIO_LEN         = 4000     # default min audio len


def compute_image_num_tokens_dynamic(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    downsample_ratio: int,
) -> List[int]:
    """
    专门针对图像的计算逻辑：每两张图合并为一个 grid_thw。
    计算出总 Token 数后，平分为两半，连续返回两次，以满足 Prompt 中的两个连续 <image> 标签。
    """
    r = 1.0 / math.sqrt(downsample_ratio)
    counts = []
    
    for thw in grid_thw:
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

    models = [BeeBeeOmniForConditionalGeneration]
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
            model_path = getattr(vis_cfg, "model_path", "/mnt/afs/share/qwen25_vl_encoder")
            optimized_image_processor = Qwen25VLImageProcessorOptimized.from_pretrained(model_path)
            self._processor.image_processor = optimized_image_processor
            logger.info("🚀 Successfully injected Qwen25VLImageProcessorOptimized (numexpr + torch)!")
        except Exception as e:
            logger.warning(f"Failed to load optimized image processor, using default. Error: {e}")


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
            
            image_outputs = self._processor.image_processor.preprocess(
                batch_images,
                return_tensors="pt",
                patch_reshape_method="torch",
                **kwargs
            )
            
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

        logger.debug(f"Optimized Image Preprocess cost: {(time.perf_counter() - t_vision_start)*1000:.2f} ms")

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

    async def process_mm_data_async(
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
                image_grid_thw, self._spatial_merge_size, self.image_downsample_ratio
            )
       
        audio_num_tokens = []
        if base_output.audios:
            # mel_list 为 [num_chunks, 128, 3000] 的列表
            audio_num_tokens = compute_audio_num_tokens(
                raw_waveform_lengths, self.audio_frame_length, self.audio_downsample_ratio
            )

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
                    model_specific_data={"image_grid_thw": thw},
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