import asyncio
import concurrent.futures
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    load_audio,
    load_image,
    load_video,
    random_uuid,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 10

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="


def _make_dummy_png_data_uri(width: int, height: int) -> str:
    """Generate a solid-color PNG data-URI at the specified resolution."""
    import base64 as _b64
    import io as _io

    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    b64 = _b64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
rid_to_err_msg: Dict[str, str] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)

# Whether to isolate image preprocessing in a separate process to avoid
# numexpr thread starvation when mooncake/torch threads are present.
# Default OFF: the primary fix is limiting numexpr threads via
# SGLANG_NUMEXPR_NUM_THREADS (set below). Enable isolation only if needed.
use_image_processor_isolation = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_ISOLATED", "0")) == 1
)

# Limit numexpr threads to avoid barrier stalls in the encoder process.
# With ~1000 threads (mooncake/torch/NCCL), 16-thread numexpr barrier can
# stall for 10-30s. 4 threads gives ~15ms normalize (vs 5ms at 16 threads)
# but is safe from scheduling contention.
if "SGLANG_NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["SGLANG_NUMEXPR_NUM_THREADS"] = "16"


class ImageProcessorWorker:
    """Persistent subprocess for image preprocessing using shared memory.

    Isolates numexpr-based image preprocessing from the main encoder process
    which has ~1000 threads (mooncake, torch, NCCL). Without isolation,
    numexpr's multi-thread barrier can stall for 10-30+ seconds due to
    thread scheduling contention.

    Uses shared memory for zero-copy data passing (no pickle overhead).
    Only small metadata (shape, dtype, offsets) goes through mp.Queue.
    """

    # Pre-allocate shared memory for input images (max ~128 images * 644*364*3 = ~90MB)
    _INPUT_SHM_SIZE = 256 * 1024 * 1024   # 256 MB
    # Pre-allocate shared memory for output (max ~90MB float32 patches)
    _OUTPUT_SHM_SIZE = 256 * 1024 * 1024   # 256 MB

    def __init__(self, model_path: str):
        from multiprocessing import shared_memory

        # Create persistent shared memory buffers
        self._input_shm = shared_memory.SharedMemory(
            create=True, size=self._INPUT_SHM_SIZE
        )
        self._output_shm = shared_memory.SharedMemory(
            create=True, size=self._OUTPUT_SHM_SIZE
        )

        ctx = mp.get_context("spawn")
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()
        self._process = ctx.Process(
            target=ImageProcessorWorker._worker_loop,
            args=(
                self._input_queue,
                self._output_queue,
                model_path,
                self._input_shm.name,
                self._output_shm.name,
                self._INPUT_SHM_SIZE,
                self._OUTPUT_SHM_SIZE,
            ),
            daemon=True,
        )
        self._process.start()
        logger.info(
            f"ImageProcessorWorker started (pid={self._process.pid}), "
            f"input_shm={self._input_shm.name}, output_shm={self._output_shm.name}"
        )

    @staticmethod
    def _worker_loop(
        input_queue, output_queue, model_path,
        input_shm_name, output_shm_name,
        input_shm_size, output_shm_size,
    ):
        """Worker process entry point: load processor, loop forever."""
        import signal
        from multiprocessing import shared_memory

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        from sglang.srt.multimodal.image_processor_opt import (
            Qwen25VLImageProcessorOptimized,
        )

        processor = Qwen25VLImageProcessorOptimized.from_pretrained(model_path)

        # Attach to shared memory
        in_shm = shared_memory.SharedMemory(name=input_shm_name, create=False)
        out_shm = shared_memory.SharedMemory(name=output_shm_name, create=False)
        in_buf = np.ndarray(input_shm_size, dtype=np.uint8, buffer=in_shm.buf)
        out_buf = np.ndarray(output_shm_size, dtype=np.uint8, buffer=out_shm.buf)

        logger.info("ImageProcessorWorker: processor loaded, ready.")

        while True:
            try:
                meta = input_queue.get()
                # meta = {num_images, shape: (H, W, C), dtype, total_bytes, kwargs}
                num_images = meta["num_images"]
                h, w, c = meta["shape"]
                dtype = np.dtype(meta["dtype"])
                total_bytes = meta["total_bytes"]
                kwargs = meta["kwargs"]

                # Reconstruct numpy array from shared memory (zero-copy view)
                images_flat = np.ndarray(
                    (num_images, h, w, c), dtype=dtype,
                    buffer=in_shm.buf[:total_bytes]
                )
                # Convert to list of numpy arrays as expected by preprocess
                images = [images_flat[i] for i in range(num_images)]

                result = processor.preprocess(images, **kwargs)

                # Write output to shared memory
                pixel_values = result["pixel_values"]
                grid_thw = result["grid_thw"]

                # pixel_values can be numpy or torch tensor
                if isinstance(pixel_values, torch.Tensor):
                    pv_np = pixel_values.numpy()
                else:
                    pv_np = np.ascontiguousarray(pixel_values)

                pv_bytes = pv_np.nbytes
                out_buf[:pv_bytes] = pv_np.view(np.uint8).ravel()

                # Send only metadata back (tiny)
                output_queue.put({
                    "pv_shape": list(pv_np.shape),
                    "pv_dtype": str(pv_np.dtype),
                    "pv_bytes": pv_bytes,
                    "grid_thw": grid_thw,  # small tensor, pickle is fine
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                output_queue.put({"error": str(e)})

    def preprocess(self, images, **kwargs) -> dict:
        """Send images to worker via shared memory and wait for result."""
        import time as _time
        t0 = _time.perf_counter()

        # All images must have the same shape (guaranteed by bucket grouping)
        img0 = images[0]
        h, w = img0.shape[:2]
        c = img0.shape[2] if img0.ndim == 3 else 1
        dtype = img0.dtype
        num_images = len(images)

        # Stack images and copy to shared memory
        stacked = np.stack(images)  # (N, H, W, C)
        total_bytes = stacked.nbytes
        assert total_bytes <= self._INPUT_SHM_SIZE, (
            f"Input too large: {total_bytes} > {self._INPUT_SHM_SIZE}"
        )

        t_stack = _time.perf_counter()

        # Write to shared memory (fast memcpy)
        shm_view = np.ndarray(
            stacked.shape, dtype=dtype, buffer=self._input_shm.buf[:total_bytes]
        )
        np.copyto(shm_view, stacked)

        t_copy_in = _time.perf_counter()

        # Send metadata only (tiny, <1KB)
        self._input_queue.put({
            "num_images": num_images,
            "shape": (h, w, c),
            "dtype": str(dtype),
            "total_bytes": total_bytes,
            "kwargs": kwargs,
        })

        t_put = _time.perf_counter()

        # Wait for result
        result = self._output_queue.get()

        t_wait = _time.perf_counter()

        if "error" in result:
            raise RuntimeError(
                f"ImageProcessorWorker failed: {result['error']}"
            )

        # Read output from shared memory (zero-copy: no .copy() needed since
        # the shm won't be overwritten until we call preprocess again)
        pv_shape = result["pv_shape"]
        pv_dtype = np.dtype(result["pv_dtype"])
        pv_bytes = result["pv_bytes"]

        pv_np = np.ndarray(
            pv_shape, dtype=pv_dtype,
            buffer=self._output_shm.buf[:pv_bytes]
        )
        # torch.from_numpy shares memory with numpy; since the caller
        # slices immediately into per-pair tensors (which creates copies),
        # this is safe without an explicit copy here.
        pixel_values = torch.from_numpy(pv_np)
        grid_thw = result["grid_thw"]

        t_copy_out = _time.perf_counter()

        logger.info(
            f"[WORKER IPC TIMING] num_images={num_images}, "
            f"input_bytes={total_bytes / 1024 / 1024:.1f}MB, "
            f"output_bytes={pv_bytes / 1024 / 1024:.1f}MB, "
            f"stack={( t_stack - t0)*1000:.2f}ms, "
            f"copy_in={( t_copy_in - t_stack)*1000:.2f}ms, "
            f"queue_put={( t_put - t_copy_in)*1000:.2f}ms, "
            f"wait_result={( t_wait - t_put)*1000:.2f}ms, "
            f"copy_out={( t_copy_out - t_wait)*1000:.2f}ms, "
            f"TOTAL={( t_copy_out - t0)*1000:.2f}ms"
        )

        return {"pixel_values": pixel_values, "grid_thw": grid_thw}

    def is_alive(self):
        return self._process.is_alive()

    def __del__(self):
        try:
            self._input_shm.close()
            self._input_shm.unlink()
            self._output_shm.close()
            self._output_shm.unlink()
        except Exception:
            pass


class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


def _convert(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


_mm_grid_attrs = {
    # Kimi K2.5 HF processor uses grid_thws (see base_processor.ATTR_NAME_TO_MODALITY).
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws", "grid_thws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality, model_type: Optional[str] = None):
    # Kimi K2.5 vision processor only emits `grid_thws`; prefer it over generic keys
    # so we never pick a mis-typed or stale `image_grid_hws` field from kwargs.
    attrs = _mm_grid_attrs[modality]
    if (model_type or "").lower() in [
        "kimi_k25",
        "kimi_vl",
    ] and modality == Modality.IMAGE:
        attrs = ("grid_thws", "image_grid_thw", "image_grid_hws")
    for attr in attrs:
        if attr in mm_inputs and mm_inputs[attr] is not None:
            return mm_inputs[attr]
    raise ValueError(f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}")


def _get_mm_feature(mm_inputs, modality):
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _build_mm_aux_data(mm_inputs):
    """
    Build auxiliary data for video modality.
    """
    aux_data = {
        "video_timestamps": mm_inputs.get("video_timestamps", None),
        "second_per_grid_ts": mm_inputs.get("second_per_grid_ts", None),
    }
    return aux_data


class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        self._build_vision_config(server_args.mm_process_config)

        # BeeBeeOmni: replace image processor with optimized variant that
        # supports the required paired-image preprocessing.
        if self.model_type == "llavaqwen2_omni":
            try:
                from sglang.srt.multimodal.image_processor_opt import (
                    Qwen25VLImageProcessorOptimized,
                )

                self.image_processor = (
                    Qwen25VLImageProcessorOptimized.from_pretrained(
                        server_args.tokenizer_path or server_args.model_path
                    )
                )
                logger.info(
                    "BeeBeeOmni: injected Qwen25VLImageProcessorOptimized"
                )
            except Exception as e:
                logger.warning(
                    f"BeeBeeOmni: failed to load optimized image processor: {e}"
                )

            # Create isolated worker process for image preprocessing to avoid
            # numexpr thread starvation from mooncake/torch threads.
            if use_image_processor_isolation:
                try:
                    self.image_processor_worker = ImageProcessorWorker(
                        model_path=server_args.tokenizer_path or server_args.model_path,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create ImageProcessorWorker, "
                        f"falling back to in-process: {e}"
                    )
                    self.image_processor_worker = None
            else:
                self.image_processor_worker = None

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
        )
        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()

        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
                EmbeddingCacheController,
            )

            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                max_pool_size_gb=server_args.mm_global_cache_pool_size_gb,
                max_batch_groups=server_args.mm_global_cache_max_batch_groups,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            self.server_args.disaggregation_ib_device
                            or self.server_args.mooncake_ib_device
                        ),
                    )

            self.embedding_to_send = dict()

        logger.info(f"rank {rank} init finish ")

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time."""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            # BeeBeeOmni: projector output_size is the final embedding dim
            out_size = getattr(vision_cfg, "output_size", None)
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_size is not None and int(out_size) > 0:
                dims[Modality.IMAGE] = int(out_size)
                dims[Modality.VIDEO] = int(out_size)
            elif out_hs is not None:
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            # BeeBeeOmni: projector output_size is the final embedding dim,
            # not d_model (which is the Whisper encoder internal dim).
            for attr in ("output_size", "output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val) 
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    def _build_vision_config(self, mm_process_config):
        """
        Validate vision config, used for image/video/audio.
        If not provided, keep default values.
        """
        self.vision_config = (
            mm_process_config.get("vision_config", {})
            if mm_process_config is not None
            else {}
        )
        for modality_str in ["image", "video", "audio"]:
            if not self.vision_config.get(modality_str, None):
                self.vision_config[modality_str] = {}
            if self.use_image_processor_gpu:
                self.vision_config[modality_str]["device"] = self.device

            if modality_str == "video":
                video_defaults = {"fps": 2.0, "max_frames": 768, "min_frames": 4}
                for k, v in video_defaults.items():
                    self.vision_config["video"].setdefault(k, v)

            if modality_str == "audio":
                if "return_attention_mask" not in self.vision_config["audio"]:
                    self.vision_config["audio"]["return_attention_mask"] = True
                if "padding" not in self.vision_config["audio"]:
                    if self.model_type == "qwen2_audio":
                        # For Qwen2Audio, use padding="max_length"
                        # (same as https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L93)
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    # keep same logic as base_processor.py
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        if self.audio_processor.__class__.__name__ in {
                            "Gemma3nProcessor",
                            "GlmAsrProcessor",
                            "Qwen2AudioProcessor",
                            "Qwen3OmniMoeProcessor",
                        }:
                            self.vision_config["audio"]["truncation"] = False

    def _load_mm_processor(self, server_args: ServerArgs):
        """
        Load image/video/audio processor separately,
        avoid issues with AutoProcessor not recognizing certain models
        """
        from transformers import AutoImageProcessor, AutoVideoProcessor

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            self.video_processor = AutoVideoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            # Note: AutoProcessor is used for audio processor
            _audio_proc = AutoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
            if not hasattr(_audio_proc, "feature_extractor"):
                logger.warning(
                    "Loaded AutoProcessor has no feature_extractor attribute, "
                    "audio processing will be unavailable."
                )
                self.audio_processor = None
            else:
                self.audio_processor = _audio_proc
        except Exception as e:
            logger.warning(f"Failed to load audio processor: {e}")
            self.audio_processor = None

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.
        If data is precomputed, returns directly.
        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, False)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    # Needed only when `img` is a PIL image
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(self, items, modalities):
        futures = []
        task_info = []

        for data, modality in zip(items, modalities):
            if modality is not None:
                futures.append(
                    self.io_executor.submit(
                        self._load_single_item,
                        data,
                        modality,
                    )
                )
                task_info.append((modality, data))
        return futures, task_info

    def _get_feat_extract_output_lengths(self, feature_lens):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # qwen2_audio/qwen2.5_omni
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        # qwen3_asr / qwen3_omni_moe (same audio encoder architecture)
        elif self.model_type in ["qwen3_asr", "qwen3_omni_moe"]:
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        else:
            # fallback to original HF audio sample logic for other models
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

    async def _flatten_and_load_videos(self, mm_items):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        futures, _ = self.submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items)
        )
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

        video_processor_kwargs = {}
        if "qwen" in self.model_type:
            # for qwen-series model, do sample frames before preprocess
            video_processed = [
                await preprocess_video(
                    video, video_config=self.vision_config.get("video", {})
                )
                for video in video_items
            ]
            videos, video_metadata = map(list, zip(*video_processed))
            video_processor_kwargs["do_sample_frames"] = False
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            return videos, video_processor_kwargs
        else:
            raise NotImplementedError(
                f"Video processing is not supported for {self.model_type} model."
            )

    async def _flatten_and_load_data_by_modality(self, mm_items, modality):
        """
        Flatten mm_items structure, load multimodal data concurrently, and restore original structure.

        Returns:
            Same structure as load_mm_items would return, support for image/audio
        """
        # Handle single mm_item (not a list)
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self.submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        # Handle nested list (list of lists)
        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            # Flatten nested structure
            flat_data = []
            flat_indices = []  # Track which group each item belongs to
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            # Submit all tasks concurrently
            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            # Restore nested structure
            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        # Handle simple list
        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            return int(grid[0] * grid[1] * grid[2])

    def _kimi_tokens_from_patch_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """MoonViT + tpool: output len is (h//mh)*(w//mw); temporal dim is pooled (not t*h*w/merge^2)."""
        if isinstance(grid, torch.Tensor):
            flat = grid.flatten()
            _t, h, w = (int(x) for x in flat[:3].tolist())
        else:
            _t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        merge_h, merge_w = self.model_config.hf_config.vision_config.merge_kernel_size
        return (h * w) // (merge_h * merge_w)

    def _beebee_tokens_from_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """BeeBeeOmni: token count = t * round(h/merge * r) * round(w/merge * r),
        where r = 1/sqrt(downsample_ratio). Each grid_thw represents a pair of images,
        so total tokens cover both images in the pair."""
        import math

        if isinstance(grid, torch.Tensor):
            flat = grid.flatten()
            t, h, w = (int(x) for x in flat[:3].tolist())
        else:
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])

        vis_cfg = getattr(self.model_config.hf_config, "vision_config", None)
        spatial_merge_size = getattr(vis_cfg, "spatial_merge_size", 2)
        downsample_ratio = getattr(vis_cfg, "image_downsample_ratio", 16)
        r = 1.0 / math.sqrt(downsample_ratio)

        Mh = max(1, int(round(h / spatial_merge_size * r)))
        Nw = max(1, int(round(w / spatial_merge_size * r)))
        return t * Mh * Nw

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing."""
        if modality == Modality.AUDIO:
            if self.model_type == "llavaqwen2_omni":
                # audio_feature_lens_raw already stores final token counts
                return int(grid.item()) if isinstance(grid, torch.Tensor) else int(grid)
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            if (
                self.model_type in ["kimi_k25", "kimi_vl"]
                and modality == Modality.IMAGE
            ):
                return self._kimi_tokens_from_patch_grid(grid)
            if self.model_type == "llavaqwen2_omni" and modality == Modality.IMAGE:
                return self._beebee_tokens_from_grid(grid)
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    def slice_embedding(
        self, mm_embedding: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor into individual image embeddings."""
        slices, offset = [], 0
        for grid in grid_thw:
            count = self.get_num_tokens(grid, modality)
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        return slices

    def _calculate_hashes_from_raw(
        self, mm_items: List, modality: Modality,
    ) -> tuple:
        """Compute hashes directly from raw mm_items (base64 strings) without
        running the image/audio processor.

        For BeeBeeOmni IMAGE: every 2 consecutive items form one pair → one hash.
          Also returns grid_thw by reading JPEG/PNG header dimensions.
        For AUDIO: each item → one hash, grid_thw is None.
        Returns (hashes: List[int], grid_thw: Optional[torch.Tensor]).
        """
        import hashlib
        import base64
        import io
        import struct

        flat = self._flatten_nested_items(mm_items)
        hashes = []

        if modality == Modality.IMAGE:
            patch_size = getattr(self.image_processor, "patch_size", 14)
            grids = []
            # BeeBeeOmni: every 2 images share one grid_thw → one cache entry
            for i in range(0, len(flat), 2):
                hasher = hashlib.sha256()
                item0 = flat[i]
                hasher.update(item0.encode("utf-8") if isinstance(item0, str) else item0)
                j = min(i + 1, len(flat) - 1)
                item1 = flat[j]
                hasher.update(item1.encode("utf-8") if isinstance(item1, str) else item1)
                hash_bytes = hasher.digest()[:8]
                hashes.append(int.from_bytes(hash_bytes, byteorder="big", signed=False))

                # Read image size from first image of the pair (header only)
                w, h = self._get_image_size_from_raw(item0)
                grid_h = h // patch_size
                grid_w = w // patch_size
                grids.append([1, grid_h, grid_w])

            grid_thw = torch.tensor(grids, dtype=torch.long)
            return hashes, grid_thw
        else:
            # AUDIO: each item is one cache entry
            for item in flat:
                hasher = hashlib.sha256()
                hasher.update(item.encode("utf-8") if isinstance(item, str) else item)
                hash_bytes = hasher.digest()[:8]
                hashes.append(int.from_bytes(hash_bytes, byteorder="big", signed=False))

            return hashes, None

    @staticmethod
    def _get_image_size_from_raw(item) -> tuple:
        """Get (width, height) from a raw image item (base64 data-URI or bytes).
        Uses struct to parse JPEG/PNG header directly — no PIL, no pixel decode.
        Falls back to PIL for other formats.
        """
        import base64
        import struct

        if isinstance(item, str):
            if item.startswith("data:"):
                img_bytes = base64.b64decode(item.split(",", 1)[1])
            else:
                img_bytes = base64.b64decode(item)
        elif isinstance(item, bytes):
            img_bytes = item
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")

        # Try JPEG: starts with FF D8
        if img_bytes[:2] == b'\xff\xd8':
            # Scan for SOF0/SOF2 marker
            i = 2
            while i < len(img_bytes) - 9:
                if img_bytes[i] != 0xFF:
                    i += 1
                    continue
                marker = img_bytes[i + 1]
                if marker in (0xC0, 0xC2):  # SOF0 or SOF2
                    h, w = struct.unpack('>HH', img_bytes[i + 5:i + 9])
                    return (w, h)
                # Skip this segment
                seg_len = struct.unpack('>H', img_bytes[i + 2:i + 4])[0]
                i += 2 + seg_len
        # Try PNG: starts with 89 50 4E 47
        elif img_bytes[:4] == b'\x89PNG':
            w, h = struct.unpack('>II', img_bytes[16:24])
            return (w, h)

        # Fallback to PIL
        import io
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(img_bytes))
        return img.size  # (width, height)

    def _compute_grid_thw_from_raw_images(self, mm_items: List) -> torch.Tensor:
        """Quickly compute grid_thw from raw image data (decode header only, no pixel processing).

        For BeeBeeOmni: every 2 images form one pair → one grid_thw row [1, h//patch, w//patch].
        Uses PIL to read image header (size only, no decompression).
        """
        import io
        import base64
        from PIL import Image as PILImage

        flat = self._flatten_nested_items(mm_items)
        patch_size = getattr(self.image_processor, "patch_size", 14)

        grids = []
        for i in range(0, len(flat), 2):
            item = flat[i]
            # Decode enough to get image size from header
            if isinstance(item, str):
                if item.startswith("data:"):
                    img_bytes = base64.b64decode(item.split(",")[1])
                else:
                    img_bytes = base64.b64decode(item)
            elif isinstance(item, bytes):
                img_bytes = item
            else:
                raise ValueError(f"Unsupported mm_item type: {type(item)}")

            img = PILImage.open(io.BytesIO(img_bytes))
            w, h = img.size  # PIL gives (width, height)
            grid_h = h // patch_size
            grid_w = w // patch_size
            grids.append([1, grid_h, grid_w])

        return torch.tensor(grids, dtype=torch.long)

    def _calculate_hashes_from_features(
        self, mm_feature: torch.Tensor, grid_thw: List, modality: Modality,
        mm_inputs: Optional[dict] = None,
    ) -> List[str]:
        """CPU Task: Compute hashes based on processed feature patches."""
        hashes, offset = [], 0
        logger.info(f"{mm_feature.shape=} with {modality=}")

        # BeeBeeOmni audio: slice by chunk counts (not token counts)
        if (
            self.model_type == "llavaqwen2_omni"
            and modality == Modality.AUDIO
            and mm_inputs is not None
        ):
            chunk_counts = mm_inputs["_beebee_audio_chunk_counts"]
            for cc in chunk_counts:
                feature_slice = mm_feature[offset : offset + cc]
                tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
                tmp_item.set_pad_value()
                hashes.append(tmp_item.hash)
                offset += cc
            return hashes

        for grid in grid_thw:
            num_patches = self.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    async def _encode_missing(
        self,
        mm_feature: torch.Tensor,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

        if self.model_type == "llavaqwen2_omni":
            return await self._encode_missing_beebee(
                mm_feature, mm_inputs, indices, grid_thw, modality, get_feature_fn
            )

        # 1. Slice mm_feature to get only the patches for missing mm items
        sub_feature_list = []
        offsets = [0]
        curr = 0
        for g in grid_thw:
            curr += self.get_num_patches(g, modality)
            offsets.append(curr)

        for idx in indices:
            sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])

        sub_feature = torch.cat(sub_feature_list, dim=0)

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": _convert(sub_feature),
            }
        )

        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs.get(modality, []):
                continue
            val = _convert(v)
            if k in _mm_grid_attrs.get(modality, []):
                mm_item.set(k, val[indices])
            else:
                mm_item.set(k, val)

        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item]).cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

        sub_grids = [grid_thw[i] for i in indices]
        return self.slice_embedding(new_embeddings, sub_grids, modality)

    async def _encode_missing_beebee(
        self,
        mm_feature: torch.Tensor,
        mm_inputs: dict,
        indices: List[int],
        grid_thw,
        modality: Modality,
        get_feature_fn,
    ) -> List[torch.Tensor]:
        """BeeBeeOmni-specific _encode_missing for IMAGE and AUDIO.

        IMAGE:
          - mm_feature is cat'd pixel_values [total_patches, patch_dim].
          - Each grid_thw[i] covers one pair (2 images), patch count = h*w.
          - get_image_feature reads mm_item.model_specific_data["image_grid_thw"].

        AUDIO:
          - mm_feature is input_features [total_chunks, 128, 3000].
          - grid_thw is audio_feature_lens_raw (per-audio token counts) — NOT chunk counts.
          - Chunk counts are in mm_inputs["_beebee_audio_chunk_counts"].
          - get_audio_feature reads mm_item.model_specific_data["audio_length"].
        """
        if modality == Modality.IMAGE:
            # Slice pixel patches by grid (each grid's patch count = t*h*w = h*w since t=1)
            offsets = [0]
            curr = 0
            for g in grid_thw:
                curr += self.get_num_patches(g, modality)  # = t*h*w
                offsets.append(curr)

            sub_feature_list = []
            for idx in indices:
                sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])
            sub_feature = torch.cat(sub_feature_list, dim=0)

            sub_grid_thw = grid_thw[indices] if isinstance(grid_thw, torch.Tensor) else \
                torch.stack([grid_thw[i] for i in indices])

            mm_item = MultimodalDataItem.from_dict(
                {"modality": modality, "feature": _convert(sub_feature)}
            )
            mm_item.model_specific_data["image_grid_thw"] = sub_grid_thw

            with torch.inference_mode():
                new_embeddings = get_feature_fn([mm_item]).cpu()
                if new_embeddings.ndim != 2:
                    new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

            # Slice output by token count per grid pair
            return self.slice_embedding(new_embeddings, sub_grid_thw, modality)

        elif modality == Modality.AUDIO:
            # Slice input_features (mel chunks) by per-audio chunk counts
            chunk_counts = mm_inputs["_beebee_audio_chunk_counts"]  # List[int]
            chunk_lens_grouped = mm_inputs["_beebee_audio_chunk_lens_grouped"]  # List[List[int]]

            # Build offsets based on chunk counts (not token counts)
            offsets = [0]
            curr = 0
            for cc in chunk_counts:
                curr += cc
                offsets.append(curr)

            sub_feature_list = []
            sub_chunk_lens = []
            for idx in indices:
                sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])
                sub_chunk_lens.extend(chunk_lens_grouped[idx])
            sub_feature = torch.cat(sub_feature_list, dim=0)

            mm_item = MultimodalDataItem.from_dict(
                {"modality": modality, "feature": sub_feature}
            )
            mm_item.model_specific_data["audio_length"] = sub_chunk_lens

            with torch.inference_mode():
                new_embeddings = get_feature_fn([mm_item]).cpu()
                if new_embeddings.ndim != 2:
                    new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

            # Slice output by per-audio token count
            sub_token_counts = [grid_thw[i] for i in indices]
            return self._slice_by_token_counts(new_embeddings, sub_token_counts)

        else:
            raise ValueError(f"BeeBeeOmni does not support {modality} in global cache path")

    def _slice_by_token_counts(
        self, embedding: torch.Tensor, token_counts: List
    ) -> List[torch.Tensor]:
        """Slice embedding by a list of token counts (for BeeBeeOmni audio)."""
        slices, offset = [], 0
        for count in token_counts:
            c = int(count.item()) if isinstance(count, torch.Tensor) else int(count)
            slices.append(embedding[offset : offset + c])
            offset += c
        return slices

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        # BeeBeeOmni fast path: hash from raw bytes BEFORE processor
        if self.model_type == "llavaqwen2_omni":
            return await self._encode_with_global_cache_beebee(
                mm_items, modality, req_id, num_parts, part_idx, hashes
            )

        # Generic path: must run processor first to compute hash from features
        mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        if self.rank == 0:
            if hashes is None:
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality, mm_inputs=mm_inputs
                )
            else:
                mm_hashes = hashes
            exist_mask = await self.mm_global_cache.batch_is_exist(mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mm_hashes = None
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)

        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]

        # Step 2: All ranks run ViT together on cache-miss images.
        new_slices = []
        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    prefetch_status[0] = 0

        # Step 4: Broadcast prefetch result to all ranks so they stay in sync.
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, all ranks fallback to ViT for the hit mm items.
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, all ranks running ViT fallback "
                f"for {len(hit_indices)} mm items."
            )
            fallback_slices = await self._encode_missing(
                mm_feature, mm_inputs, hit_indices, modality, get_feature_fn
            )
        else:
            fallback_slices = None

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in cache-hit embeddings (from prefetch or fallback)
            if prefetch_status.item() == 1 and hit_indices:
                cached_slices = self.mm_global_cache.get_embeddings(
                    [mm_hashes[i] for i in hit_indices]
                )
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = cached_slices[i]
            elif fallback_slices is not None:
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:
                # Synchronously register metadata so next request sees the cache
                keys, ptrs, sizes, copy_tasks = self.mm_global_cache.register_batch(
                    all_new_hashes, all_new_slices
                )

                # Background: pinned copy + RDMA PUT (slow, doesn't block next request)
                async def _background_commit():
                    await asyncio.to_thread(
                        self.mm_global_cache.commit_batch,
                        keys, ptrs, sizes, copy_tasks,
                    )

                task = asyncio.create_task(_background_commit())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            aux_data = _build_mm_aux_data(mm_inputs)
            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                mm_embedding,
                **aux_data,
            )
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            return (0, 0, 0, None, None)

    async def _encode_with_global_cache_beebee(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ):
        """BeeBeeOmni optimized global cache path.

        Computes hash from raw base64 strings BEFORE running the image/audio
        processor, so cache hits skip processor + ViT entirely.
        Only cache-miss items go through processor → ViT.
        """
        import time as _time

        t0 = _time.perf_counter()

        # Step 0: Compute hashes from raw data + grid_thw from image headers
        if hashes is None:
            mm_hashes, full_grid_thw = self._calculate_hashes_from_raw(mm_items, modality)
        else:
            mm_hashes = hashes
            full_grid_thw = None
        num_items = len(mm_hashes)

        t_hash = _time.perf_counter()
        logger.info(
            f"BeeBeeOmni raw hash: {num_items} items, "
            f"time={( t_hash - t0)*1000:.2f}ms"
        )

        # Step 1: Rank 0 checks global cache
        t_cache_check = _time.perf_counter()
        if self.rank == 0:
            exist_mask = await self.mm_global_cache.batch_is_exist(mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)
        t_cache_check_done = _time.perf_counter()

        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]

        logger.info(
            f"Req {req_id}: {len(hit_indices)} hits, {len(missing_indices)} misses "
            f"(cache_check={(t_cache_check_done - t_cache_check)*1000:.2f}ms, "
            f"skipping processor for hits)"
        )

        # Step 2: Only process + ViT the cache-miss items
        new_slices = []
        grid_thw = None  # will be set if we have misses
        if missing_indices:
            # Extract only the miss items from raw mm_items
            t_extract = _time.perf_counter()
            miss_mm_items = self._extract_miss_items(mm_items, missing_indices, modality)
            t_extract_done = _time.perf_counter()

            # Run processor only on miss items
            t_proc = _time.perf_counter()
            mm_inputs, get_feature_fn = await self._process_mm_items(
                miss_mm_items, modality
            )
            t_proc_done = _time.perf_counter()
            grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

            # Run ViT on all miss items (no further slicing needed — they're all misses)
            t_vit = _time.perf_counter()
            if modality == Modality.IMAGE:
                mm_feature = torch.cat(mm_inputs["pixel_values"], dim=0)
                mm_item = MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": _convert(mm_feature)}
                )
                mm_item.model_specific_data["image_grid_thw"] = grid_thw

                with torch.inference_mode():
                    new_embeddings = get_feature_fn([mm_item]).cpu()
                    if new_embeddings.ndim != 2:
                        new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

                # Slice output per pair
                new_slices = self.slice_embedding(new_embeddings, grid_thw, modality)

            elif modality == Modality.AUDIO:
                mm_feature = mm_inputs["input_features"]
                chunk_counts = mm_inputs["_beebee_audio_chunk_counts"]
                chunk_lens_grouped = mm_inputs["_beebee_audio_chunk_lens_grouped"]

                # Flatten chunk lens for the model
                all_chunk_lens = []
                for group in chunk_lens_grouped:
                    all_chunk_lens.extend(group)

                mm_item = MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": mm_feature}
                )
                mm_item.model_specific_data["audio_length"] = all_chunk_lens

                with torch.inference_mode():
                    new_embeddings = get_feature_fn([mm_item]).cpu()
                    if new_embeddings.ndim != 2:
                        new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

                # Slice output by per-audio token count
                audio_token_counts = mm_inputs.get("audio_feature_lens_raw", None)
                if audio_token_counts is not None:
                    new_slices = self._slice_by_token_counts(
                        new_embeddings, audio_token_counts
                    )
                else:
                    new_slices = [new_embeddings]

            t_vit_done = _time.perf_counter()
            logger.info(
                f"Req[{req_id}] MISS breakdown: "
                f"extract={( t_extract_done - t_extract)*1000:.2f}ms, "
                f"processor={( t_proc_done - t_proc)*1000:.2f}ms, "
                f"ViT={( t_vit_done - t_vit)*1000:.2f}ms, "
                f"total_miss={( t_vit_done - t_extract)*1000:.2f}ms"
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)
        t_prefetch_start = _time.perf_counter()

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                # Get token counts from cache metadata (already stored at insert time)
                hit_tokens = []
                for h in hit_hashes:
                    meta = self.mm_global_cache.hash_to_metadata.get(h)
                    if meta is not None:
                        hit_tokens.append(meta[1])  # (offset, num_tokens, dim, size_bytes)
                    else:
                        # Fallback: shouldn't happen if batch_is_exist returned True
                        hit_tokens.append(0)

                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to full ViT."
                    )
                    prefetch_status[0] = 0

        t_prefetch_done = _time.perf_counter()
        if hit_indices:
            logger.info(
                f"Req {req_id} HIT prefetch: {len(hit_indices)} items, "
                f"time={(t_prefetch_done - t_prefetch_start)*1000:.2f}ms"
            )

        # Step 4: Broadcast prefetch result
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, run processor + ViT for ALL items as fallback
        fallback_slices = None
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, running full ViT for all items."
            )
            # Process ALL items (hits need ViT now)
            hit_mm_items = self._extract_miss_items(mm_items, hit_indices, modality)
            hit_inputs, hit_feat_fn = await self._process_mm_items(hit_mm_items, modality)
            hit_grid = _get_mm_grid_dim(hit_inputs, modality, self.model_type)

            if modality == Modality.IMAGE:
                hit_feature = torch.cat(hit_inputs["pixel_values"], dim=0)
                hit_item = MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": _convert(hit_feature)}
                )
                hit_item.model_specific_data["image_grid_thw"] = hit_grid
                with torch.inference_mode():
                    hit_emb = hit_feat_fn([hit_item]).cpu()
                    if hit_emb.ndim != 2:
                        hit_emb = hit_emb.reshape(-1, hit_emb.shape[-1])
                fallback_slices = self.slice_embedding(hit_emb, hit_grid, modality)
            elif modality == Modality.AUDIO:
                hit_feature = hit_inputs["input_features"]
                hit_chunk_lens = []
                for group in hit_inputs["_beebee_audio_chunk_lens_grouped"]:
                    hit_chunk_lens.extend(group)
                hit_item = MultimodalDataItem.from_dict(
                    {"modality": modality, "feature": hit_feature}
                )
                hit_item.model_specific_data["audio_length"] = hit_chunk_lens
                with torch.inference_mode():
                    hit_emb = hit_feat_fn([hit_item]).cpu()
                    if hit_emb.ndim != 2:
                        hit_emb = hit_emb.reshape(-1, hit_emb.shape[-1])
                hit_token_counts = hit_inputs.get("audio_feature_lens_raw", None)
                if hit_token_counts is not None:
                    fallback_slices = self._slice_by_token_counts(hit_emb, hit_token_counts)
                else:
                    fallback_slices = [hit_emb]

            # Also update grid_thw if we didn't have misses
            if grid_thw is None:
                grid_thw = hit_grid

        # Step 6: Rank 0 assembles final embedding
        if self.rank == 0:
            t_assemble = _time.perf_counter()
            # full_grid_thw was already computed in Step 0 (from image headers)
            # For AUDIO, use the grid from processor if available
            if full_grid_thw is None:
                full_grid_thw = grid_thw

            # Assemble final embedding
            if not missing_indices and prefetch_status.item() == 1:
                # All-hit fast path: single merged copy from cache (non-pinned)
                mm_embedding = self.mm_global_cache.get_embeddings_merged(mm_hashes)
                logger.info(
                    f"Req {req_id} assemble ALL-HIT path: shape={mm_embedding.shape}, "
                    f"size={mm_embedding.numel() * mm_embedding.element_size() / 1024 / 1024:.1f}MB"
                )
            else:
                final_slices = [None] * num_items

                for i, idx in enumerate(missing_indices):
                    final_slices[idx] = new_slices[i]

                if prefetch_status.item() == 1 and hit_indices:
                    cached_slices = self.mm_global_cache.get_embeddings(
                        [mm_hashes[i] for i in hit_indices]
                    )
                    for i, idx in enumerate(hit_indices):
                        final_slices[idx] = cached_slices[i]
                elif fallback_slices is not None:
                    for i, idx in enumerate(hit_indices):
                        final_slices[idx] = fallback_slices[i]

                # Pre-allocate contiguous buffer and copy
                total_tokens = sum(s.shape[0] for s in final_slices)
                embed_dim = final_slices[0].shape[1]
                mm_embedding = torch.empty(
                    total_tokens, embed_dim, dtype=final_slices[0].dtype
                )
                _offset = 0
                for s in final_slices:
                    n = s.shape[0]
                    mm_embedding[_offset : _offset + n].copy_(s)
                    _offset += n

            # Background insert new embeddings
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:
                # Synchronously register metadata so next request sees the cache
                keys, ptrs, sizes, copy_tasks = self.mm_global_cache.register_batch(
                    all_new_hashes, all_new_slices
                )

                # Background: pinned copy + RDMA PUT (slow, doesn't block next request)
                async def _background_commit():
                    await asyncio.to_thread(
                        self.mm_global_cache.commit_batch,
                        keys, ptrs, sizes, copy_tasks,
                    )

                task = asyncio.create_task(_background_commit())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            # Build aux_data with full grid info
            if full_grid_thw is not None:
                if modality == Modality.IMAGE:
                    aux_data = {
                        "image_grid_thw": full_grid_thw,
                    }
                elif modality == Modality.AUDIO:
                    aux_data = {}
                else:
                    aux_data = {}
            else:
                aux_data = {}

            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                full_grid_thw if full_grid_thw is not None else torch.zeros(num_items, 3, dtype=torch.long),
                modality,
                mm_embedding,
                **aux_data,
            )

            t_end = _time.perf_counter()
            logger.info(
                f"Req {req_id} TIMING SUMMARY: "
                f"hash={(t_hash - t0)*1000:.2f}ms, "
                f"cache_check={(t_cache_check_done - t_cache_check)*1000:.2f}ms, "
                f"prefetch={(t_prefetch_done - t_prefetch_start)*1000:.2f}ms, "
                f"assemble={(t_end - t_assemble)*1000:.2f}ms, "
                f"TOTAL={(t_end - t0)*1000:.2f}ms "
                f"({len(hit_indices)} hits, {len(missing_indices)} misses)"
            )

            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            return (0, 0, 0, None, None)

    def _extract_miss_items(
        self, mm_items: List, indices: List[int], modality: Modality
    ) -> List:
        """Extract a subset of raw mm_items corresponding to given indices.

        For BeeBeeOmni IMAGE: each index corresponds to a pair (2 images).
        For AUDIO: each index corresponds to one audio item.
        """
        flat = self._flatten_nested_items(mm_items)

        if modality == Modality.IMAGE:
            # Each index = one pair = 2 consecutive images
            result = []
            for idx in indices:
                i = idx * 2
                result.append(flat[i])
                j = min(i + 1, len(flat) - 1)
                result.append(flat[j])
            return result
        else:
            # AUDIO: 1:1 mapping
            return [flat[i] for i in indices]

    async def _flatten_and_load_audios(self, mm_items):
        """
        Flatten mm_items structure, load audios concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    async def _flatten_and_load_images(self, mm_items):
        """
        Flatten mm_items structure, load images concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames, used for qwen3_vl models."""
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L255
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    @staticmethod
    def _flatten_nested_items(items):
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(MMEncoder._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    def _normalize_kimi_encoder_images(self, images):
        """Normalize Kimi image inputs for the image processor call."""
        from PIL import Image as PILImage

        def wrap_one(img):
            if isinstance(img, dict) and img.get("type") in ("image", "video_chunk"):
                return [img]
            if isinstance(img, PILImage.Image):
                return [{"type": "image", "image": img}]
            return [img]

        if not images:
            return images

        # Disagg may supply nested lists from grouped routing.
        images = self._flatten_nested_items(images)

        # Kimi-VL image processor expects a flat list of concrete images.
        if self.model_type == "kimi_vl":
            normalized = []
            for img in images:
                if (
                    isinstance(img, dict)
                    and img.get("type") == "image"
                    and "image" in img
                ):
                    inner = img["image"]
                    if isinstance(inner, (list, tuple)):
                        normalized.extend(self._flatten_nested_items(inner))
                    else:
                        normalized.append(inner)
                else:
                    normalized.append(img)
            return normalized

        # Kimi-K2.5 vision processor expects media dicts.
        normalized = []
        for img in images:
            wrapped = wrap_one(img)
            for media in wrapped:
                # Some pipelines may produce {"type": "image", "image": [PIL]}.
                # Split it into one media item per concrete image object.
                if (
                    isinstance(media, dict)
                    and media.get("type") == "image"
                    and isinstance(media.get("image"), (list, tuple))
                ):
                    for inner in self._flatten_nested_items(media["image"]):
                        normalized.append({**media, "image": inner})
                else:
                    normalized.append(media)

        return normalized

    # ── BeeBeeOmni-specific processing ────────────────────────────────

    def _process_beebee_images_sync(self, images):
        """BeeBeeOmni paired-image processing (sync, runs in thread pool).

        Every two consecutive images share one ``grid_thw`` row.
        Images are bucketed by (w, h), batch-preprocessed with
        ``Qwen25VLImageProcessorOptimized``, then sliced per pair.

        If an ImageProcessorWorker is available, preprocessing is dispatched
        to an isolated subprocess to avoid numexpr thread starvation.
        """
        from collections import defaultdict

        from PIL import Image as PILImage

        total_imgs = len(images)
        pairs_metadata = []
        for i in range(0, total_imgs, 2):
            if i + 1 < total_imgs:
                curr_pair = [images[i], images[i + 1]]
            else:
                curr_pair = [images[i], images[i]]
            # Support both PIL Image and numpy array
            img0 = curr_pair[0]
            if isinstance(img0, PILImage.Image):
                w, h = img0.size  # PIL: (width, height)
            else:
                h, w = img0.shape[:2]  # numpy: (height, width, channels)
            pairs_metadata.append(
                {"idx": i // 2, "images": curr_pair, "size": (w, h)}
            )

        buckets = defaultdict(list)
        for p in pairs_metadata:
            buckets[p["size"]].append(p)

        ordered_pixel_values = [None] * len(pairs_metadata)
        ordered_grids = [None] * len(pairs_metadata)

        # Choose preprocess backend: isolated worker or in-process
        _use_worker = (
            getattr(self, "image_processor_worker", None) is not None
            and self.image_processor_worker.is_alive()
        )

        for _size, bucket_pairs in buckets.items():
            batch_images = []
            for p in bucket_pairs:
                batch_images.extend(p["images"])
            t_pre = time.perf_counter()
            if _use_worker:
                image_outputs = self.image_processor_worker.preprocess(
                    batch_images, return_tensors="pt", patch_reshape_method="torch"
                )
            else:
                image_outputs = self.image_processor.preprocess(
                    batch_images, return_tensors="pt", patch_reshape_method="torch"
                )
            t_post = time.perf_counter()
            logger.info(
                f"BeeBeeOmni bucket preprocess: size={_size}, "
                f"num_imgs={len(batch_images)}, "
                f"time={(t_post - t_pre)*1000:.2f}ms, "
                f"isolated={_use_worker}"
            )
            bucket_features = image_outputs.get("pixel_values")
            bucket_thw = image_outputs.get("grid_thw")
            grid_h = int(bucket_thw[0, 1])
            grid_w = int(bucket_thw[0, 2])
            stride = grid_h * grid_w
            for i, p in enumerate(bucket_pairs):
                pair_feature = bucket_features[i * stride : (i + 1) * stride]
                if _use_worker:
                    # Must clone: the slice is a view into shared memory which
                    # will be overwritten by the next bucket's preprocess call.
                    pair_feature = pair_feature.clone()
                pair_grid = torch.tensor(
                    [[1, grid_h, grid_w]], dtype=torch.long
                )
                ordered_pixel_values[p["idx"]] = pair_feature
                ordered_grids[p["idx"]] = pair_grid

        return {
            "pixel_values": ordered_pixel_values,
            "image_grid_thw": torch.cat(ordered_grids, dim=0),
        }

    async def _process_beebee_images(self, images):
        """Run paired-image preprocessing.

        When an isolated worker process is used, the blocking queue.get()
        is wrapped in run_in_executor to avoid blocking the asyncio loop.
        """
        _use_worker = (
            getattr(self, "image_processor_worker", None) is not None
            and self.image_processor_worker.is_alive()
        )
        if _use_worker:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor, self._process_beebee_images_sync, images
            )
        return self._process_beebee_images_sync(images)

    @staticmethod
    def _process_beebee_single_audio(audio):
        """BeeBeeOmni audio: resample → 30-s chunks → log-mel spectrogram.

        Returns ``(mels, chunk_lens)`` where *mels* has shape
        ``[num_chunks, 128, 3000]`` and *chunk_lens* is a list of raw
        waveform sample counts per chunk.
        """
        import torch.nn.functional as F

        try:
            import torchaudio.functional as F_audio
        except ImportError:
            F_audio = None
        from whisper.audio import log_mel_spectrogram

        WHISPER_SAMPLING_RATE = 16000
        WHISPER_MAX_LENGTH = 480000  # 30 s
        MIN_AUDIO_LEN = 4000
        WHISPER_N_MEL_BINS = 128

        if isinstance(audio, dict):
            waveform = audio["array"]
            sr = audio.get("sampling_rate", WHISPER_SAMPLING_RATE)
        else:
            waveform = audio
            sr = WHISPER_SAMPLING_RATE

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        if sr != WHISPER_SAMPLING_RATE:
            if F_audio is not None:
                waveform = F_audio.resample(
                    waveform, orig_freq=sr, new_freq=WHISPER_SAMPLING_RATE
                )
            else:
                import librosa

                waveform_np = librosa.resample(
                    waveform.numpy(),
                    orig_sr=sr,
                    target_sr=WHISPER_SAMPLING_RATE,
                )
                waveform = torch.from_numpy(waveform_np)

        seq_len = waveform.shape[0]
        if seq_len < MIN_AUDIO_LEN:
            waveform = F.pad(
                waveform, (0, MIN_AUDIO_LEN - seq_len), mode="constant", value=0.0
            )
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

    async def _process_beebee_audio(self, mm_items):
        """Process audio items the BeeBeeOmni way and return a
        ``processor_input`` dict compatible with ``get_audio_feature``.

        Multiple audios are processed concurrently via the thread pool
        (mirrors the ``executor.map`` pattern in BeeBeeOmniProcessor).
        """
        import time as _time

        t0 = _time.perf_counter()
        audios = await self._flatten_and_load_audios(mm_items)
        t_load = _time.perf_counter()

        loop = asyncio.get_running_loop()
        # Fan out audio processing across the io_executor (parallel for
        # multiple audios, same pattern as _process_beebee_images).
        tasks = [
            loop.run_in_executor(
                self.executor, self._process_beebee_single_audio, audio
            )
            for audio in audios
        ]
        results = await asyncio.gather(*tasks) if tasks else []
        t_mel = _time.perf_counter()

        all_mels = []
        all_chunk_lens = []
        for mels, chunk_lens in results:
            all_mels.append(mels)
            all_chunk_lens.append(chunk_lens)

        batched_mels = torch.cat(all_mels, dim=0)  # [total_chunks, 128, 3000]

        # Flatten per-audio chunk-lens into one list (get_audio_feature
        # iterates items → chunks; with a single item this is equivalent).
        flat_chunk_lens = [l for cls in all_chunk_lens for l in cls]

        # Pre-compute per-audio token counts for downstream get_mm_data.
        WHISPER_HOP_LENGTH = 320
        audio_downsample_ratio = 10
        per_audio_tokens = []
        for chunk_lens in all_chunk_lens:
            tokens = sum(
                (l // WHISPER_HOP_LENGTH + audio_downsample_ratio - 1)
                // audio_downsample_ratio
                for l in chunk_lens
            )
            per_audio_tokens.append(tokens)

        # Per-audio chunk counts — needed for global cache slicing of input_features.
        per_audio_chunks = [len(cls) for cls in all_chunk_lens]

        t_end = _time.perf_counter()
        total_chunks = batched_mels.shape[0]
        total_duration_samples = sum(flat_chunk_lens)
        logger.info(
            f"Audio processing: load={(t_load - t0)*1000:.2f}ms, "
            f"mel_extract={(t_mel - t_load)*1000:.2f}ms, "
            f"post={(t_end - t_mel)*1000:.2f}ms, "
            f"total={(t_end - t0)*1000:.2f}ms, "
            f"num_audios={len(audios)}, chunks={total_chunks}, "
            f"samples={total_duration_samples}"
        )

        return {
            "input_features": batched_mels,
            # Raw waveform sample counts per chunk — what get_audio_feature reads.
            "audio_length": flat_chunk_lens,
            "audio_feature_lens_raw": torch.tensor(
                per_audio_tokens, dtype=torch.long
            ),
            # Per-audio chunk counts (for slicing input_features in global cache path)
            "_beebee_audio_chunk_counts": per_audio_chunks,
            # Per-audio chunk lens (grouped, for reconstructing audio_length per item)
            "_beebee_audio_chunk_lens_grouped": all_chunk_lens,
        }

    # ── End BeeBeeOmni-specific ───────────────────────────────────────

    async def _process_mm_items(self, mm_items, modality):
        if modality == Modality.IMAGE and self.image_processor:
            t_start = time.perf_counter()
            if self.model_type == "llavaqwen2_omni":
                # Use fast_load_image_to_numpy (cv2 decode) instead of PIL
                # for BeeBeeOmni — matches the original processor's fast path.
                from sglang.srt.multimodal.processors.beebeeomni import (
                    fast_load_image_to_numpy,
                )

                loop = asyncio.get_running_loop()
                flat_items = self._flatten_nested_items(mm_items)
                futures = [
                    self.io_executor.submit(fast_load_image_to_numpy, item)
                    for item in flat_items
                ]
                async_futures = [asyncio.wrap_future(f) for f in futures]
                images = list(await asyncio.gather(*async_futures))
                t_load = time.perf_counter()
                processor_input = await self._process_beebee_images(images)
            else:
                images = await self._flatten_and_load_images(mm_items)
                t_load = time.perf_counter()
                image_config = self.vision_config.get("image", {})
                if self.model_type in ["kimi_k25", "kimi_vl"]:
                    images = self._normalize_kimi_encoder_images(images)
                    processor_input = self.image_processor(
                        images=images, **image_config
                    )
                else:
                    processor_input = self.image_processor(
                        images=images, **image_config
                    )
            t_process = time.perf_counter()
            logger.info(
                f"Image processing: load={(t_load - t_start)*1000:.2f}ms, "
                f"preprocess={(t_process - t_load)*1000:.2f}ms, "
                f"total={(t_process - t_start)*1000:.2f}ms, "
                f"num_images={len(images)}"
            )
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_image_feature
            else:
                get_feature_method = self.model.get_image_feature
        elif modality == Modality.VIDEO and self.video_processor:
            videos, video_processor_kwargs = await self._flatten_and_load_videos(
                mm_items
            )
            processor_input = self.video_processor(
                videos=videos, **video_processor_kwargs
            )
            # Get additional video metadata
            if (
                self.model_type
                in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
                and video_processor_kwargs.get("video_metadata", None) is not None
            ):
                # For qwen3-vl/qwen3.5 models, we need to store the video timestamps
                video_metadata = video_processor_kwargs["video_metadata"]
                try:
                    merge_size = (
                        self.model_config.hf_config.vision_config.spatial_merge_size
                    )
                except (AttributeError, KeyError):
                    merge_size = 2  # Default merge_size

                video_timestamps = []
                for metadata in video_metadata:
                    video_fps = metadata.get("fps", None) or 24  # original video fps
                    frames_indices = metadata.get("frames_indices", None)
                    timestamps = self._calculate_timestamps(
                        frames_indices, video_fps, merge_size
                    )
                    video_timestamps.append(timestamps)
                processor_input["video_timestamps"] = video_timestamps
            elif (
                self.model_type in ["qwen2_5_vl", "qwen2_5_omni", "qwen3_omni_moe"]
                and processor_input.get("video_grid_thw", None) is not None
            ):
                # For omni/qwen2_5_vl models, calculate second_per_grid_ts for rotary embedding
                video_grid_thw = processor_input["video_grid_thw"]
                try:
                    temporal_patch_size = self.video_processor.temporal_patch_size
                except AttributeError:
                    temporal_patch_size = 2  # Default temporal_patch_size
                # get sampled fps, default: 2
                fps_list = [
                    self.vision_config.get("video", {}).get("fps", None) or 2
                ] * len(video_grid_thw)
                second_per_grid_ts = [(temporal_patch_size / fps) for fps in fps_list]
                second_per_grid_ts_tensor = torch.tensor(
                    second_per_grid_ts, dtype=torch.float32
                )
                processor_input["second_per_grid_ts"] = second_per_grid_ts_tensor

            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_video_feature
            else:
                get_feature_method = self.model.get_video_feature
        elif modality == Modality.AUDIO:
            if self.model_type == "llavaqwen2_omni":
                # BeeBeeOmni: custom 30-s chunked whisper mel pipeline
                processor_input = await self._process_beebee_audio(mm_items)
            elif self.audio_processor:
                audios = await self._flatten_and_load_audios(mm_items)
                audio_config = self.vision_config.get("audio", {})
                processor_input = self.audio_processor.feature_extractor(
                    audios, **audio_config
                )
                processor_input["feature_attention_mask"] = processor_input.pop(
                    "attention_mask"
                )
                # convert to same format as image/video
                input_lengths = torch.tensor(
                    processor_input["feature_attention_mask"].sum(-1), dtype=torch.long
                )
                processor_input["audio_feature_lens_raw"] = input_lengths
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)
                processor_input["audio_feature_lens"] = output_lengths
            else:
                raise ValueError(
                    f"Audio modality requested but no audio processor available."
                )
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_audio_feature
            else:
                get_feature_method = self.model.get_audio_feature
        else:
            raise ValueError(
                f"Currently only support image, video and audio modalities, {modality} modality has no processor available."
            )

        return processor_input, get_feature_method

    async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
        """Encode multimodal items.

        Optimized path (BeeBeeOmni with enable_prefix_mm_cache):
          1. Compute hash from raw data BEFORE processor (cheap, uses _calculate_hashes_from_raw)
          2. Check local mm_cache — cache hits skip processor + ViT
          3. Only process & encode cache-miss items
          4. Assemble final embedding from cache hits + new encodings

        IMAGE: every 2 raw items form one pair → one hash → one cache entry.
        AUDIO: each raw item → one hash → one cache entry.
        """
        import time as _time

        t0 = _time.perf_counter()

        # ── Fast path: hash from raw data, then check cache before processor ──
        if self.server_args.enable_prefix_mm_cache and self.model_type == "llavaqwen2_omni":
            # Step 0: compute hashes from raw bytes (IMAGE: pair-wise, AUDIO: per-item)
            mm_hashes, full_grid_thw = self._calculate_hashes_from_raw(mm_items, modality)
            num_items = len(mm_hashes)

            # Step 1: check local mm_cache for each hash
            cached_embeddings = [None] * num_items
            missing_indices = []
            async with self.mm_cache_lock:
                for i, h in enumerate(mm_hashes):
                    result = self.mm_cache.get_single(h)
                    if result is not None:
                        cached_embeddings[i] = result.embedding
                    else:
                        missing_indices.append(i)

            t_cache = _time.perf_counter()
            hit_count = num_items - len(missing_indices)
            logger.info(
                f"_encode cache check: {num_items} items, "
                f"{hit_count} hits, {len(missing_indices)} misses, "
                f"time={( t_cache - t0)*1000:.2f}ms"
            )

            # All cache hit — skip processor + ViT entirely
            if not missing_indices:
                mm_embedding = torch.cat(
                    [e for e in cached_embeddings if e is not None], dim=0
                )
                # grid_thw already computed from raw image headers
                aux_data = {}
                if modality == Modality.IMAGE and full_grid_thw is not None:
                    aux_data["image_grid_thw"] = full_grid_thw
                    grid_dim = full_grid_thw
                elif modality == Modality.AUDIO:
                    # Reconstruct audio_feature_lens_raw from cached slice shapes
                    audio_feature_lens = torch.tensor(
                        [e.shape[0] for e in cached_embeddings if e is not None],
                        dtype=torch.long,
                    )
                    grid_dim = audio_feature_lens
                else:
                    grid_dim = full_grid_thw

                t_end = _time.perf_counter()
                logger.info(
                    f"_encode ALL-HIT ({modality.name}): {num_items} items, "
                    f"hash={( t_cache - t0)*1000:.2f}ms, "
                    f"cache_lookup={( t_cache - t0)*1000:.2f}ms, "
                    f"total={( t_end - t0)*1000:.2f}ms"
                )
                return (grid_dim, mm_embedding, aux_data)

            # Step 2: extract only miss items and process them
            miss_mm_items = self._extract_miss_items(mm_items, missing_indices, modality)

            try:
                mm_inputs, get_feature_fn = await self._process_mm_items(
                    miss_mm_items, modality
                )
            except NotImplementedError as e:
                raise InternalError(f"Not implemented error: {str(e)}")
            except Exception as e:
                raise BadRequestError(f"Failed to process mm items: {str(e)}")

            t_proc = _time.perf_counter()

            try:
                # Step 3: run ViT only on miss items
                if modality == Modality.IMAGE:
                    pixel_values = mm_inputs["pixel_values"]
                    feature = torch.cat(pixel_values, dim=0)
                    mm_item = MultimodalDataItem.from_dict(
                        {"modality": modality, "feature": feature}
                    )
                    miss_grid_thw = mm_inputs["image_grid_thw"]
                    mm_item.model_specific_data["image_grid_thw"] = miss_grid_thw

                    with torch.inference_mode():
                        new_embedding = get_feature_fn([mm_item]).cpu()
                    if new_embedding.ndim != 2:
                        new_embedding = new_embedding.reshape(-1, new_embedding.shape[-1])

                    # Slice per pair
                    new_slices = self.slice_embedding(new_embedding, miss_grid_thw, modality)

                elif modality == Modality.AUDIO:
                    mm_feature = mm_inputs["input_features"]
                    chunk_lens_grouped = mm_inputs["_beebee_audio_chunk_lens_grouped"]
                    all_chunk_lens = []
                    for group in chunk_lens_grouped:
                        all_chunk_lens.extend(group)

                    mm_item = MultimodalDataItem.from_dict(
                        {"modality": modality, "feature": mm_feature}
                    )
                    mm_item.model_specific_data["audio_length"] = all_chunk_lens

                    with torch.inference_mode():
                        new_embedding = get_feature_fn([mm_item]).cpu()
                    if new_embedding.ndim != 2:
                        new_embedding = new_embedding.reshape(-1, new_embedding.shape[-1])

                    audio_token_counts = mm_inputs.get("audio_feature_lens_raw", None)
                    if audio_token_counts is not None:
                        new_slices = self._slice_by_token_counts(
                            new_embedding, audio_token_counts
                        )
                    else:
                        new_slices = [new_embedding]
                else:
                    raise ValueError(
                        f"BeeBeeOmni does not support {modality} in encoder-only mode"
                    )

                t_vit = _time.perf_counter()

                # Step 4: store miss results into cache and assemble final embedding
                async with self.mm_cache_lock:
                    for i, idx in enumerate(missing_indices):
                        h = mm_hashes[idx]
                        self.mm_cache.set(h, EmbeddingResult(embedding=new_slices[i]))
                        cached_embeddings[idx] = new_slices[i]

                mm_embedding = torch.cat(
                    [e for e in cached_embeddings if e is not None], dim=0
                )

                # Build aux_data and grid_dim
                aux_data = {}
                if modality == Modality.IMAGE and full_grid_thw is not None:
                    aux_data["image_grid_thw"] = full_grid_thw
                    grid_dim = full_grid_thw
                elif modality == Modality.AUDIO:
                    # Reconstruct audio_feature_lens_raw from all slice shapes
                    audio_feature_lens = torch.tensor(
                        [e.shape[0] for e in cached_embeddings if e is not None],
                        dtype=torch.long,
                    )
                    grid_dim = audio_feature_lens
                else:
                    grid_dim = full_grid_thw

                if self.profiler is not None:
                    self.profiler.step()

                t_end = _time.perf_counter()
                logger.info(
                    f"_encode PARTIAL ({modality.name}): "
                    f"hash={( t_cache - t0)*1000:.2f}ms, "
                    f"proc={( t_proc - t_cache)*1000:.2f}ms, "
                    f"get_feature={( t_vit - t_proc)*1000:.2f}ms, "
                    f"cache_store={( t_end - t_vit)*1000:.2f}ms, "
                    f"total={( t_end - t0)*1000:.2f}ms "
                    f"({hit_count} hits, {len(missing_indices)} misses)"
                )
                return (grid_dim, mm_embedding, aux_data)

            except BadRequestError as e:
                raise BadRequestError(f"Bad request error: {str(e)}")
            except Exception as e:
                raise InternalError(f"Internal encoding error: {str(e)}")

        # ── Original path: no prefix_mm_cache or non-beebee model ──
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        try:
            mm_embedding = None
            mm_hash = None

            if self.model_type == "llavaqwen2_omni":
                if modality == Modality.IMAGE:
                    pixel_values = mm_inputs["pixel_values"]
                    feature = torch.cat(pixel_values, dim=0)
                    mm_item = MultimodalDataItem.from_dict(
                        {"modality": modality, "feature": feature}
                    )
                    mm_item.model_specific_data["image_grid_thw"] = mm_inputs[
                        "image_grid_thw"
                    ]
                elif modality == Modality.AUDIO:
                    mm_item = MultimodalDataItem.from_dict(
                        {
                            "modality": modality,
                            "feature": mm_inputs["input_features"],
                        }
                    )
                    mm_item.model_specific_data["audio_length"] = mm_inputs[
                        "audio_length"
                    ]
                else:
                    raise ValueError(
                        f"BeeBeeOmni does not support {modality} in encoder-only mode"
                    )
            else:
                mm_item = MultimodalDataItem.from_dict(
                    {
                        "modality": modality,
                        "feature": _convert(
                            _get_mm_feature(mm_inputs, modality)
                        ),
                    }
                )
                for k, v in mm_inputs.items():
                    if k in _mm_feature_attrs[modality]:
                        continue
                    mm_item.set(k, _convert(v))

            if self.server_args.enable_prefix_mm_cache:
                mm_item.set_pad_value()
                mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get([mm_item.hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding

            if mm_embedding is None:
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = get_feature_fn([mm_item])
                    mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

            if self.server_args.enable_prefix_mm_cache:
                async with self.mm_cache_lock:
                    self.mm_cache.set(mm_hash, EmbeddingResult(embedding=mm_embedding))
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs)
            return (
                _get_mm_grid_dim(mm_inputs, modality, self.model_type),
                mm_embedding,
                aux_data,
            )
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
    ):
        import time as _time
        t_send_start = _time.perf_counter()

        if self.server_args.encoder_transfer_backend == "mooncake":
            logger.info(
                f"[SEND TIMING] req_id={mm_data.req_id} mooncake RDMA transfer start: "
                f"embedding device={'cuda' if embedding.is_cuda else 'cpu'}, "
                f"is_pinned={embedding.is_pinned()}, "
                f"dtype={embedding.dtype}, shape={list(embedding.shape)}, "
                f"nbytes={embedding.nbytes / 1024 / 1024:.2f}MB"
            )
            t_register = _time.perf_counter()
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            t_register_done = _time.perf_counter()
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), buffer_address, embedding.nbytes
            )
            t_transfer_done = _time.perf_counter()
            self.engine.deregister(embedding.data_ptr())
            t_deregister_done = _time.perf_counter()
            logger.info(
                f"[SEND TIMING] req_id={mm_data.req_id} mooncake RDMA transfer done: "
                f"register={( t_register_done - t_register)*1000:.2f}ms, "
                f"transfer_sync={( t_transfer_done - t_register_done)*1000:.2f}ms, "
                f"deregister={( t_deregister_done - t_transfer_done)*1000:.2f}ms, "
                f"total_rdma={( t_deregister_done - t_register)*1000:.2f}ms"
            )

            mm_data.embedding = None

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        t_serialize = _time.perf_counter()
        if self.server_args.encoder_transfer_backend == "mooncake":
            serialized_data = pickle.dumps(mm_data)
            buffer = None
        else:
            new_mm_data = mm_data.copy_without_embedding()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()
        t_serialize_done = _time.perf_counter()

        # Use thread pool executor for parallel ZMQ send operations
        def send_with_socket():
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    sock.send_multipart([serialized_data, buffer], copy=False)
                else:
                    sock.send_multipart([serialized_data], copy=False)
            finally:
                sock.close()

        t_zmq_start = _time.perf_counter()
        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)
        t_zmq_done = _time.perf_counter()

        logger.info(
            f"[SEND TIMING] req_id={mm_data.req_id} _send total: "
            f"serialize={( t_serialize_done - t_serialize)*1000:.2f}ms, "
            f"zmq_send={( t_zmq_done - t_zmq_start)*1000:.2f}ms, "
            f"total={( t_zmq_done - t_send_start)*1000:.2f}ms"
        )

    async def encode(self, mm_items, modality: Modality, req_id, num_parts, part_idx):
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(mm_items, modality)

            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_dim,
                    modality,
                    mm_embedding,
                    **aux_data,
                )
                self.embedding_to_send[req_id] = mm_data
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(f"Rank {self.rank} encode failed: {error_msg} {error_code = }")
            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    None,
                    modality,
                    error_msg=error_msg,
                    error_code=error_code,
                )
                self.embedding_to_send[req_id] = mm_data
                logger.debug(f"Created error EmbeddingData: {mm_data}")
            return 0, 0, 0, error_msg, error_code

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self._send(
            mm_data.embedding,
            mm_data,
            session_id=session_id,
            buffer_address=buffer_address,
            prefill_host=prefill_host,
            embedding_port=embedding_port,
        )

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        mm_data = self.embedding_to_send.get(req_id)
        if not mm_data:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await get_condition(req_id)

        try:
            while True:
                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        task = asyncio.create_task(
                            self._send(
                                mm_data.embedding,
                                mm_data,
                                url=url,
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            logger.info(f"Cleaning up resources for req_id {req_id}")
            async with rid_lock:
                rid_to_receive_endpoint.pop(req_id, None)
                rid_to_receive_count.pop(req_id, None)
            async with cond_dict_lock:
                rid_to_cond.pop(req_id, None)
            self.embedding_to_send.pop(req_id, None)

    async def get_embedding_port(self, prefill_url):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            response = await session.post(
                f"{prefill_url}/embedding_bootstrap",
                json={"embedding_port": None},
            )
            response_json = await response.json()
            return response_json["embedding_port"]


class EncoderProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


app = FastAPI()
encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        else:
            if encoder.mm_global_cache is not None:
                print("Using Global Cache!!!!!!!!!!!")
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            else:
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    global encoder
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


async def get_condition(rid):
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_id = request["req_id"]
    try:

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request
        request.update({"enter_time": time.time()})
        for socket in send_sockets:
            socket.send_pyobj(request)
        if encoder.mm_global_cache is not None:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            )
        else:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )
            )

        if error_msg:
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Performs a dummy encode to verify the encoder is functional.
    Returns 200 if the encoder is healthy, 503 otherwise.
    """
    if encoder is None:
        return Response(status_code=503)

    # Skip the dummy encode when real requests are already in flight — the
    # ongoing traffic already proves liveness, matching the scheduler's
    # `is_fully_idle`-based health-check skip pattern.
    if encoder.embedding_to_send:
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    if encoder.image_processor is not None:
        if encoder.model_type == "llavaqwen2_omni":
            # BeeBeeOmni requires paired 448x448 images
            mm_items = [
                _make_dummy_png_data_uri(448, 448),
                _make_dummy_png_data_uri(448, 448),
            ]
        else:
            mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor available, fall back to liveness check only
        return Response(status_code=200)

    try:
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        for socket in send_sockets:
            socket.send_pyobj(dummy_request)

        # Run encode on rank 0 with timeout
        _, _, _, error_msg, _ = await asyncio.wait_for(
            encoder.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )

        # Clean up stored embedding
        encoder.embedding_to_send.pop(req_id, None)

        if error_msg:
            logger.error(f"Encoder health check failed: {error_msg}")
            return Response(status_code=503)

        return Response(status_code=200)

    except asyncio.TimeoutError:
        logger.error(f"Encoder health check timed out after {HEALTH_CHECK_TIMEOUT}s")
        return Response(status_code=503)
    except Exception as e:
        logger.error(f"Encoder health check failed: {e}")
        return Response(status_code=503)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
