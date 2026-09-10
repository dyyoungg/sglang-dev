import time
import os
from typing import Dict, List, Optional, Union, Iterable
import PIL
import numpy as np
from math import ceil
import numexpr as ne
from functools import lru_cache

import torch
from transformers import CLIPImageProcessor
from transformers.image_utils import (
    PILImageResampling,
    ChannelDimension,
    ImageInput,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb

_numexpr_override = os.environ.get("SGLANG_NUMEXPR_NUM_THREADS")
if _numexpr_override is not None:
    ne.set_num_threads(int(_numexpr_override))
else:
    ne.set_num_threads(min(os.cpu_count() // 4, 16))  # 动态计算线程数

_IMG_PREPROCESS_BACKEND = os.environ.get("SGLANG_IMG_PREPROCESS_BACKEND", "numexpr").lower()
_IMG_TORCH_THREADS = os.environ.get("SGLANG_IMG_TORCH_THREADS")
print(f"Image preprocess backend: {_IMG_PREPROCESS_BACKEND}!! torch thread: {_IMG_TORCH_THREADS}")

def batch_center_crop(
    image: np.ndarray,
    size: dict,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    return_numpy = True if return_numpy is None else return_numpy

    #
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input images must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    # Determine if input has batch dimension (4D tensor)
    is_batched = len(image.shape) == 4

    if not is_batched:
        image = image[np.newaxis]

    batch_size = image.shape[0]

    input_data_format = ChannelDimension.FIRST if image.shape[1] in [1, 3, 4] else ChannelDimension.LAST

    if input_data_format == ChannelDimension.LAST:
        # Convert from NHWC to NCHW
        image = np.transpose(image, (0, 3, 1, 2))

    _, _, orig_height, orig_width = image.shape

    crop_height, crop_width = size["height"], size["width"]

    crop_height, crop_width = int(crop_height), int(crop_width)
    # Calculate crop coordinates
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        cropped_images = image[:, :, top:bottom, left:right]
    else:
        # Need to pad
        new_height = max(crop_height, orig_height)
        new_width = max(crop_width, orig_width)
        new_images = np.zeros((batch_size, image.shape[1], new_height, new_width), dtype=image.dtype)

        # If the image is too small, pad it with zeros
        top_pad = ceil((new_height - orig_height) / 2)
        bottom_pad = top_pad + orig_height
        left_pad = ceil((new_width - orig_width) / 2)
        right_pad = left_pad + orig_width

        # Pad all images in the batch
        new_images[:, :, top_pad:bottom_pad, left_pad:right_pad] = image

        # Adjust crop coordinates
        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        cropped_images = new_images[
            :,
            :,
            max(0, top) : min(new_height, bottom),
            max(0, left) : min(new_width, right),
        ]

    return cropped_images


def batch_normalize_numexpr(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    **kwargs,
) -> np.ndarray:
    # image = np.asarray(image, dtype=np.float32) # 如果这里开启,精度无损, 增加30%耗时
    # mean = np.asarray(mean, dtype=np.float32)
    # std = np.asarray(std, dtype=np.float32)
    is_batched = len(image.shape) == 4
    if not is_batched:
        image = image[np.newaxis]

    if input_data_format == ChannelDimension.LAST:  # ensure
        mean = mean.reshape(1, 1, 1, -1)
        std = std.reshape(1, 1, 1, -1)

    # 使用 numexpr 进行优化计算
    out = np.empty(image.shape, dtype=np.float32)
    ne.evaluate("(image - mean) / std", local_dict={"image": image, "mean": mean, "std": std}, out=out)

    return out


def batch_rescale_numexpr(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")
    # image = image.astype(np.float64)
    rescaled_image = ne.evaluate("image * scale", local_dict={"image": image, "scale": scale})
    return rescaled_image  # float64精度


class OpimizedCLIPImageProcessor(CLIPImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        init_kwargs = dict(
            do_resize=do_resize,
            resample=resample,
            do_center_crop=do_center_crop,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        if size is not None:
            init_kwargs["size"] = size
        if crop_size is not None:
            init_kwargs["crop_size"] = crop_size
        super().__init__(**init_kwargs)

        if not hasattr(self, "_valid_processor_keys"):
            vk = getattr(self, "valid_kwargs", None)
            if vk is not None and hasattr(vk, "__annotations__"):
                self._valid_processor_keys = list(vk.__annotations__.keys())
            elif isinstance(vk, (set, list, tuple)):
                self._valid_processor_keys = list(vk)
            else:
                self._valid_processor_keys = []

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> tuple:

        image_mean = np.asarray(image_mean, dtype=np.float32)
        image_std = np.asarray(image_std, dtype=np.float32)
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = image_mean * (1.0 / rescale_factor)
            image_std = image_std * (1.0 / rescale_factor)
            if input_data_format == ChannelDimension.LAST:
                image_mean = image_mean.reshape(1, 1, 1, -1)
                image_std = image_std.reshape(1, 1, 1, -1)
            else:
                image_mean = image_mean.reshape(1, -1, 1, 1)
                image_std = image_std.reshape(1, -1, 1, 1)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample=None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self._valid_processor_keys,
        )

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        all_images = []
        # for image in images: # remove resize operation, must ensure correct image size
        #     if do_resize:
        #         image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        #         all_images.append(image)

        images = all_images if len(all_images) else images

        images = np.stack(images)
        image_mean = tuple(image_mean)
        image_std = tuple(image_std)

        input_data_format = ChannelDimension.FIRST if images.shape[1] in [1, 3, 4] else ChannelDimension.LAST

        if input_data_format == ChannelDimension.LAST:
            images = np.transpose(images, (0, 3, 1, 2))
            # images = np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))
            input_data_format = ChannelDimension.FIRST

        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            input_data_format=input_data_format,
        )

        if do_center_crop:
            image = batch_center_crop(image=images, size=crop_size, input_data_format=input_data_format)

        if do_rescale:  # will skip because of fused rescale
            image = batch_rescale_numexpr(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = batch_normalize_numexpr(
                image=image,
                mean=image_mean,
                std=image_std,
                input_data_format=input_data_format,
            )

        data = {"pixel_values": image}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _get_torch_fused_scale_bias(
        self,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
    ):
        """融合 rescale+normalize 为单次 `t * scale - bias`(原地)。

        out = (image * rescale_factor - mean) / std
            = image * (rescale_factor / std) - (mean / std)
        缓存小 tensor,避免每次重建。
        """
        key = (
            bool(do_rescale),
            bool(do_normalize),
            tuple(float(x) for x in image_mean),
            tuple(float(x) for x in image_std),
            float(rescale_factor),
        )
        cached = self._torch_fused_cache.get(key)
        if cached is not None:
            return cached

        mean = torch.tensor(image_mean, dtype=torch.float32)
        std = torch.tensor(image_std, dtype=torch.float32)
        if do_rescale and do_normalize:
            scale = (float(rescale_factor) / std).view(1, -1, 1, 1)
            bias = (mean / std).view(1, -1, 1, 1)
        elif do_normalize:
            scale = (1.0 / std).view(1, -1, 1, 1)
            bias = (mean / std).view(1, -1, 1, 1)
        elif do_rescale:
            scale = torch.tensor(float(rescale_factor), dtype=torch.float32)
            bias = None
        else:
            scale = None
            bias = None
        self._torch_fused_cache[key] = (scale, bias)
        return scale, bias

    def _preprocess_torch_fused(
        self,
        images,
        do_center_crop,
        crop_size,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        return_tensors,
        patch_reshape_method,
    ):
        """numpy stack + torch(cast+normalize fused) + torch patch。

        与 numexpr 路径等价, 但 normalize 用 torch 替代 numexpr:
        - numexpr: ne.evaluate 读 strided uint8 → 输出连续 float32 (~18ms)
        - torch:   from_numpy(strided).float() → 连续 float32, 再原地 mul/sub (~12ms)
        patch extraction 用 torch (输入已连续, fast_patch_extraction_torch)。
        """
        import time as _time
        t_start = _time.perf_counter()

        # ── numpy stack: [N, H, W, C] uint8, 连续 ──
        arr = np.stack(images)
        t_stack = _time.perf_counter()

        # ── transpose 为 NCHW (仅 view, 非连续, 0ms) ──
        if arr.shape[-1] in (1, 3, 4):
            H, W = int(arr.shape[1]), int(arr.shape[2])
            arr = np.transpose(arr, (0, 3, 1, 2))
        else:
            H, W = int(arr.shape[2]), int(arr.shape[3])

        # ── cast: 读非连续 uint8 → 写连续 NCHW float32 (一次 kernel) ──
        # from_numpy 零拷贝 view strided uint8, .float() 产出连续 float32
        t = torch.from_numpy(arr).float()
        t_float = _time.perf_counter()

        # ── 融合 rescale + normalize (原地, 连续内存上跑, 快) ──
        scale, bias = self._get_torch_fused_scale_bias(
            do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        if scale is not None:
            t.mul_(scale)
            if bias is not None:
                t.sub_(bias)
        t_norm = _time.perf_counter()

        # ── temporal 偶数对齐 ──
        if t.shape[0] % 2 == 1:
            t = torch.cat([t, t[-1:]], dim=0)

        # ── patch extraction (输入是连续 torch tensor) ──
        flatten_patches = fast_patch_extraction_torch(
            t, self.temporal_patch_size, self.patch_size, self.merge_size
        )

        # ── 转 float16: 模型端第一步就 .to(dtype) 转半精度, 提前转可以:
        #    1. shm 传输量减半 (270MB → 135MB, wrap_shm 从 ~41ms → ~20ms)
        #    2. 模型端少一次 dtype cast
        flatten_patches = flatten_patches.to(torch.float16)
        t_patch = _time.perf_counter()

        grid_t = t.shape[0] // self.temporal_patch_size
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

        print(
            f"[IMAGE PROCESSOR TIMING][torch] num_images={t.shape[0]}, "
            f"shape={tuple(t.shape)}, "
            f"stack={(t_stack - t_start)*1000:.2f}ms, "
            f"float={(t_float - t_stack)*1000:.2f}ms, "
            f"normalize={(t_norm - t_float)*1000:.2f}ms, "
            f"patch+f16={(t_patch - t_norm)*1000:.2f}ms, "
            f"TOTAL={(t_patch - t_start)*1000:.2f}ms"
        )

        data = {"pixel_values": flatten_patches, "grid_thw": grid_thw}
        return BatchFeature(data=data, tensor_type=return_tensors)


def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def fast_patch_extraction_torch(patches, temporal_patch_size, patch_size, merge_size):
    """使用PyTorch优化的patch提取"""
    is_numpy = isinstance(patches, np.ndarray)
    if is_numpy:
        patches_tensor = torch.from_numpy(patches)
    else:
        patches_tensor = patches

    N, C, H, W = patches_tensor.shape
    grid_t = N // temporal_patch_size
    grid_h, grid_w = H // patch_size, W // patch_size
    final_features = C * temporal_patch_size * patch_size * patch_size

    patches_reshaped = patches_tensor.view(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches_transposed = patches_reshaped.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches_transposed.contiguous().view(grid_t * grid_h * grid_w, final_features)
    return flatten_patches
    # return flatten_patches.numpy() if is_numpy else flatten_patches


def patch_extraction_numpy(patches, temporal_patch_size, patch_size, merge_size):
    N, C, H, W = patches.shape
    grid_t = N // temporal_patch_size
    grid_h, grid_w = H // patch_size, W // patch_size
    # patches = np.ascontiguousarray(patches)
    # 直接使用原始的reshape和transpose逻辑
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    final_features = C * temporal_patch_size * patch_size * patch_size
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, final_features)
    return flatten_patches


class Qwen25VLImageProcessorOptimized(OpimizedCLIPImageProcessor):
    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=True,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        min_pixels: int = 100 * 28 * 28,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ):
        init_kwargs = dict(
            do_resize=do_resize,
            resample=resample,
            do_center_crop=do_center_crop,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        if size is not None:
            init_kwargs["size"] = size
        if crop_size is not None:
            init_kwargs["crop_size"] = crop_size
        super().__init__(**init_kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}

        # torch-fused preprocess backend (env-gated, default off → original numexpr path)
        self._preprocess_backend = _IMG_PREPROCESS_BACKEND
        self._torch_fused_cache: dict = {}
        if self._preprocess_backend == "torch" and _IMG_TORCH_THREADS is not None:
            try:
                torch.set_num_threads(int(_IMG_TORCH_THREADS))
            except Exception:
                pass

    def preprocess(
        self,
        images,
        do_resize=None,
        size=None,
        resample=None,
        do_center_crop=None,
        crop_size=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        do_convert_rgb=None,
        return_tensors=None,
        data_format=ChannelDimension.FIRST,
        input_data_format=None,
        patch_reshape_method="numpy",
        **kwargs,
    ):
        import time as _time
        t_start = _time.perf_counter()

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self._valid_processor_keys,
        )

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # ── torch-fused fast path (env-gated, numpy uint8 input only) ──
        # 失败/不支持时返回 None,自动回退到下方原始 numexpr 路径,原逻辑完全不动。

        if (
            self._preprocess_backend == "torch"
            and images
            and all(isinstance(im, np.ndarray) for im in images)
        ):
            torch_ret = self._preprocess_torch_fused(
                images=images,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                return_tensors=return_tensors,
                patch_reshape_method=patch_reshape_method,
            )
            if torch_ret is not None:
                return torch_ret

        t_validate = _time.perf_counter()

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        t_convert_rgb = _time.perf_counter()

        images = [to_numpy_array(image) for image in images]

        t_to_numpy = _time.perf_counter()

        all_images = []
        # for image in images: # remove resize operation, must ensure correct image size
        #     if do_resize:
        #         image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        #         all_images.append(image)

        images = all_images if len(all_images) else images

        images = np.stack(images)
        t_stack = _time.perf_counter()

        image_mean = tuple(image_mean)
        image_std = tuple(image_std)

        input_data_format = ChannelDimension.FIRST if images.shape[1] in [1, 3, 4] else ChannelDimension.LAST

        if input_data_format == ChannelDimension.LAST:
            images = np.transpose(images, (0, 3, 1, 2))
            input_data_format = ChannelDimension.FIRST

        t_transpose = _time.perf_counter()

        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            input_data_format=input_data_format,
        )

        if do_rescale:  # will skip because of fused rescale
            images = batch_rescale_numexpr(image=images, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            images = batch_normalize_numexpr(
                image=images,
                mean=image_mean,
                std=image_std,
                input_data_format=input_data_format,
            )

        t_normalize = _time.perf_counter()

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        patches = images
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)  # [N, channel, h, w]

        if patches.shape[0] % 2 == 1:
            last_frame = patches[-1:]
            patches = np.concatenate([patches, last_frame], axis=0)

        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        if patch_reshape_method == "torch":
            flatten_patches = fast_patch_extraction_torch(
                patches, self.temporal_patch_size, self.patch_size, self.merge_size
            )
            # 转 float16 减少 shm 传输量 (模型端第一步就会转 dtype)
            flatten_patches = flatten_patches.to(torch.float16)
        else:
            flatten_patches = patch_extraction_numpy(
                patches, self.temporal_patch_size, self.patch_size, self.merge_size
            )

        t_patch = _time.perf_counter()

        pixel_values, vision_grid_thws = [], []
        pixel_values = flatten_patches

        # vision_grid_thws.append((grid_t, grid_h, grid_w))
        vision_grid_thws = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

        data = {"pixel_values": pixel_values, "grid_thw": vision_grid_thws}

        t_end = _time.perf_counter()
        print(
            f"[IMAGE PROCESSOR TIMING][numexpr] num_images={len(images) if isinstance(images, list) else images.shape[0]}, "
            f"shape={images.shape if hasattr(images, 'shape') else 'N/A'}, "
            f"validate={( t_validate - t_start)*1000:.2f}ms, "
            f"convert_rgb={( t_convert_rgb - t_validate)*1000:.2f}ms, "
            f"to_numpy={( t_to_numpy - t_convert_rgb)*1000:.2f}ms, "
            f"np_stack={( t_stack - t_to_numpy)*1000:.2f}ms, "
            f"transpose={( t_transpose - t_stack)*1000:.2f}ms, "
            f"normalize={( t_normalize - t_transpose)*1000:.2f}ms, "
            f"patch_extract({patch_reshape_method})={( t_patch - t_normalize)*1000:.2f}ms, "
            f"TOTAL={( t_end - t_start)*1000:.2f}ms"
        )

        return BatchFeature(data=data, tensor_type=return_tensors)


def create_random_images(batch_size: int, height: int, width: int) -> List[PIL.Image.Image]:
    images = []
    for _ in range(batch_size):
        # 创建随机 RGB 图片
        random_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # pil_image = PIL.Image.fromarray(random_image)
        images.append(random_image)
    return images


def run_performance_test(processor, batch_size, height, width, num_runs: int = 5, **kwargs) -> float:

    total_time = 0

    for _ in range(num_runs):
        images = create_random_images(batch_size, height, width)
        start_time = time.perf_counter()
        _ = processor.preprocess(images, **kwargs)
        end_time = time.perf_counter()
        total_time += (end_time - start_time) * 1000  # 转换为毫秒

    return total_time / num_runs


def compare_outputs(original_output, optimized_output, tolerance=1e-6) -> bool:
    if not isinstance(original_output, type(optimized_output)):
        print(f"输出类型不匹配: 原始输出 {type(original_output)} vs 优化输出 {type(optimized_output)}")
        return False

    # 获取像素值数组
    original_pixels = original_output["pixel_values"].numpy()
    optimized_pixels = optimized_output["pixel_values"].numpy()

    if original_pixels.shape != optimized_pixels.shape:
        print(f"输出形状不匹配: 原始输出 {original_pixels.shape} vs 优化输出 {optimized_pixels.shape}")
        return False

    # 计算最大误差
    # is_equal = np.array_equal(original_pixels, optimized_pixels)
    max_diff = np.max(np.abs(original_pixels - optimized_pixels))
    is_close = max_diff <= tolerance

    if not is_close:
        print(f"输出值不匹配: 最大误差 = {max_diff:.6f}, 超过容许误差 {tolerance}")
    # print("is equal", is_equal, "max_diff", max_diff)
    return max_diff


def compare_processors():
    """比较两种图片处理器的性能和输出结果"""

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    image_sizes = [(364, 644)]

    vision_model_path = "/mnt/afs/zhouhang/model/clip-vit-large-patch14-644x364"

    original_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
    optimized_processor = OpimizedCLIPImageProcessor.from_pretrained(vision_model_path)
    encoder_path = "/mnt/afs/share/qwen25_vl_encoder/"

    qwen25_processor = Qwen25VLImageProcessorOptimized.from_pretrained(encoder_path)

    print("\n性能测试报告:")
    print("=" * 180)
    print(
        f"{'批次大小':^10} | {'图片尺寸':^12} | {'CLIP 原始(ms)':^12} | {'CLIP 优化后(ms)':^12} | {'Qwen25 Processor numpy(ms)':^12} | {'Qwen25 Processor torch(ms)':^12} | {'性能提升':^12} | {'最大误差':^10} | {'单张处理时间':^10}"
    )
    print("-" * 180)

    for batch_size in batch_sizes:
        for height, width in image_sizes:

            test_images = create_random_images(batch_size, height, width)

            original_output = original_processor.preprocess(test_images, return_tensors="pt")
            optimized_output = optimized_processor.preprocess(test_images, return_tensors="pt")
            _ = qwen25_processor.preprocess(test_images, return_tensors="pt", patch_reshape_method="torch")
            _ = qwen25_processor.preprocess(test_images, return_tensors="pt", patch_reshape_method="numpy")

            max_diff = compare_outputs(original_output, optimized_output)

            optimized_time = run_performance_test(optimized_processor, batch_size, height, width, num_runs=10)
            original_time = run_performance_test(original_processor, batch_size, height, width, num_runs=10)
            qwen25_time_torch = run_performance_test(
                qwen25_processor,
                batch_size,
                height,
                width,
                num_runs=10,
                patch_reshape_method="torch",
            )
            qwen25_time_numpy = run_performance_test(
                qwen25_processor,
                batch_size,
                height,
                width,
                num_runs=10,
                patch_reshape_method="numpy",
            )

            # optimized_time = run_performance_test(optimized_processor, test_images, num_runs=10)

            improvement = original_time / optimized_time - 1

            avg_time = qwen25_time_torch / batch_size

            # 输出结果
            print(
                f"{batch_size:^14} | {f'{height}x{width}':^16} | "
                f"{original_time:^14.2f} | {optimized_time:^15.2f} | {qwen25_time_numpy:^26.2f} | {qwen25_time_torch:^26.2f} |"
                f"{improvement:^16.1f}x | {max_diff:^14.8f} |{avg_time:^15.8f} "
            )

    print("=" * 80)


if __name__ == "__main__":

    np.random.seed(42)
    compare_processors()

    # encoder_path = "/mnt/afs/share/qwen25_vl_encoder/"

    # processor = Qwen25VLImageProcessorOptimized.from_pretrained(encoder_path)

    # test_images = create_random_images(batch_size=3, height=364, width=644)

    # results=processor.preprocess(test_images, return_tensors="pt")
    # print(results["pixel_values"].shape, results["grid_thw"])
