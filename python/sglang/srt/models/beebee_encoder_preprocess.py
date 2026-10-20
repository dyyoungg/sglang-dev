"""BeeBeeOmni encoder-side CPU preprocessing.

The EPD encoder owns media loading and invokes model hooks for model-specific
representations.  Keeping this code here lets the generic encoder pipeline
remain independent of the BeeBee pair/chunk conventions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput


def _image_output(model, images, image_processor):
    if image_processor is None:
        raise ValueError("BeeBee encoder preprocessing requires an image processor")
    pairs = []
    for index in range(0, len(images), 2):
        pair = list(images[index : index + 2])
        if len(pair) == 1:
            pair.append(pair[0])
        first = pair[0]
        shape = getattr(first, "shape", None)
        # PIL exposes ``size`` as (width, height), while numpy/tensor inputs
        # expose ``size`` as a scalar/method.  Only use the PIL-style tuple.
        size_attr = getattr(first, "size", None)
        if isinstance(size_attr, tuple) and len(size_attr) == 2:
            size = size_attr
        elif shape is not None and len(shape) >= 2:
            size = (int(shape[1]), int(shape[0]))
        else:
            size = (0, 0)
        pairs.append((index // 2, pair, size))

    buckets = defaultdict(list)
    for pair in pairs:
        buckets[pair[2]].append(pair)
    items = []
    for _, bucket in buckets.items():
        batch = [image for _, pair, _ in bucket for image in pair]
        kwargs = {"return_tensors": "pt"}
        if hasattr(image_processor, "preprocess"):
            kwargs["patch_reshape_method"] = "torch"
            output = image_processor.preprocess(batch, **kwargs)
        else:
            output = image_processor(images=batch, **kwargs)
        pixels = output["pixel_values"]
        grid = output.get("grid_thw")
        if grid is None:
            grid = output.get("image_grid_thw")
        if grid is None:
            raise ValueError("BeeBee image processor did not return grid_thw")
        grid_h, grid_w = int(grid[0, 1]), int(grid[0, 2])
        stride = grid_h * grid_w
        for offset, (pair_index, _, _) in enumerate(bucket):
            feature = pixels[offset * stride : (offset + 1) * stride].clone()
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                model_specific_data={
                    "image_grid_thw": torch.tensor(
                        [[1, grid_h, grid_w]], dtype=torch.long
                    ),
                },
            )
            items.append((pair_index, item))

    items.sort(key=lambda value: value[0])
    ordered = [item for _, item in items]
    grids = [item.image_grid_thw for item in ordered]
    return EncoderPreprocessOutput(
        {
            "pixel_values": [item.feature for item in ordered],
            "image_grid_thw": torch.cat(grids),
        },
        mm_items=ordered,
    )


def _audio_one(audio):
    try:
        import torchaudio.functional as audio_f
    except ImportError:
        audio_f = None
    from whisper.audio import log_mel_spectrogram

    sample_rate = 16000
    max_length = 480000
    min_length = 4000
    if isinstance(audio, dict):
        waveform = audio["array"]
        sample_rate = audio.get("sampling_rate", sample_rate)
    else:
        waveform = audio
    waveform = torch.as_tensor(waveform, dtype=torch.float32).flatten()
    if sample_rate != 16000:
        if audio_f is not None:
            waveform = audio_f.resample(waveform, sample_rate, 16000)
        else:
            import librosa

            waveform = torch.from_numpy(
                librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=16000)
            )
    length = max(int(waveform.numel()), min_length)
    waveform = F.pad(waveform, (0, length - waveform.numel()))
    chunks = (length + max_length - 1) // max_length
    padded = F.pad(waveform, (0, chunks * max_length - length))
    chunk_lengths = [
        max(min(length - i * max_length, max_length), min_length) for i in range(chunks)
    ]
    return (
        log_mel_spectrogram(padded.view(chunks, max_length), n_mels=128),
        chunk_lengths,
    )


def _audio_output(model, audios):
    items = []
    for audio in audios:
        mel, lengths = _audio_one(audio)
        audio_model_type = getattr(
            getattr(model, "audio_config", None), "model_type", ""
        )
        if audio_model_type == "beebee_qwen3_audio_model":
            from sglang.srt.models.beebee_audio_encoders import (
                _get_feat_extract_output_lengths,
            )

            ratio = int(getattr(model.audio_config, "audio_downsample_ratio", 2))
            n_window = int(getattr(model.audio_config, "n_window", 50))
            token_count = 0
            for length in lengths:
                mel_frames = max(1, int(length) // 160)
                encoded = int(
                    _get_feat_extract_output_lengths(
                        torch.tensor(mel_frames), n_window
                    ).item()
                )
                token_count += (encoded + ratio - 1) // ratio
        else:
            token_count = sum((length // 320 + 9) // 10 for length in lengths)
        items.append(
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=mel,
                model_specific_data={"audio_length": lengths},
            )
        )
        items[-1].set(
            "audio_feature_lens_raw", torch.tensor(token_count, dtype=torch.long)
        )
    return EncoderPreprocessOutput(
        {
            "input_features": [item.feature for item in items],
            "audio_feature_lens_raw": torch.tensor(
                [int(item.audio_feature_lens_raw) for item in items], dtype=torch.long
            ),
        },
        mm_items=items,
    )


def preprocess_beebee_mm_for_encoder(
    model,
    mm_data: Any,
    modality: Modality,
    config: dict,
    *,
    image_processor=None,
    **kwargs,
):
    if modality == Modality.IMAGE:
        return _image_output(model, mm_data, image_processor)
    if modality == Modality.AUDIO:
        return _audio_output(model, mm_data)
    raise NotImplementedError(
        f"BeeBee encoder preprocessing does not support {modality}"
    )
