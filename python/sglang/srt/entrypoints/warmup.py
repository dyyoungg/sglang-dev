from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List


import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("beebee_omni_warmup")
async def beebee_omni_warmup(disaggregation_mode: str, tokenizer_manager):
    import io
    from PIL import Image
    import base64
    import wave
    logger.info("🚀 Start BeeBeeOmni warmup")

    if disaggregation_mode == "decode":
        logger.info("当前为 Decode 节点，跳过多模态预热。")
        return

    dummy_img = Image.new("RGB", (448, 448), (0, 0, 0))
    img_buffer = io.BytesIO()
    dummy_img.save(img_buffer, format="JPEG")
    img_b64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    dummy_image_input1 = f"data:image/jpeg;base64,{img_b64_str}"

    dummy_img = Image.new("RGB", (1120, 672), (0, 0, 0))
    img_buffer = io.BytesIO()
    dummy_img.save(img_buffer, format="JPEG")
    img_b64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    dummy_image_input2 = f"data:image/jpeg;base64,{img_b64_str}"    

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)                # 单声道
        wav_file.setsampwidth(2)                # 16-bit (2 bytes)
        wav_file.setframerate(16000)            # 16kHz
        wav_file.writeframes(b'\x00\x00' * 16000)
    
    audio_b64_str = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
    dummy_audio_input = f"data:audio/wav;base64,{audio_b64_str}"
    try:
        dp_size = tokenizer_manager.server_args.dp_size

        async def _run_one_warmup():
            req = GenerateReqInput(
                text="<image>"*16 + "<audio>"*8 + "请描述这段画面。",
                image_data=[dummy_image_input1]*8 + [dummy_image_input2] * 8,
                audio_data=[dummy_audio_input]*8,
                sampling_params={"max_new_tokens": 1}
            )
            await tokenizer_manager.generate_request(req, None).__anext__()

        import asyncio
        await asyncio.gather(*[_run_one_warmup() for _ in range(dp_size)])

        logger.info("✅ warmup successfully!!")
    except Exception as e:
        logger.error(f"❌ warmup failed! error:{e}")