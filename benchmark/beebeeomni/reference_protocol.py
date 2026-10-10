"""Prompt and scoring protocol copied from VeOmni/scripts/eval_sglang.py.
Source SHA256: 0008ae5d603ef745cbbc1035f8be7b2eed6f72ad80329595bb28d08161d1c64a
Keep this snapshot so paired precision experiments use a fixed protocol.
"""

import re
import os
from typing import Tuple, Union
import jiwer

from PIL import Image

DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"

DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".gif",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".svg"}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".pcm",
}


def _mixed_language_chars(text: list[str]) -> list[list[str]]:
    return [re.findall(r"[a-zA-Z0-9]+|[一-鿿]", s) for s in text]


def calculate_cer(references: list[str], hypotheses: list[str]) -> list[float]:
    return [
        jiwer.cer(
            ref,
            hyp,
            reference_transform=_mixed_language_chars,
            hypothesis_transform=_mixed_language_chars,
        )
        for ref, hyp in zip(references, hypotheses)
    ]


def calculate_wer(references: list[str], hypotheses: list[str]) -> list[float]:
    from jiwer import transforms as tr

    wer_transform = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemoveMultipleSpaces(),
            tr.RemovePunctuation(),
            tr.Strip(),
            tr.ExpandCommonEnglishContractions(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    return [
        jiwer.wer(
            ref,
            hyp,
            reference_transform=wer_transform,
            hypothesis_transform=wer_transform,
        )
        for ref, hyp in zip(references, hypotheses)
    ]


def calculate_edit_distance(
    references: list[str], hypotheses: list[str]
) -> list[float]:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if not ref and not hyp:
            scores.append(0.0)
            continue
        m, n = len(ref), len(hyp)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                if ref[i - 1] == hyp[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        norm = max(m, n, 1)
        scores.append(dp[n] / norm)
    return scores


def calculate_accuracy(references: list[str], hypotheses: list[str]) -> list[float]:
    return [
        1.0 if r.strip() == h.strip() else 0.0 for r, h in zip(references, hypotheses)
    ]


def detect_media_type(sample: dict) -> Tuple[str, Union[str, list]]:
    if "audio" in sample or "audio_path" in sample:
        path = sample.get("audio") or sample.get("audio_path", "")
        if path:
            return "audio", path

    if "video_path" in sample or "path" in sample:
        path = sample.get("video_path") or sample.get("path", "")
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return "video", path

    if "image_path" in sample or "image" in sample:
        path = sample.get("image_path") or sample.get("image", "")
        ext = os.path.splitext(path)[1].lower() if isinstance(path, str) else ""
        if ext in IMAGE_EXTENSIONS or isinstance(path, list):
            return "image", path

    for key in ("path", "video_path", "image_path", "image", "audio", "audio_path"):
        path = sample.get(key, "")
        if not path:
            continue
        p = (
            path
            if isinstance(path, str)
            else (path[0] if isinstance(path, list) and path else "")
        )
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return "video", path
        if ext in IMAGE_EXTENSIONS:
            return "image", path
        if ext in AUDIO_EXTENSIONS:
            return "audio", path

    return "unknown", sample.get(
        "path", sample.get("image_path", sample.get("video_path", ""))
    )


MCQ_INSTRUCTION_VIDEO = (
    "\nPlease answer the question use a single option like (A) or (B) or...etc."
)

MCQ_INSTRUCTION_IMAGE = "\nAnswer the question using single option word (e.g. A or B or C or D) and nothing else."


def build_prompt(
    question: Union[str, list],
    media_type: str,
    task: str,
    num_images: int = 0,
    language: str = "en",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>"

    # vision tokens go between system and first user turn
    vision_prefix = ""
    if media_type in ("image", "video") and num_images > 0:
        vision_prefix = "<|vision_start|>" + "<image>" * num_images + "<|vision_end|>"

    if isinstance(question, list):
        for i, (role, text) in enumerate(question):
            if role.lower() in ("human", "user"):
                if i == 0 and vision_prefix:
                    prompt += vision_prefix
                prompt += "\n<|im_start|>user\n"
                if i == 0 and media_type == "audio":
                    prompt += (
                        f"{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}\n"
                    )
                prompt += text + "<|im_end|>"
            else:
                prompt += f"\n<|im_start|>assistant\n{text}<|im_end|>"
        prompt += "\n<|im_start|>assistant\n"
    else:
        if vision_prefix:
            prompt += vision_prefix
        prompt += "\n<|im_start|>user\n"
        if media_type == "audio":
            prompt += f"{DEFAULT_AUDIO_START_TOKEN}<audio>{DEFAULT_AUDIO_END_TOKEN}\n"
        prompt += question + "<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def _parse_conversation_question(question_list: list) -> list[Tuple[str, str]]:
    """Convert [{"role": "Human", "value": "..."}, ...] into [(role, text), ...]."""
    turns = []
    for item in question_list:
        if isinstance(item, dict) and "role" in item and "value" in item:
            turns.append((item["role"], item["value"]))
    return turns


def prepare_question(
    sample: dict,
    task: str,
    media_type: str,
    language: str = "en",
) -> Tuple[Union[str, list], str]:
    """Build the question text and extract the ground-truth answer.

    Returns (question_text_or_turns, gt_answer).
    question_text_or_turns is a str for simple questions, or a list of
    (role, text) tuples for multi-turn conversation-format questions.
    """
    if media_type == "audio":
        gt = sample.get("text", "")
        if isinstance(gt, list):
            gt = " ".join(str(t) for t in gt)
        if task == "generate":
            q = (
                "请转录这段音频，不要输出多余解释"
                if language == "zh"
                else "transcribe the audio directly."
            )
        else:
            q = sample.get("question", "")
            if isinstance(q, list) and q and isinstance(q[0], dict):
                q = _parse_conversation_question(q)
            elif isinstance(q, list):
                q = "\n".join(str(x) for x in q)
        return q, str(gt)

    question = sample.get("question", "")
    is_conversation = (
        isinstance(question, list) and question and isinstance(question[0], dict)
    )

    if is_conversation:
        turns = _parse_conversation_question(question)
    else:
        if isinstance(question, list):
            question = "\n".join(str(q) for q in question)

    answer = sample.get("answer", sample.get("gt_answer", ""))
    if isinstance(answer, list):
        answer = answer[0] if len(answer) == 1 else "\n".join(str(a) for a in answer)

    if task == "mcq":
        answer_option = ""
        if is_conversation:
            if "candidates" in sample and media_type in ("image", "video"):
                last_role, last_text = turns[-1]
                for idx, c in enumerate(sample["candidates"]):
                    option_char = chr(ord("A") + idx)
                    last_text += f"\n({option_char}) {c}\n"
                    if c == answer:
                        answer_option = option_char
                turns[-1] = (last_role, last_text)
            elif "candidates" in sample:
                for idx, c in enumerate(sample["candidates"]):
                    if c == answer:
                        answer_option = chr(ord("A") + idx)
            return turns, answer_option or str(answer)
        else:
            if "candidates" in sample:
                for idx, c in enumerate(sample["candidates"]):
                    option_char = chr(ord("A") + idx)
                    question += f"\n({option_char}) {c}\n"
                    if c == answer:
                        answer_option = option_char
            return question, answer_option or str(answer)
    else:
        if is_conversation:
            return turns, str(answer)
        return question, str(answer)


def extract_mcq_answer(text: str) -> str:
    match = re.search(r"([A-Z])", text)
    return match.group(1) if match else ""


def resolve_sample_task_metric(
    sample: dict,
    media_type: str,
    global_task: str = "",
    global_metric: str = "",
) -> Tuple[str, str]:
    """Per-sample task and metric resolution.

    Rules (applied when no global override is given):
    - category ending with '_generate' -> generate task, edit_distance metric
    - audio -> generate task, cer (zh) or wer (en)
    - image/video -> mcq task, accuracy metric
    """
    category = sample.get("category", sample.get("task", ""))

    if not global_task and isinstance(category, str) and category.endswith("_generate"):
        task = "generate"
    elif global_task:
        task = global_task
    elif media_type == "audio":
        task = "generate"
    else:
        task = "mcq"

    if (
        not global_metric
        and isinstance(category, str)
        and category.endswith("_generate")
    ):
        metric = "edit_distance"
    elif global_metric:
        metric = global_metric
    elif media_type == "audio":
        language = sample.get("language", "en")
        metric = "cer" if language == "zh" else "wer"
    else:
        metric = "accuracy"

    return task, metric


import math

IMAGE_FACTOR = 28
MIN_PIXELS_SEQ = 64
MAX_PIXELS_SEQ = 1980
MAX_RATIO = 32
IMAGE_MIN_SIDE = 112


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS_SEQ * IMAGE_FACTOR**2,
    max_pixels: int = MAX_PIXELS_SEQ * IMAGE_FACTOR**2,
    min_side: int = IMAGE_MIN_SIDE,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
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


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(
            pil_image, mask=pil_image.split()[3]
        )  # Use alpha channel as mask
        return white_background
    if pil_image.mode == "RGB":
        return pil_image
    else:
        return pil_image.convert("RGB")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor
