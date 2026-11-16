"""Bounded local uploads and model-independent image/video/audio preparation."""

import base64
import io
import json
import math
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
    ".webp": ("image", "image/webp"),
    ".gif": ("image", "image/gif"),
    ".bmp": ("image", "image/bmp"),
    ".mp4": ("video", "video/mp4"),
    ".mov": ("video", "video/quicktime"),
    ".webm": ("video", "video/webm"),
    ".mkv": ("video", "video/x-matroska"),
    ".wav": ("audio", "audio/wav"),
    ".mp3": ("audio", "audio/mpeg"),
    ".m4a": ("audio", "audio/mp4"),
    ".flac": ("audio", "audio/flac"),
    ".ogg": ("audio", "audio/ogg"),
}
MAX_UPLOAD = 100 * 1024 * 1024
MAX_STORAGE = 2 * 1024 * 1024 * 1024


def command(args, timeout=60):
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise ValueError("请安装 ffmpeg 和 ffprobe 后再使用音视频") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("媒体处理超时，请上传较短的音视频") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("无法解码媒体，请确认文件格式和内容有效") from exc
    return result.stdout


def inspect_media(path):
    kind, mime = EXTENSIONS[path.suffix.lower()]
    if kind == "image":
        try:
            with Image.open(path) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("图片超过 4000 万像素，请缩小后上传")
                image.verify()
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise ValueError("无法解码图片，请确认文件内容有效") from exc
        return {"kind": kind, "mime": mime}
    info = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    duration = float(info.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or not 0 < duration <= 600:
        raise ValueError("音视频时长需在 0–600 秒之间")
    streams = {s["codec_type"] for s in info.get("streams", [])}
    if kind not in streams:
        raise ValueError(f"文件中没有 {kind} 流")
    return {
        "kind": kind,
        "mime": mime,
        "duration": duration,
        "has_audio": "audio" in streams,
    }


class MediaStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        for path in self.root.iterdir():
            if path.is_file() and time.time() - path.stat().st_mtime > 7 * 86400:
                path.unlink(missing_ok=True)

    async def upload(self, upload):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in EXTENSIONS:
            raise ValueError("不支持此格式，请上传图片、视频或音频文件")
        self.cleanup()
        used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        uid = uuid.uuid4().hex
        path = self.root / (uid + suffix)
        size = 0
        try:
            with path.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD or used + size > MAX_STORAGE:
                        raise ValueError("单个文件上限 100 MB；附件存储总上限 2 GB")
                    stream.write(chunk)
            import asyncio

            info = await asyncio.to_thread(inspect_media, path)
            info.update(
                id=uid, name=Path(upload.filename).name[:200], size=size, suffix=suffix
            )
            (self.root / (uid + ".json")).write_text(
                json.dumps(info, ensure_ascii=False)
            )
            return self.public(info)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

    def get(self, uid):
        if len(uid) != 32 or any(c not in "0123456789abcdef" for c in uid):
            raise ValueError("无效的附件 ID")
        try:
            info = json.loads((self.root / (uid + ".json")).read_text())
            path = self.root / (uid + info["suffix"])
            if not path.is_file():
                raise FileNotFoundError
            path.touch()
            (self.root / (uid + ".json")).touch()
            return path, info
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError("附件已过期，请重新上传（保留 7 天）") from exc

    @staticmethod
    def public(info):
        return {**info, "url": f"/api/media/{info['id']}"}


def jpeg(image, max_edge):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1, max_edge / max(image.size))
    size = tuple(max(32, round(v * scale / 32) * 32) for v in image.size)
    image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def prepare_media(store, ids, settings):
    items = []
    for uid in ids:
        path, info = store.get(uid)
        if info["kind"] == "image":
            with Image.open(path) as image:
                items.append({"kind": "image", "data": jpeg(image, settings.max_edge)})
        elif info["kind"] == "video":
            # Uniform samples span the clip and carry group IDs for BeeBee's
            # sampled-FPS prompt layout. Standard VLMs receive timestamped images.
            count = min(
                settings.video_frames,
                max(1, math.ceil(info["duration"] * settings.video_fps)),
            )
            for i in range(count):
                timestamp = (i + 0.5) * info["duration"] / count
                raw = command(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-nostdin",
                        "-ss",
                        str(timestamp),
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale={settings.max_edge}:{settings.max_edge}:force_original_aspect_ratio=decrease",
                        "-threads",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "pipe:1",
                    ]
                )
                with Image.open(io.BytesIO(raw)) as image:
                    items.append(
                        {
                            "kind": "image",
                            "data": jpeg(image, settings.max_edge),
                            "timestamp": timestamp,
                            "video_id": uid,
                            "video_group": i // settings.video_fps,
                        }
                    )
        if info["kind"] == "audio" or (
            info["kind"] == "video" and settings.video_audio and info.get("has_audio")
        ):
            raw = command(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-threads",
                    "1",
                    "-f",
                    "wav",
                    "pipe:1",
                ]
            )
            items.append({"kind": "audio", "data": base64.b64encode(raw).decode()})
    return items
