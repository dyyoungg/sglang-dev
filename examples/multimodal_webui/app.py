"""Local multimodal chat gateway. Run with ./deploy.sh."""

import argparse
import asyncio
import base64
import contextlib
import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .adapters import chat_events
from .media import MediaStore
from .registry import Registry

ROOT = Path(__file__).resolve().parent


class Settings(BaseModel):
    thinking: bool = False
    downsample: Literal[1, 4, 12, 16] = 4
    video_layout: Literal["fps_groups", "single_segment"] = "fps_groups"
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    presence_penalty: float = Field(default=1.5, ge=-2, le=2)
    max_edge: int = Field(default=1280, ge=224, le=2048)
    video_frames: int = Field(default=16, ge=1, le=64)
    video_fps: Literal[2, 4, 6] = 2
    video_audio: bool = False
    system: str = Field(default="", max_length=16000)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100000)
    attachments: list[str] = Field(default_factory=list, max_length=8)


class Chat(BaseModel):
    model_id: str
    messages: list[Message] = Field(min_length=1, max_length=80)
    settings: Settings = Field(default_factory=Settings)

    @model_validator(mode="after")
    def validate_history(self):
        if self.messages[-1].role != "user":
            raise ValueError("最后一条消息必须来自用户")
        if sum(len(m.attachments) for m in self.messages) > 24:
            raise ValueError("单次对话最多包含 24 个媒体附件，请新建对话")
        if sum(len(m.content) for m in self.messages) > 200000:
            raise ValueError("对话过长，请新建对话")
        for i, message in enumerate(self.messages):
            if message.role != ("user" if i % 2 == 0 else "assistant"):
                raise ValueError("用户和助手消息必须交替出现")
            if message.role == "assistant" and message.attachments:
                raise ValueError("助手消息不能包含输入附件")
        return self


class Service(BaseModel):
    url: str = Field(max_length=300)
    name: str = Field(default="", max_length=120)


def create_app(config, media_dir, ports=(8000, 30000, 18005, 18006), own_port=7860):
    registry = Registry(config, ports, own_port)
    store = MediaStore(media_dir)
    upload_lock = asyncio.Lock()
    preparation_slots = asyncio.Semaphore(4)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(registry.watch())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app = FastAPI(title="Omni Studio", lifespan=lifespan)
    app.state.registry = registry
    app.state.store = store
    token = os.environ.get("WEBUI_TOKEN", "")

    @app.middleware("http")
    async def access(request: Request, call_next):
        if token:
            try:
                scheme, value = request.headers.get("authorization", "").split(" ", 1)
                password = base64.b64decode(value).decode().split(":", 1)[1]
                valid = scheme.lower() == "basic" and hmac.compare_digest(
                    password.encode(), token.encode()
                )
            except (ValueError, IndexError, UnicodeError):
                valid = False
            if not valid:
                return JSONResponse(
                    {"detail": "请输入访问密码"},
                    401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="Omni Studio", charset="UTF-8"'
                    },
                )
        # Reject browser cross-origin mutations; no cross-origin proxy access.
        origin = request.headers.get("origin")
        if (
            request.method not in ("GET", "HEAD")
            and origin
            and origin.rstrip("/") != str(request.base_url).rstrip("/")
        ):
            return JSONResponse({"detail": "不允许跨站请求"}, 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/api/models")
    async def models():
        return {
            "models": registry.public(),
            "updated_at": registry.updated_at,
            "error": registry.last_error,
        }

    @app.post("/api/discover")
    async def discover():
        await registry.refresh()
        return await models()

    @app.post("/api/services")
    async def add_service(service: Service):
        try:
            async with registry.lock:
                registry.add(service.url, service.name)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        await registry.refresh()
        return await models()

    @app.post("/api/upload")
    async def upload(file: UploadFile):
        try:
            async with upload_lock:
                return await store.upload(file)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/media/{uid}")
    async def media(uid: str):
        try:
            path, info = store.get(uid)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type=info["mime"])

    @app.post("/api/chat")
    async def chat(body: Chat):
        model = registry.models.get(body.model_id)
        if not model or not model["online"]:
            raise HTTPException(409, "所选模型服务已离线，请刷新模型列表")
        try:
            frame_count = 0
            for message in body.messages:
                for uid in message.attachments:
                    _, info = store.get(uid)
                    frame_count += (
                        body.settings.video_frames
                        if info["kind"] == "video"
                        else int(info["kind"] == "image")
                    )
                    needed = {info["kind"]}
                    if (
                        info["kind"] == "video"
                        and body.settings.video_audio
                        and info.get("has_audio")
                    ):
                        needed.add("audio")
                    if model["capabilities_known"] and not needed.issubset(
                        model["capabilities"]
                    ):
                        raise ValueError(
                            "所选模型不支持此附件类型；请切换模型或关闭视频音轨"
                        )
            if frame_count > 128:
                raise ValueError("对话媒体超过 128 帧，请减少视频采样帧数或新建对话")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        async def stream():
            async with preparation_slots:
                async for event in chat_events(model.copy(), body, store):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description="Omni Studio: local multimodal chat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", type=Path, default=ROOT / "services.local.json")
    parser.add_argument("--media-dir", type=Path, default=ROOT / ".data" / "media")
    parser.add_argument("--scan-ports", default="8000,30000,18005,18006")
    args = parser.parse_args()
    ports = [int(port) for port in args.scan_ports.split(",") if port.strip()]
    if any(not 0 < port < 65536 for port in ports):
        parser.error("scan ports must be between 1 and 65535")
    uvicorn.run(
        create_app(args.config, args.media_dir, ports, args.port),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
