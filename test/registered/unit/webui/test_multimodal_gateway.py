"""Protocol invariants for the optional local multimodal example gateway."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from examples.multimodal_webui import adapters, registry
from examples.multimodal_webui.app import Chat, create_app

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class NativeTemplate:
    """External tokenizer boundary; production renders with AutoTokenizer."""

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        return (
            text
            + "<|im_start|>assistant\n<think>\n"
            + ("" if kwargs["enable_thinking"] else "\n</think>\n\n")
        )


class TestMultimodalGateway(CustomTestCase):
    def test_native_media_stays_with_its_turn_and_thinking_is_explicit(self):
        """Media must precede the owning user turn, never migrate to turn one."""
        body = Chat(
            model_id="m",
            messages=[
                {"role": "user", "content": "read", "attachments": ["image"]},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "listen", "attachments": ["audio"]},
            ],
            settings={"system": "system", "downsample": 1, "thinking": True},
        )

        def media(store, ids, settings):
            return [{"kind": uid, "data": uid + "-bytes"} for uid in ids]

        with (
            patch.object(adapters, "tokenizer_for", return_value=NativeTemplate()),
            patch.object(adapters, "prepare_media", side_effect=media),
        ):
            endpoint, payload = adapters.build_payload(
                {"adapter": "beebee", "tokenizer_path": "unused"}, body, None
            )
            self.assertEqual(endpoint, "/generate")
            self.assertEqual(payload["image_data"], ["image-bytes", "image-bytes"])
            self.assertEqual(payload["image_downsample_ratios"], [1])
            self.assertEqual(payload["audio_data"], ["audio-bytes"])
            self.assertIn(
                "system<|im_end|><|vision_start|><image><image><|vision_end|>\n<|im_start|>user\nread",
                payload["text"],
            )
            self.assertIn(
                "first answer<|im_end|>\n<|im_start|>user\n<audio>listen",
                payload["text"],
            )
            self.assertTrue(payload["text"].endswith("<think>\n"))
            body.settings.thinking = False
            body.settings.system = ""
            _, payload = adapters.build_payload(
                {"adapter": "beebee", "tokenizer_path": "unused"}, body, None
            )
            self.assertTrue(payload["text"].startswith("<|vision_start|>"))
            self.assertTrue(payload["text"].endswith("<think>\n\n</think>\n\n"))

    def test_empty_native_reasoning_does_not_swallow_reopened_span(self):
        result = adapters.split_reasoning(
            "</think>\n\n<think>second reasoning</think>\nanswer", True
        )
        self.assertEqual(result["content"], "answer")
        self.assertEqual(
            result["reasoning_blocks"],
            [
                {"text": "", "complete": True},
                {"text": "second reasoning", "complete": True},
            ],
        )
        self.assertEqual(result["reasoning"], "second reasoning")
        self.assertEqual(adapters.split_reasoning("unfinished", True)["content"], "")
        self.assertEqual(adapters.split_reasoning("answer", False)["content"], "answer")

    def test_video_frame_layout_supports_fps_groups_and_single_segment(self):
        body = Chat(
            model_id="m",
            messages=[{"role": "user", "content": "summarize", "attachments": ["video"]}],
            settings={"video_layout": "fps_groups", "video_fps": 2, "downsample": 4},
        )

        def media(store, ids, settings):
            return [
                {"kind": "image", "data": "frame-1", "timestamp": 0.5, "video_id": "video", "video_group": 0},
                {"kind": "image", "data": "frame-2", "timestamp": 0.8, "video_id": "video", "video_group": 0},
                {"kind": "image", "data": "frame-3", "timestamp": 1.0, "video_id": "video", "video_group": 1},
                {"kind": "image", "data": "frame-4", "timestamp": 1.3, "video_id": "video", "video_group": 1},
            ]

        with (
            patch.object(adapters, "tokenizer_for", return_value=NativeTemplate()),
            patch.object(adapters, "prepare_media", side_effect=media),
        ):
            _, payload = adapters.build_payload(
                {"adapter": "beebee", "tokenizer_path": "unused"}, body, None
            )
            self.assertEqual(payload["image_data"], ["frame-1", "frame-2", "frame-3", "frame-4"])
            self.assertEqual(payload["image_downsample_ratios"], [4, 4, 4, 4])
            self.assertIn(
                "<|vision_start|><image><image><|vision_end|><|vision_start|><image><image><|vision_end|>",
                payload["text"],
            )
            self.assertNotIn("0.50s", payload["text"])

            body.settings.video_layout = "single_segment"
            _, payload = adapters.build_payload(
                {"adapter": "beebee", "tokenizer_path": "unused"}, body, None
            )
            self.assertEqual(payload["image_data"], ["frame-1", "frame-2", "frame-3", "frame-4"])
            self.assertIn(
                "<|vision_start|><image><image><image><image><|vision_end|>",
                payload["text"],
            )

    def test_same_model_two_ports_offline_then_recovery(self):
        """Model IDs alone cannot identify two independent local deployments."""

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                reg = registry.Registry(
                    Path(directory) / "services.json", [18005, 18006], 7860
                )
                online = {18005, 18006}

                def handler(request):
                    if request.url.port not in online:
                        raise httpx.ConnectError("offline")
                    if request.url.path == "/v1/models":
                        return httpx.Response(
                            200, json={"data": [{"id": "same-model"}]}
                        )
                    return httpx.Response(404)

                client_type = httpx.AsyncClient

                def client(**kwargs):
                    return client_type(transport=httpx.MockTransport(handler), **kwargs)

                with (
                    patch.object(registry, "listening_urls", return_value=set()),
                    patch.object(registry.httpx, "AsyncClient", side_effect=client),
                ):
                    await reg.refresh()
                    self.assertEqual(len(reg.models), 2)
                    keys = set(reg.models)
                    online.remove(18005)
                    await reg.refresh()
                    self.assertEqual(sum(m["online"] for m in reg.models.values()), 1)
                    online.add(18005)
                    await reg.refresh()
                    self.assertEqual(set(reg.models), keys)
                    self.assertTrue(all(m["online"] for m in reg.models.values()))

        asyncio.run(run())

    def test_stream_cumulative_and_delta_are_not_confused(self):
        async def run():
            for adapter in ("beebee", "openai"):
                if adapter == "beebee":
                    chunks = [
                        {"text": "你"},
                        {
                            "text": "你好",
                            "meta_info": {
                                "completion_tokens": 2,
                                "finish_reason": {"type": "stop"},
                            },
                        },
                    ]
                else:
                    chunks = [
                        {"choices": [{"delta": {"content": c}}]} for c in ("你", "好")
                    ]
                wire = (
                    ": ping\n\n"
                    + "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                    + "data: [DONE]\n\n"
                )
                client_type = httpx.AsyncClient

                def client(client_type=client_type, wire=wire, **kwargs):
                    return client_type(
                        transport=httpx.MockTransport(
                            lambda r: httpx.Response(
                                200,
                                text=wire,
                                headers={"content-type": "text/event-stream"},
                            )
                        ),
                        **kwargs,
                    )

                body = Chat(model_id="m", messages=[{"role": "user", "content": "hi"}])
                model = {"url": "http://127.0.0.1:18006", "adapter": adapter}
                with (
                    patch.object(
                        adapters, "build_payload", return_value=("/generate", {})
                    ),
                    patch.object(adapters.httpx, "AsyncClient", side_effect=client),
                ):
                    events = [e async for e in adapters.chat_events(model, body, None)]
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual(
                    [e for e in events if e["type"] == "snapshot"][-1]["content"],
                    "你好",
                )

        asyncio.run(run())

    def test_upload_validation_and_media_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                Path(directory) / "services.json", Path(directory) / "media"
            )
            client = TestClient(app)
            image = io.BytesIO()
            Image.new("RGB", (32, 32)).save(image, "PNG")
            response = client.post(
                "/api/upload",
                files={"file": ("photo.png", image.getvalue(), "image/png")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(client.get(data["url"]).content, image.getvalue())
            self.assertEqual(client.get("/api/media/not-an-id").status_code, 404)
            self.assertEqual(
                client.post(
                    "/api/upload", files={"file": ("x.html", b"<script>", "text/html")}
                ).status_code,
                400,
            )
            self.assertEqual(
                client.post(
                    "/api/chat",
                    json={
                        "model_id": "missing",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ).status_code,
                409,
            )
            client.close()

    def test_only_loopback_registration_and_override_preservation(self):
        for url in (
            "http://example.com:8000",
            "http://127.0.0.1@evil.test:80",
            "http://127.0.0.1:8000/admin",
            "file:///etc/passwd",
        ):
            with self.assertRaises(ValueError):
                registry.local_url(url)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "services.json"
            path.write_text(
                json.dumps(
                    {
                        "services": [
                            {
                                "url": "http://127.0.0.1:8000",
                                "adapter": "beebee",
                                "tokenizer_path": "/model",
                            }
                        ]
                    }
                )
            )
            reg = registry.Registry(path, [], 7860)
            reg.add("http://localhost:8000/v1", "renamed")
            self.assertEqual(reg.configured()[0]["tokenizer_path"], "/model")
            self.assertEqual(reg.configured()[0]["adapter"], "beebee")


if __name__ == "__main__":
    unittest.main()
