"""The `/generate` schema gate (`entrypoints/api_contract.py`): the typed media
contract that replaced the raw_json passthrough, checked without a server, and
where the route applies it."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.api_contract import generate_contract_error
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGenerateContract(CustomTestCase):
    def test_every_media_container_form_is_accepted(self):
        for image_data in (
            "u",
            {"url": "u", "detail": "high"},
            ["u", None, {"url": "v"}],
            [],
            [["a", "b"], None, []],
        ):
            with self.subTest(image_data=image_data):
                self.assertIsNone(
                    generate_contract_error({"text": "hi", "image_data": image_data})
                )
        self.assertIsNone(
            generate_contract_error({"text": "hi", "mm_hashes": ["a1b2", "0xff"]})
        )
        self.assertIsNone(
            generate_contract_error(
                {
                    "text": "hi",
                    "video_data": {"url": "v", "fps": 2.0, "use_audio": True},
                }
            )
        )

    def test_unknown_media_hint_is_rejected(self):
        """A hint the schema does not name used to be silently dropped."""
        error = generate_contract_error(
            {
                "text": "hi",
                "image_data": {"url": "u", "max_dynamic_patc": 4},
            }  # codespell:ignore patc
        )
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)

    def test_glm_external_ar_grid_payload_is_accepted(self):
        """GLM external AR sends grid metadata with pre-tokenized inputs."""
        for image_data in (
            # generate_prior_tokens: one request, one grid list
            [{"image_grid_thw": [[1, 32, 32], [1, 64, 64]]}],
            # generate_prior_tokens_batch: one grid list per batch entry
            [
                [{"image_grid_thw": [[1, 32, 32], [1, 64, 64]]}],
                [{"image_grid_thw": [[1, 32, 32], [1, 16, 16]]}],
            ],
        ):
            with self.subTest(image_data=image_data):
                self.assertIsNone(
                    generate_contract_error(
                        {"input_ids": [1, 2, 3], "image_data": image_data}
                    )
                )
        # A grid row of the wrong scalar type is still a contract error.
        error = generate_contract_error(
            {"input_ids": [1], "image_data": [{"image_grid_thw": [[1, "h", 32]]}]}
        )
        self.assertIsNotNone(error)

    def test_preprocessed_inputs_are_not_an_http_shape(self):
        """processor_output / precomputed_embedding carry tensors: Engine-only."""
        error = generate_contract_error(
            {"input_ids": [1, 2], "image_data": [{"format": "processor_output"}]}
        )
        self.assertIsNotNone(error)

    def test_mixed_list_shapes_are_rejected(self):
        error = generate_contract_error({"text": "hi", "image_data": ["a", ["b"]]})
        self.assertIsNotNone(error)

    def test_sampling_params_follow_the_schema(self):
        self.assertIsNone(
            generate_contract_error(
                {"text": "hi", "sampling_params": {"beam_width": 2, "n": 2}}
            )
        )
        bad = {"temperatur": 1}  # codespell:ignore temperatur
        error = generate_contract_error({"text": "hi", "sampling_params": bad})
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)


class TestGenerateRouteGate(CustomTestCase):
    def test_route_checks_the_contract_and_serve_does_not(self):
        """`/generate` refuses a contract violation before serving it;
        `serve_generate_request`, which routes with their own parser call,
        serves it without decoding and checking the body again."""
        served = []

        class FakeTokenizerManager:
            async def generate_request(self, obj, request):
                served.append(obj)
                yield {"text": "ok"}

        bad = {"temperatur": 1}  # codespell:ignore temperatur
        body = {"text": "hi", "sampling_params": bad}

        async def admitted(request: Request):
            obj = GenerateReqInput(**(await request.json()))
            return await http_server.serve_generate_request(obj, request)

        app = FastAPI()
        app.add_api_route("/generate", http_server.generate_request, methods=["POST"])
        app.add_api_route("/admitted", admitted, methods=["POST"])
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=FakeTokenizerManager())
        )
        try:
            client = TestClient(app)
            response = client.post("/generate", json=body)
            self.assertEqual(response.status_code, 400)
            self.assertIn("unknown field", response.json()["error"])
            self.assertEqual(served, [])

            response = client.post("/admitted", json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"text": "ok"})
            self.assertEqual(len(served), 1)
        finally:
            http_server._global_state = prior_state


class TestGenerateStreamRoute(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.served = []
        self.cleaned = []
        self.generation_error = None
        self.generation_error_type = ValueError

        async def generate(obj, request):
            self.served.append(obj)
            if self.generation_error is not None:
                raise self.generation_error_type(self.generation_error)
            yield {"text": "first"}
            yield {"text": "first second"}

        manager = SimpleNamespace(
            generate_request=generate,
            create_abort_task=lambda obj: BackgroundTask(self.cleaned.append, obj),
        )
        state = patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
        )
        state.start()
        self.addCleanup(state.stop)
        # Exercise the actual registered routes without starting model workers.
        app = FastAPI()
        app.router.routes.extend(
            route
            for route in http_server.app.router.routes
            if route.path in {"/generate", "/generate_stream"}
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_legacy_route_always_streams_and_preserves_generate(self):
        """The merge dropped /generate_stream; legacy clients received 404.
        Both verbs must force SSE even when stream is absent or false, while
        /generate must still honor a non-streaming request.
        """
        for method in ("POST", "PUT"):
            for stream in (None, False, True):
                with self.subTest(method=method, stream=stream):
                    body = {"text": "hi"}
                    if stream is not None:
                        body["stream"] = stream
                    response = self.client.request(
                        method, "/generate_stream", json=body
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(
                        response.headers["content-type"].startswith("text/event-stream")
                    )
                    self.assertEqual(
                        response.text,
                        'data: {"text":"first"}\n\n'
                        'data: {"text":"first second"}\n\n'
                        "data: [DONE]\n\n",
                    )
                    self.assertTrue(self.served[-1].stream)
                    self.assertIs(self.cleaned[-1], self.served[-1])

        response = self.client.post("/generate", json={"text": "hi", "stream": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "first"})
        self.assertFalse(self.served[-1].stream)
        self.assertEqual(len(self.cleaned), 6)

    def test_stream_route_rejects_contract_violations_before_serving(self):
        """The compatibility route must not bypass the typed media gate."""
        for method in ("POST", "PUT"):
            with self.subTest(method=method):
                response = self.client.request(
                    method,
                    "/generate_stream",
                    json={"text": "hi", "image_data": {"unknown_hint": 1}},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("unknown field", response.json()["error"])
        self.assertEqual(self.served, [])
        self.assertEqual(self.cleaned, [])

    def test_stream_errors_use_the_shared_sse_error_contract(self):
        """A generation failure must reach clients as an SSE error frame."""
        self.generation_error = "invalid sampling parameters"
        response = self.client.post("/generate_stream", json={"text": "hi"})
        self.assertEqual(response.status_code, 200)
        error, done, _ = response.text.split("\n\n")
        self.assertEqual(
            json.loads(error.removeprefix("data: ")),
            {
                "error": {
                    "message": self.generation_error,
                    "type": "invalid_request_error",
                    "code": 400,
                    "retryable": False,
                }
            },
        )
        self.assertEqual(done, "data: [DONE]")
        self.assertEqual(len(self.cleaned), 1)

    def test_stream_runtime_errors_complete_the_sse_response(self):
        """Unhandled generator exceptions must not truncate chunked SSE."""
        self.generation_error = "worker failed"
        self.generation_error_type = RuntimeError
        response = self.client.post("/generate_stream", json={"text": "hi"})
        self.assertEqual(response.status_code, 200)
        frames = response.text.split("\n\n")
        self.assertEqual(frames[-2], "data: [DONE]")
        error = json.loads(frames[0].removeprefix("data: "))["error"]
        self.assertEqual(error["code"], 500)
        self.assertEqual(error["type"], "server_error")
        self.assertTrue(error["retryable"])


if __name__ == "__main__":
    unittest.main()
