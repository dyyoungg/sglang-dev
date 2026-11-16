"""Discover local OpenAI-compatible servers without loading model weights."""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)


def local_url(value):
    parsed = urlsplit(value.strip().rstrip("/"))
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    try:
        valid = ipaddress.ip_address(host).is_loopback
    except ValueError:
        valid = False
    if (
        not valid
        or parsed.scheme != "http"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/v1")
    ):
        raise ValueError("请输入本机 HTTP 服务地址，例如 http://127.0.0.1:18006")
    port = parsed.port or 80
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}:{port}"


def listening_urls():
    """Linux LISTEN sockets, including services bound to 0.0.0.0 / ::."""
    urls = set()
    for name in ("tcp", "tcp6"):
        try:
            rows = Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if fields[3] == "0A":
                port = int(fields[1].split(":")[1], 16)
                urls.add(f"http://{'[::1]' if name == 'tcp6' else '127.0.0.1'}:{port}")
    return urls


def describe_model(info, model, override):
    path = (
        override.get("tokenizer_path")
        or info.get("model_path")
        or model.get("root")
        or model["id"]
    )
    config = {}
    try:
        config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        pass
    signature = json.dumps(config).lower() + str(path).lower()
    beebee = any(
        s in signature for s in ("beebee", "llavaqwen35", "llavaqwen3", "llavaqwen2")
    )
    adapter = override.get("adapter", "beebee" if beebee else "openai")
    if adapter not in ("beebee", "openai"):
        raise ValueError(f"Unknown adapter: {adapter}")
    caps = override.get("capabilities")
    if caps is None:
        caps = ["text"]
        if (
            beebee
            or "vision_config" in config
            or "omni" in signature
            or "-vl" in signature
        ):
            caps += ["image", "video"]
        if beebee or "audio_config" in config or "omni" in signature:
            caps += ["audio"]
    return {
        "adapter": adapter,
        "tokenizer_path": path,
        "capabilities": caps,
        "capabilities_known": bool(
            config or override.get("capabilities") or len(caps) > 1
        ),
    }


class Registry:
    def __init__(self, config_path, scan_ports, own_port):
        self.config_path = Path(config_path)
        self.scan_ports = scan_ports
        self.own_port = own_port
        self.models = {}
        self.lock = asyncio.Lock()
        self.updated_at = None
        self.last_error = None

    def configured(self):
        if not self.config_path.exists():
            return []
        data = json.loads(self.config_path.read_text())
        return data.get("services", [])

    def add(self, url, name=""):
        url = local_url(url)
        entries = self.configured()
        previous = next((e for e in entries if local_url(e["url"]) == url), {})
        entries = [e for e in entries if local_url(e["url"]) != url]
        entries.append({**previous, "url": url, "name": name[:120]})
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"services": entries}, ensure_ascii=False, indent=2))
        tmp.replace(self.config_path)

    async def refresh(self):
        async with self.lock:
            try:
                entries = self.configured()
                configured = {local_url(e["url"]): e for e in entries}
            except (ValueError, OSError, KeyError) as exc:
                self.last_error = f"服务配置读取失败：{exc}"
                return
            urls = listening_urls() | set(configured)
            urls |= {f"http://127.0.0.1:{p}" for p in self.scan_ports}
            urls = {u for u in urls if urlsplit(u).port != self.own_port}
            # Prefer IPv4 when both stacks expose the same server.
            urls = {
                u
                for u in urls
                if "[::1]" not in u or u.replace("[::1]", "127.0.0.1") not in urls
            }
            semaphore = asyncio.Semaphore(16)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(2, connect=0.5), trust_env=False
            ) as client:

                async def probe(url):
                    override = configured.get(url, {})
                    key = os.getenv(override.get("api_key_env", ""), "")
                    headers = {"Authorization": f"Bearer {key}"} if key else {}
                    async with semaphore:
                        try:
                            response = await client.get(
                                url + "/v1/models", headers=headers
                            )
                            response.raise_for_status()
                            data = response.json().get("data", [])
                            if not isinstance(data, list) or not data:
                                return []
                            info = {}
                            try:
                                response = await client.get(
                                    url + "/get_model_info", headers=headers
                                )
                                if response.is_success:
                                    info = response.json()
                            except (httpx.HTTPError, ValueError):
                                pass
                            result = []
                            for model in data:
                                if not isinstance(model, dict) or not isinstance(
                                    model.get("id"), str
                                ):
                                    continue
                                uid = hashlib.sha256(
                                    f"{url}|{model['id']}".encode()
                                ).hexdigest()[:20]
                                model_name = Path(model["id"]).name
                                if model_name in ("hf_ckpt", "checkpoint", "model"):
                                    model_name = (
                                        Path(model["id"]).parent.name or model_name
                                    )
                                result.append(
                                    {
                                        "id": uid,
                                        "model": model["id"],
                                        "url": url,
                                        "name": override.get("name") or model_name,
                                        "online": True,
                                        "last_seen": time.time(),
                                        "api_key_env": override.get("api_key_env", ""),
                                        **describe_model(info, model, override),
                                    }
                                )
                            return result
                        except (httpx.HTTPError, ValueError, TypeError, KeyError):
                            return []

                discovered = await asyncio.gather(*(probe(url) for url in sorted(urls)))
            for model in self.models.values():
                model["online"] = False
            for group in discovered:
                for model in group:
                    self.models[model["id"]] = model
            self.updated_at = time.time()
            self.last_error = None

    def public(self):
        return [
            {
                k: v
                for k, v in model.items()
                if k not in ("api_key_env", "tokenizer_path")
            }
            for model in sorted(
                self.models.values(),
                key=lambda m: (not m["online"], m["name"], m["url"]),
            )
        ]

    async def watch(self):
        while True:
            try:
                await self.refresh()
            except Exception:
                log.exception("Local service discovery failed")
            await asyncio.sleep(10)
