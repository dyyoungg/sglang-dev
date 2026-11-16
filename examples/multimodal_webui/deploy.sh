#!/usr/bin/env bash
set -euo pipefail
WEBUI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_PYTHON="${WEBUI_PYTHON:-python3}"
if [[ "${1:-}" == "--install" ]]; then
  shift
  "$WEBUI_PYTHON" -m venv --system-site-packages "$WEBUI_DIR/.venv"
  "$WEBUI_DIR/.venv/bin/python" -m pip install -r "$WEBUI_DIR/requirements.txt"
fi
if [[ -x "$WEBUI_DIR/.venv/bin/python" ]]; then
  WEBUI_PYTHON="$WEBUI_DIR/.venv/bin/python"
fi
if ! "$WEBUI_PYTHON" -c 'import fastapi, uvicorn, httpx, multipart, PIL' 2>/dev/null; then
  echo "Dependencies missing. Run: $0 --install [--host 127.0.0.1 --port 7860]" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "ffmpeg/ffprobe not found; install them to enable audio/video uploads." >&2
fi
cd "$WEBUI_DIR/../.."
exec "$WEBUI_PYTHON" -m examples.multimodal_webui.app "$@"
