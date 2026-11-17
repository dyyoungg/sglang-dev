#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_27B_PATH="${MODEL_27B_PATH:-${MODEL_PATH:-/mnt/afs/yangdeyu/VeOmni/ckpt/1106_llava_27B_gametext_omnisft_adamw_1e5_8k/hf_ckpt}}"
MODEL_14B_PATH="${MODEL_14B_PATH:-/mnt/afs/share/20260907_beebee_14B_veomni_28k}"
HOST="${HOST:-127.0.0.1}"
PORT_27B="${PORT_27B:-18006}"
PORT_14B="${PORT_14B:-18005}"
ACTIVE_TIME="${ACTIVE_TIME:-720}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16000}"
SYSTEM_PROMPT_TOKENS="${SYSTEM_PROMPT_TOKENS:-8192}"
TEXT_TOKEN_BUCKET="${TEXT_TOKEN_BUCKET:-10000}"
USER_COUNTS="${USER_COUNTS:-1 2 4}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/benchmarks/online_fp8_vs_bf16_$(date +%Y%m%d_%H%M%S)}"

read -r -a USERS <<< "${USER_COUNTS}"
for num_users in "${USERS[@]}"; do
    if (( num_users < 1 || num_users > 4 )); then
        echo "USER_COUNTS entries must be between 1 and 4" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_DIR}"
printf 'Online benchmark\n27B FP8=%s @ %s:%s (pair vision, patch16/down12)\n14B BF16=%s @ %s:%s (single vision, patch14/down16)\nusers=%s\nmax_context=%s\nsystem_prompt_padding=%s\ntext_history_budget=%s\nactive_time_per_user=%ss\nresults=%s\n' \
    "${MODEL_27B_PATH}" "${HOST}" "${PORT_27B}" \
    "${MODEL_14B_PATH}" "${HOST}" "${PORT_14B}" "${USER_COUNTS}" \
    "${MAX_CONTEXT_LEN}" "${SYSTEM_PROMPT_TOKENS}" "${TEXT_TOKEN_BUCKET}" \
    "${ACTIVE_TIME}" "${OUTPUT_DIR}"

/usr/bin/python3 - "${HOST}" "${PORT_27B}" "${MODEL_27B_PATH}" \
    "${PORT_14B}" "${MODEL_14B_PATH}" <<'PY'
import sys
import requests

host, port_27b, path_27b, port_14b, path_14b = sys.argv[1:]
expected = [
    (int(port_27b), path_27b, "fp8"),
    (int(port_14b), path_14b, None),
]
for port, model_path, quantization in expected:
    info = requests.get(f"http://{host}:{port}/server_info", timeout=5).json()
    actual = (info.get("model_path"), info.get("quantization"))
    assert actual == (model_path, quantization), (port, actual, model_path, quantization)
    assert info.get("tp_size") == 1 and info.get("max_running_requests") == 4, (
        port, info.get("tp_size"), info.get("max_running_requests")
    )
    assert info.get("disable_radix_cache") is False, (port, info.get("disable_radix_cache"))
    assert info.get("mem_fraction_static", 0) >= 0.90, (port, info.get("mem_fraction_static"))
    print(
        f"Validated {port}: model={model_path}, quantization={quantization}, "
        f"TP1, max-running=4, radix-cache=on, mem-fraction={info['mem_fraction_static']}"
    )
PY

for num_users in "${USERS[@]}"; do
    echo "Starting synchronized online benchmark with ${num_users} users per model"
    /usr/bin/python3 "${SCRIPT_DIR}/benchmark_online_serve.py" \
        --host "${HOST}" --port "${PORT_27B}" --tokenizer "${MODEL_27B_PATH}" \
        --active-time "${ACTIVE_TIME}" --max-context-len "${MAX_CONTEXT_LEN}" \
        --text-token-bucket "${TEXT_TOKEN_BUCKET}" \
        --patch-size 16 --downsample-ratio 12 --vision-format pair \
        --system-prompt-tokens "${SYSTEM_PROMPT_TOKENS}" --seed 42 \
        --num-users "${num_users}" --request-mode all \
        --result-file "${OUTPUT_DIR}/27b_fp8_users_${num_users}.json" \
        2>&1 | tee "${OUTPUT_DIR}/27b_fp8_users_${num_users}.log" &
    pid_27b=$!

    /usr/bin/python3 "${SCRIPT_DIR}/benchmark_online_serve.py" \
        --host "${HOST}" --port "${PORT_14B}" --tokenizer "${MODEL_14B_PATH}" \
        --active-time "${ACTIVE_TIME}" --max-context-len "${MAX_CONTEXT_LEN}" \
        --text-token-bucket "${TEXT_TOKEN_BUCKET}" \
        --patch-size 14 --downsample-ratio 16 --vision-format single \
        --system-prompt-tokens "${SYSTEM_PROMPT_TOKENS}" --seed 42 \
        --num-users "${num_users}" --request-mode all \
        --result-file "${OUTPUT_DIR}/14b_bf16_users_${num_users}.json" \
        2>&1 | tee "${OUTPUT_DIR}/14b_bf16_users_${num_users}.log" &
    pid_14b=$!

    status_27b=0
    status_14b=0
    wait "${pid_27b}" || status_27b=$?
    wait "${pid_14b}" || status_14b=$?
    if (( status_27b != 0 || status_14b != 0 )); then
        echo "Benchmark failed for ${num_users} users (27B=${status_27b}, 14B=${status_14b})" >&2
        exit 1
    fi
done

/usr/bin/python3 "${SCRIPT_DIR}/summarize_online_benchmark.py" "${OUTPUT_DIR}"
echo "All synchronized user-count runs completed. Results: ${OUTPUT_DIR}"
