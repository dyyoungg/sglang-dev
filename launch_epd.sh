#!/bin/bash
# ─── BeeBeeOmni EPD Disaggregation Launch Script ────────────────────────────
#
# EP mode (Encoder + Language-only, no PD split):
#   Terminal 1: bash launch_epd.sh <model_path> encoder 1
#   Terminal 2: bash launch_epd.sh <model_path> language 1
#
# Multi-encoder DP (each encoder on its own GPU, no TP):
#   Terminal 1: bash launch_epd.sh <model_path> encoders 1 4   # 4 encoders on GPU 0-3
#   Terminal 2: bash launch_epd.sh <model_path> language 1 4   # connects to all 4
#
# Single encoder with specific GPU:
#   Terminal 1: bash launch_epd.sh <model_path> encoder 1 0 30000  # GPU 0, port 30000
#   Terminal 2: bash launch_epd.sh <model_path> encoder 1 1 30001  # GPU 1, port 30001
#   Terminal 3: bash launch_epd.sh <model_path> language 1 2       # connects to 2 encoders
#
# EPD mode (Encoder + Prefill + Decode + Router):
#   Terminal 1: bash launch_epd.sh <model_path> encoders 1 2
#   Terminal 2: bash launch_epd.sh <model_path> prefill 1 2
#   Terminal 3: bash launch_epd.sh <model_path> decode 1
#   Terminal 4: bash launch_epd.sh <model_path> router
#
# Test:
#   python test_epd_e2e.py --port 30002 --test image --image-num 16
# ──────────────────────────────────────────────────────────────────────────────

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=True
export SGLANG_IO_WORKERS=8
export SGLANG_VLM_CACHE_SIZE_MB=2048

export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_MASTER="127.0.0.1:50051"
export MOONCAKE_PROTOCOL="rdma"    # 没有 RDMA 网卡就用 tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE="16gb"

MODEL_PATH=${1:?Usage: $0 <model_path> <role> [tp_size] [extra_args...]}
ROLE=${2:?Usage: $0 <model_path> <role> [tp_size] [extra_args...]}
TP_SIZE=${3:-1}

ENCODER_BASE_PORT=30000
LANGUAGE_PORT=30002

# Common args (from launch.sh)
COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --tokenizer-path "$MODEL_PATH"
    --model-impl sglang
    --host 0.0.0.0
    --log-level debug
    --chunked-prefill-size 8192
    --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 8}'
    --cuda-graph-max-bs 16
    --tp-size "$TP_SIZE"
    --enable-mfu-metrics
    --enable-metrics
    --enable-request-time-stats-logging
    --show-time-cost
    --enable-dynamic-batch-tokenizer
    --disable-piecewise-cuda-graph
    --enable-multimodal
    --enable-broadcast-mm-inputs-process
)

# Build encoder URLs string for a given number of encoders
build_encoder_urls() {
    local num=$1
    local urls=""
    for ((i=0; i<num; i++)); do
        local port=$((ENCODER_BASE_PORT + i))
        if [ -n "$urls" ]; then
            urls="$urls "
        fi
        urls="${urls}http://127.0.0.1:${port}"
    done
    echo "$urls"
}

case "$ROLE" in

    # ─── Single encoder (specify GPU and port) ──────────────────────────
    encoder)
        GPU_ID=${4:-0}
        PORT=${5:-$ENCODER_BASE_PORT}
        echo "=== Starting Encoder Server on GPU $GPU_ID, port $PORT ==="
        sglang serve \
            "${COMMON_ARGS[@]}" \
            --port $PORT \
            --base-gpu-id $GPU_ID \
            --encoder-only \
            --encoder-transfer-backend zmq_to_scheduler \
            --enable-prefix-mm-cache \
            --mm-attention-backend fa2
            # --mm-global-cache-pool-size-gb 8.0 \
            # --mm-global-cache-max-batch-groups 256 \
        ;;

    # ─── Multiple encoders (DP-style, one per GPU) ──────────────────────
    # Usage: launch_epd.sh <model> encoders <tp_size> <num_encoders> [base_gpu_id]
    encoders)
        NUM_ENCODERS=${4:-2}
        BASE_GPU=${5:-0}
        echo "=== Starting $NUM_ENCODERS Encoder Servers (GPU $BASE_GPU-$((BASE_GPU+NUM_ENCODERS-1))) ==="

        PIDS=()
        for ((i=0; i<NUM_ENCODERS; i++)); do
            GPU_ID=$((BASE_GPU + i * TP_SIZE))
            PORT=$((ENCODER_BASE_PORT + i))
            echo "  Encoder $i: GPU $GPU_ID, port $PORT"
            sglang serve \
                "${COMMON_ARGS[@]}" \
                --port $PORT \
                --base-gpu-id $GPU_ID \
                --encoder-only \
                --encoder-transfer-backend zmq_to_scheduler \
                --enable-prefix-mm-cache \
                --mm-attention-backend fa2 &
            PIDS+=($!)
        done

        echo ""
        echo "Encoder URLs: $(build_encoder_urls $NUM_ENCODERS)"
        echo "PIDs: ${PIDS[*]}"
        echo "Press Ctrl+C to stop all encoders"

        # Wait for all and forward signals
        trap 'kill ${PIDS[*]} 2>/dev/null; exit' INT TERM
        wait
        ;;

    # ─── Language-only server ───────────────────────────────────────────
    # Usage: launch_epd.sh <model> language <tp_size> <num_encoders>
    language)
        NUM_ENCODERS=${4:-1}
        ENCODER_URLS=$(build_encoder_urls $NUM_ENCODERS)
        echo "=== Starting Language-only Server on port $LANGUAGE_PORT ==="
        echo "=== Encoder URLs: $ENCODER_URLS ==="
        sglang serve \
            "${COMMON_ARGS[@]}" \
            --port $LANGUAGE_PORT \
            --language-only \
            --encoder-urls $ENCODER_URLS \
            --encoder-transfer-backend zmq_to_scheduler \
            --warmups "beebee_omni_warmup"
        ;;

    # ─── EPD: Prefill ───────────────────────────────────────────────────
    # Usage: launch_epd.sh <model> prefill <tp_size> <num_encoders>
    prefill)
        NUM_ENCODERS=${4:-1}
        PREFILL_PORT=${5:-30002}
        ENCODER_URLS=$(build_encoder_urls $NUM_ENCODERS)
        echo "=== Starting Prefill Server on port $PREFILL_PORT ==="
        echo "=== Encoder URLs: $ENCODER_URLS ==="
        sglang serve \
            "${COMMON_ARGS[@]}" \
            --port $PREFILL_PORT \
            --disaggregation-mode prefill \
            --language-only \
            --encoder-urls $ENCODER_URLS \
            --encoder-transfer-backend mooncake
        ;;

    # ─── EPD: Decode ────────────────────────────────────────────────────
    decode)
        DECODE_PORT=${4:-30003}
        echo "=== Starting Decode Server on port $DECODE_PORT ==="
        sglang serve \
            "${COMMON_ARGS[@]}" \
            --port $DECODE_PORT \
            --disaggregation-mode decode
        ;;

    # ─── EPD: Router ────────────────────────────────────────────────────
    router)
        ROUTER_PORT=${4:-8000}
        PREFILL_HOST=${5:-127.0.0.1}
        PREFILL_PORT=${6:-30002}
        DECODE_HOST=${7:-127.0.0.1}
        DECODE_PORT=${8:-30003}
        echo "=== Starting Router on port $ROUTER_PORT ==="
        python -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill "http://$PREFILL_HOST:$PREFILL_PORT" \
            --decode "http://$DECODE_HOST:$DECODE_PORT" \
            --port $ROUTER_PORT
        ;;

    *)
        echo "Unknown role: $ROLE"
        echo ""
        echo "Roles:"
        echo "  encoder   - Single encoder (args: tp_size gpu_id port)"
        echo "  encoders  - Multi-encoder DP (args: tp_size num_encoders base_gpu_id)"
        echo "  language  - Language-only server (args: tp_size num_encoders)"
        echo "  prefill   - Prefill server (args: tp_size num_encoders prefill_port)"
        echo "  decode    - Decode server (args: tp_size decode_port)"
        echo "  router    - PD router (args: router_port prefill_host prefill_port decode_host decode_port)"
        exit 1
        ;;
esac
