#!/bin/bash

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=True
export SGLANG_IO_WORKERS=8
export SGLANG_VLM_CACHE_SIZE_MB=16384 # 2G
export SGLANG_MM_PRECOMPUTE_HASH=True
export SGLANG_PROCESSOR_CACHE_SIZE_MB=16384 # 4G
export SGLANG_IMG_PREPROCESS_BACKEND="numexpr"
export SGLANG_IMG_TORCH_THREADS=16
export SGLANG_NUMEXPR_NUM_THREADS=32
export SGLANG_MM_BATCH_SHM="1"
export SGLANG_MM_SHM_COPY_THREADS=10 # for pro 5000，else 0
# export SGLANG_IMG_TORCH_THREADS=8
# export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
MODEL_PATH=$1
TP_SIZE=$2
PORT=$3
shift 3
QUANTIZATION="${SGLANG_QUANTIZATION:-fp8}"
# A common shell/config mistake is setting SGLANG_QUANTIZATION=None.  Bash
# treats that as a real value, so it would otherwise be passed to argparse as
# `--quantization None`.  Keep lowercase `none` available for the BF16
# reference run, but treat Python-style `None` as the default FP8 setting.
if [[ "$QUANTIZATION" == "None" || "$QUANTIZATION" == "NONE" ]]; then
   echo "[FP8 diagnostics] SGLANG_QUANTIZATION=$QUANTIZATION is not a valid CLI value; defaulting to fp8"
   QUANTIZATION="fp8"
fi
echo "Starting SGLang server..."
echo "[FP8 diagnostics] quantization=$QUANTIZATION; FP8 GEMM backend=auto; MoE backend=auto; KV cache dtype=auto"
echo "[FP8 diagnostics] See worker logs for effective config, component quantization methods, and selected kernels."
if [[ "$QUANTIZATION" == "fp8" ]]; then
   echo "[FP8 diagnostics] SGLANG_FP8_IGNORED_LAYERS=${SGLANG_FP8_IGNORED_LAYERS:-<none>}"
fi
CAPTURE_ARGS=()
if [[ -n "${SGLANG_FP8_ACTIVATION_DUMP_DIR:-}" ]]; then
   CAPTURE_ARGS+=(--disable-radix-cache)
   mkdir -p "$SGLANG_FP8_ACTIVATION_DUMP_DIR"
   export SGLANG_FP8_ACTIVATION_MAX_TOKENS="${SGLANG_FP8_ACTIVATION_MAX_TOKENS:-1024}"
   echo "[FP8 diagnostics] Runtime activation capture enabled: dir=$SGLANG_FP8_ACTIVATION_DUMP_DIR, max_rows_per_request_module=$SGLANG_FP8_ACTIVATION_MAX_TOKENS"
fi
QUANT_ARGS=()
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]]; then
   QUANT_ARGS+=(--quantization "$QUANTIZATION")
fi
# --fp8-gemm-backend controls block/MXFP8 Linear dispatch, not all FP8 kernels.
# --kv-cache-dtype fp8_e4m3 is independent of weight/activation quantization.
# Request-gated prefill capture (TP=1). Use a new directory for each run:
#   SGLANG_FP8_ACTIVATION_DUMP_DIR=./fp8_analysis/my_run \
#     bash launch_fp8.sh "$MODEL_PATH" 1 18004
#   API_TEST_CAPTURE_DIR=./fp8_analysis/my_run API_TEST_MODE=images /usr/bin/python3 api_test.py
#   API_TEST_CAPTURE_DIR=./fp8_analysis/my_run API_TEST_MODE=video /usr/bin/python3 api_test.py
# Analyze each group with analyze_fp8_captured_activations.py (see its docstring).
# Selective BF16 (default remains the original FP8 configuration):
#   SGLANG_FP8_IGNORED_LAYERS=image_encoder.blocks.26.mlp.linear_fc2 \
#     bash launch_fp8.sh "$MODEL_PATH" 1 18004
# Full BF16 reference: SGLANG_QUANTIZATION=none bash launch_fp8.sh ...
# Extra server arguments can follow PORT. For output-distribution diagnostics,
# see benchmark/kernels/fused_moe_triton/compare_output_logprobs.py.
sglang serve \
   --model-path $MODEL_PATH \
   --tokenizer-path $MODEL_PATH \
   --model-impl sglang \
   --host 0.0.0.0 \
   --port $PORT \
   --log-level debug \
   --chunked-prefill-size 16384 \
   --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 8}' \
   --cuda-graph-max-bs 8 \
   --tp-size $TP_SIZE \
   --enable-mfu-metrics \
   --enable-metrics \
   --enable-request-time-stats-logging \
   --show-time-cost \
   --enable-dynamic-batch-tokenizer \
   --disable-piecewise-cuda-graph \
   --enable-multimodal \
   --skip-server-warmup \
   --mm-attention-backend fa2 \
   --max-image-bs 32 \
   "${QUANT_ARGS[@]}" "${CAPTURE_ARGS[@]}" "$@"
