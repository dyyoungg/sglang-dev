#!/bin/bash

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=True
export SGLANG_IO_WORKERS=8
export SGLANG_VLM_CACHE_SIZE_MB=16384 # 16 GiB
export SGLANG_MM_PRECOMPUTE_HASH=True
export SGLANG_PROCESSOR_CACHE_SIZE_MB=16384 # 16 GiB
export SGLANG_IMG_PREPROCESS_BACKEND="numexpr"
export SGLANG_IMG_TORCH_THREADS=16
export SGLANG_NUMEXPR_NUM_THREADS=32
export SGLANG_MM_BATCH_SHM="1"
export SGLANG_MM_SHM_COPY_THREADS=10 # for pro 5000，else 0
# export SGLANG_IMG_TORCH_THREADS=8
# export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
MODEL_PATH=${1:?Usage: bash launch.sh MODEL_PATH TP_SIZE PORT}
TP_SIZE=${2:?Usage: bash launch.sh MODEL_PATH TP_SIZE PORT}
PORT=${3:?Usage: bash launch.sh MODEL_PATH TP_SIZE PORT}
echo "Starting SGLang server..."
sglang serve \
   --model-path "$MODEL_PATH" \
   --tokenizer-path "$MODEL_PATH" \
   --model-impl sglang \
   --host 0.0.0.0 \
   --port "$PORT" \
   --log-level debug \
   --chunked-prefill-size 8192 \
   --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 8}' \
   --cuda-graph-max-bs-decode 8 \
   --tp-size "$TP_SIZE" \
   --enable-mfu-metrics \
   --enable-metrics \
   --enable-request-time-stats-logging \
   --show-time-cost \
   --enable-dynamic-batch-tokenizer \
   --cuda-graph-backend-prefill disabled \
   --enable-multimodal \
   --warmups "beebee_omni_warmup" \
   --mm-attention-backend fa2 \
   --max-image-bs 32 \
   --no-mm-use-compiled-rope \
   --mm-enable-dp-encoder \
   --sampling-backend legacy_top_kp
