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

MODEL_PATH=$1
TP_SIZE=$2
OUTPUT_DIR=${3:-/mnt/afs/yangdeyu/dependency/sglang}

mkdir -p $OUTPUT_DIR

echo "Starting SGLang server with Nsight Systems profiling..."
echo "Model: $MODEL_PATH"
echo "TP: $TP_SIZE"
echo "Profile output: $OUTPUT_DIR"

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --delay=180 \
    --duration=30 \
    --force-overwrite=true \
    --output=${OUTPUT_DIR}/beebee_omni_profile \
    sglang serve \
        --model-path $MODEL_PATH \
        --tokenizer-path $MODEL_PATH \
        --model-impl sglang \
        --host 0.0.0.0 \
        --port 18003 \
        --log-level debug \
        --chunked-prefill-size 4096 \
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
        --enable-broadcast-mm-inputs-process \
        --warmups "beebee_omni_warmup" \
        --mm-attention-backend fa2
