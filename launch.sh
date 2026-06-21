#!/bin/bash

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=True
export SGLANG_IO_WORKERS=8
export SGLANG_VLM_CACHE_SIZE_MB=2048 # 2G
export SGLANG_MM_PRECOMPUTE_HASH=True
export SGLANG_PROCESSOR_CACHE_SIZE_MB=4096 # 4G
export SGLANG_IMG_PREPROCESS_BACKEND="numexpr"
export SGLANG_NUMEXPR_NUM_THREADS=16
# export SGLANG_IMG_TORCH_THREADS=8
# export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
MODEL_PATH=$1
TP_SIZE=$2
echo "Starting SGLang server..."
sglang serve \
   --model-path $MODEL_PATH \
   --tokenizer-path $MODEL_PATH \
   --model-impl sglang \
   --host 0.0.0.0 \
   --port 18003 \
   --log-level debug \
   --chunked-prefill-size 8192 \
   --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 8}' \
   --cuda-graph-max-bs 16 \
   --tp-size $TP_SIZE \
   --enable-mfu-metrics \
   --enable-metrics \
   --enable-request-time-stats-logging \
   --show-time-cost \
   --enable-dynamic-batch-tokenizer \
   --disable-piecewise-cuda-graph \
   --enable-multimodal \
   --warmups "beebee_omni_warmup" \
   --mm-attention-backend fa2 \
