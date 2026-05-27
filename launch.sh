#!/bin/bash

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=True
export SGLANG_IO_WORKERS=8
export SGLANG_VLM_CACHE_SIZE_MB=2048 # 2G
# export SGLANG_VIT_ENABLE_CUDA_GRAPH=1
MODEL_PATH=$1

echo "Starting SGLang server..."
sglang serve \
   --model-path $MODEL_PATH \
   --tokenizer-path $MODEL_PATH \
   --model-impl sglang \
   --host 0.0.0.0 \
   --port 18003 \
   --log-level debug \
   --enable-multimodal \
   --chunked-prefill-size 4096 \
   --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 8}' \
   --warmups "beebee_omni_warmup" \
   --cuda-graph-max-bs 8 \
   --tensor-parallel-size 2 \
   --enable-mfu-metrics \
   --enable-metrics \
   --enable-request-time-stats-logging \
   --show-time-cost \
   --enable-dynamic-batch-tokenizer \
   --enable-broadcast-mm-inputs-process \
   --mm-enable-dp-encoder \
