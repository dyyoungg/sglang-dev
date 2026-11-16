#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch_model.sh [options] [-- additional SGLang arguments]

  --model-path PATH       Model checkpoint (default: 1106 BeeBee checkpoint)
  --tokenizer-path PATH   Tokenizer path (default: model path)
  --profile PROFILE       beebee (default) or generic
  --gpu DEVICES           CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1 (default: 0)
  --port PORT             Model service port (default: 18006)
  --host HOST             Listen address (default: 127.0.0.1)
  --python PATH           Python executable (default: /usr/bin/python3)
  --help                  Show this help

Example: launch_model.sh --gpu 1 --port 18007 -- --mem-fraction-static 0.8
Use --profile generic for other models. Arguments after -- are passed to SGLang.
EOF
}
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

model_path=/mnt/afs/yangdeyu/VeOmni/ckpt/1106_llava_27B_gametext_omnisft_adamw_1e5_8k/hf_ckpt
tokenizer_path=
profile=beebee
gpu=0
port=18006
host=127.0.0.1
python=/usr/bin/python3
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    --model-path|--tokenizer-path|--profile|--gpu|--port|--host|--python)
      option=$1
      if (($# < 2)) || [[ -z $2 || $2 == --* ]]; then
        echo "Missing value for $option" >&2; exit 2
      fi
      case "$option" in
        --model-path) model_path=$2 ;;
        --tokenizer-path) tokenizer_path=$2 ;;
        --profile) profile=$2 ;;
        --gpu) gpu=$2 ;;
        --port) port=$2 ;;
        --host) host=$2 ;;
        --python) python=$2 ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1 (use -- before SGLang arguments)" >&2; exit 2 ;;
  esac
done

[[ $profile == beebee || $profile == generic ]] || { echo "Invalid profile: $profile" >&2; exit 2; }
[[ $port =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1 && 10#$port <= 65535)) || { echo "Invalid port: $port" >&2; exit 2; }
[[ $gpu =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU list: $gpu" >&2; exit 2; }
tokenizer_path=${tokenizer_path:-$model_path}
if [[ $profile == beebee && ! -d $model_path ]]; then
  echo "BeeBee checkpoint directory not found: $model_path" >&2; exit 2
fi

export CUDA_VISIBLE_DEVICES=$gpu
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-/tmp/sglang-webui-${port}-${gpu//,/-}}
args=(--model-path "$model_path" --tokenizer-path "$tokenizer_path" --host "$host" --port "$port")
if [[ $profile == beebee ]]; then
  export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-True}
  export SGLANG_VLM_CACHE_SIZE_MB=${SGLANG_VLM_CACHE_SIZE_MB:-1024}
  export SGLANG_PROCESSOR_CACHE_SIZE_MB=${SGLANG_PROCESSOR_CACHE_SIZE_MB:-1024}
  args+=(--tp-size 1 --dtype bfloat16 --mem-fraction-static 0.88
    --max-total-tokens 65536 --max-mamba-cache-size 24 --max-running-requests 4
    --chunked-prefill-size 2048 --cuda-graph-max-bs-decode 4
    --cuda-graph-backend-prefill disabled
    --mm-process-config '{"video":{"fps":2,"max_frames":64,"min_pixels":100352,"max_pixels":602112}}'
    --skip-server-warmup --log-level info)
fi
exec "$python" -m sglang.launch_server "${args[@]}" "$@"
