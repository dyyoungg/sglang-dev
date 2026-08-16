model_path=$1
send_rate=$2
mode=$3
python benchmark_serve.py \
    --host 0.0.0.0 \
    --port 18003 \
    --backend sglang \
    --tokenizer /mnt/afs/share/llava_qwen2_14B-veomni-down16 \
    --num-prompts 200 \
    --request-rate $send_rate \
    --prompt-len-min 3000 \
    --prompt-len-max 3000 \
    --max_output_token 64 \
    --num-images 16 \
    --image-width 644 \
    --image-height 364 \
    --num-audios 0 \
    --audio-sec 5 \
    --mode $mode
