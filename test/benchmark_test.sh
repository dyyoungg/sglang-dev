model_path=$1
send_rate=$2
python benchmark_serve.py \
    --host 0.0.0.0 \
    --port 18003 \
    --backend sglang \
    --tokenizer $model_path \
    --num-prompts 400 \
    --request-rate $send_rate \
    --prompt-len-min 3000 \
    --prompt-len-max 3000 \
    --max_output_token 64 \
    --num-images 0 \
    --image-width 644 \
    --image-height 364 \
    --num-audios 0 \
    --audio-sec 5 \
