python benchmark_online_serve.py \
    --port 18004 \
    --tokenizer /mnt/afs/share/0926_llavaomni_30A3B_dynamic_downsample_st4_omni \
    --active-time 720 \
    --max-context-len 48192 \
    --patch-size 16 \
    --downsample-ratio 12 \
    --system-prompt-tokens 40000 \
    --seed 42 \
    --num-users 3
