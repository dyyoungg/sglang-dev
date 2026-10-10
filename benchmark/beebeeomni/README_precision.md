# BeeBee 0815 BF16 / FP8 业务评测

模型：`/mnt/afs/share/0815_llavaomni_30A3B_qwen35encoder_qwen3audio_down2_mlp_zhall`。
所有命令从 SGLang 仓库根目录执行。

## 服务

两组使用相同 TP2 × DP2 拓扑，避免张量并行度不同影响精度对比。
GPU 0–3 为 BF16（18100），GPU 4–7 为默认在线 FP8（18101）。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_QUANTIZATION=none OMP_NUM_THREADS=8 \
bash launch_fp8.sh /mnt/afs/share/0815_llavaomni_30A3B_qwen35encoder_qwen3audio_down2_mlp_zhall 2 18100 \
  --dp-size 2 --dtype bfloat16 --mem-fraction-static 0.85 \
  --context-length 65536 --max-total-tokens 131072 \
  --max-running-requests 32 --max-image-bs 8 --random-seed 42 --log-level info

CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_QUANTIZATION=fp8 OMP_NUM_THREADS=8 \
bash launch_fp8.sh /mnt/afs/share/0815_llavaomni_30A3B_qwen35encoder_qwen3audio_down2_mlp_zhall 2 18101 \
  --dp-size 2 --dtype bfloat16 --mem-fraction-static 0.85 \
  --context-length 65536 --max-total-tokens 131072 \
  --max-running-requests 32 --max-image-bs 8 --random-seed 42 --log-level info
```

## 评测

Python 依赖：aiohttp、numpy、Pillow、soundfile、decord、AOSS client、jiwer 4.0.0、rapidfuzz。
媒体缓存默认位于 `/tmp/sglang_beebee_benchmark_cache`。AOSS 配置使用 `AOSS_FILE` 环境变量，
默认读取 `/mnt/afs/yangdeyu/aoss_ydy_game.conf`；不要复制凭据到结果目录。

```bash
# 每个数据集一条联通检查；必须使用独立输出目录。
PYTHONPATH=/tmp/sglang_beebee_eval_deps_mirror /usr/bin/python3 -u benchmark/beebeeomni/eval_precision.py \
  --output fp8_analysis/business_0815_smoke --limit 1

# 图片独立评测，64 条并发；每条请求同时发送给 BF16、FP8。
PYTHONPATH=/tmp/sglang_beebee_eval_deps_mirror /usr/bin/python3 -u benchmark/beebeeomni/eval_precision.py \
  --output fp8_analysis/business_0815_v1/images \
  --datasets mmstar RealWorldQA ocr ocr_generate chartqa --image-concurrency 64

# 音频独立评测，32 条并发。
PYTHONPATH=/tmp/sglang_beebee_eval_deps_mirror /usr/bin/python3 -u benchmark/beebeeomni/eval_precision.py \
  --output fp8_analysis/business_0815_v1/audio \
  --datasets aishell wenet online_badaudio --concurrency 32

# 视频单独运行，避免和图片抢占显存。均匀采样最多 64 帧。
PYTHONPATH=/tmp/sglang_beebee_eval_deps_mirror /usr/bin/python3 -u benchmark/beebeeomni/eval_precision.py \
  --output fp8_analysis/business_0815_v1/video \
  --datasets mvbench videomme videommmu --video-concurrency 4 --frames 64
```

默认分组依次执行。重复相同命令可断点续跑；加 `--retry-errors` 重试失败配对。
`config.json` 校验数据文件、提示词协议、预处理与采样参数；参数改变须使用新目录。
图片/视频并发可独立调整，不改变评分口径。

## 口径

- 冻结参考脚本 `VeOmni/scripts/eval_sglang.py` 的提示词、答案抽取与评分规则，见 `reference_protocol.py`。
- 原始数据固定顺序，不抽样；每条媒体仅预处理一次，两种精度使用相同 payload，保存 SHA256。
- 图像 ratio=1、size_factor=32，RGB/JPEG，按 BeeBee 要求复制为两帧。
- 视频读取实际帧数/FPS，在 start/end 范围内分段取中点，最多 64 帧，舍弃尾部奇数帧。
- temperature=0、repetition_penalty=1.05。ASR 最大输出 1024 token，OCR 生成 4096 token；
  选择题最大 64 token，在选项后的右括号停止。上下文为 65536 token。
- ASR：参考脚本的混合中英文 CER（英文数字连续串为一个单位），同时报告样本均值与 corpus CER。
- 图片/视频选择题：准确率；OCR 生成：归一化编辑距离，越低越好。
- 这里的 ChartQA 是本地选择题版本，不是标准 ChartQA relaxed accuracy；其他集合也以本地文件为准。
- 主要比较双方成功的相同样本，明确失败、长度截断和覆盖率；逐条 JSONL 保存原始输出及评分。
- CI 是样本配对 bootstrap 的描述性区间；同一媒体多问题相关性可能使区间偏窄。
- 此任务测业务精度。并发、预处理、缓存及服务负载影响延迟，不能把请求耗时直接当纯模型加速比。
