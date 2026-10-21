# Qwen 图像、视频准确率评测

`eval_qwen_vl.py` 复用本目录的 MCQ 数据、`reference_protocol.py` 中的题目与选项构造、答案提取规则，通过 `/v1/chat/completions` 使用 Qwen 原生 chat template。适用于本机共享文件系统上的 SGLang 服务。

与 BeeBee 的 `eval_single.py` 相比：图片只发送一次，不传 `image_downsample_ratios`；视频使用原生 `video_url`，由服务端采样、处理时间戳；MVBench 的 `start`/`end` 先通过 FFV1 无损裁剪；关闭 thinking，要求只输出选项。图片仍使用 ratio=1、factor=32 的原评测缩放和 JPEG 默认编码。

模型服务示例见 `outputs/benchmarks/20261020_qwen27b_multimodal/launch_server.sh`。视频配置为 2 FPS、最多 64 帧，单帧像素范围 100352..602112。服务与评测端须能读取相同的 `/tmp` 路径。

```bash
/usr/bin/python3 -m pip install --target /tmp/sglang_27b_eval_deps jiwer==4.0.0 rapidfuzz

PYTHONPATH=/tmp/sglang_27b_eval_deps /usr/bin/python3 -u \
  benchmark/beebeeomni/eval_qwen_vl.py \
  --output outputs/benchmarks/qwen27b_vl_sample100 \
  --datasets mmstar RealWorldQA ocr chartqa mvbench videomme videommmu \
  --limit 100 --seed 20261020 --concurrency 1
```

`--limit 0` 使用所选文件的全部样本。其他值按固定种子在每个文件内均匀无放回抽样，索引写入 `config.json`。同一命令可以断点续跑；`--retry-errors` 重试失败条目。更改模型、种子、样本量、脚本或并发时使用新的输出目录。

结果包含逐条 JSONL、逐数据集 summary、输入/脚本 SHA256、服务配置和 token 用量。主评分为正确选项比例，同时保存成功样本准确率、失败计错准确率、覆盖率、无效选项、截断及分类统计。沿用原协议，答案取回复中的第一个大写英文字母。

`request_latency_seconds` 从发起 HTTP 请求计时，包含服务端预处理、排队及推理，不含客户端下载、缩放和裁剪；后者单独记录为 `prepare_seconds`。非流式请求不测 TTFT；这些短答案的单并发耗时不能直接当成并发吞吐结果。

这些文件是本地评测集合，抽样分数不是官方完整 benchmark 分数。`ocr` 和 `chartqa` 都是本地改写的选择题，分别不是官方 OCRBench 分数和 ChartQA relaxed accuracy。视频只输入画面，不提供音频、字幕；VideoMMMU 中的 ASR 类题目仍会计分，并在分类统计中单独列出。

全量视频使用 `--limit 0`。本次两卡命令保存在输出目录的 `run_videos_full_gpu0.sh`（MVBench）、`run_videos_full_gpu1.sh`（VideoMME）和 `run_videos_full_mmmu.sh`（VideoMMMU，VideoMME 完成后复用 GPU1）。先并发 1，尾部阶段改为每服务并发 2，延迟按并发分别统计。`prepare_full_videos.py` 保存了原 100 条结果向全量目录迁移的过程；历史预测注明来源和服务地址。

评分复核需要注意：原协议只认最后一个与答案文本匹配的选项，重复选项可能导致等价答案被计错；答案解析取首个大写字母，也可能误取解释性文字中的字母。最终报告保留原规则主分数，并分别给出等价选项和回复格式的复核结果。原始回复始终保留。
