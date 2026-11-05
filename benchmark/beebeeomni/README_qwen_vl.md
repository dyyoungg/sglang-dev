# Qwen 图像、视频准确率评测

`eval_qwen_vl.py` 复用本目录的 MCQ 数据和 `reference_protocol.py` 题目/答案规则，统一支持官方 Qwen 模型和 BeeBee Omni 自定义模型。通过 `--model-type` 选择模型协议。

`--model-type qwen` 使用 `/v1/chat/completions`、官方 Qwen chat template 和原生 `video_url`；Qwen 服务端处理视频帧及时间信息，图片只发送一次。

`--model-type beebee` 使用 BeeBee 的 `/generate` 接口和手工构造的完整 prompt。视频按 VeOmni midpoint 规则取最多 64 帧，不能将帧复制成对：相邻两帧直接组成一对；静态图片则复制两份组成 pair。默认所有帧放在一个 `<|vision_start|>...<|vision_end|>` 区段中；加 `--pair-vision-segments` 后，每一对帧各自被一对 vision token 包围。模板为 system 与 vision 输入、user 问题、空的 assistant think 段；加 `--veomni-format` 时在其后追加 `My best option: (`，并在选项结束处停止生成。

模型服务示例见 `outputs/benchmarks/20261020_qwen27b_multimodal/launch_server.sh`。视频配置为 2 FPS、最多 64 帧，单帧像素范围 100352..602112。服务与评测端须能读取相同的 `/tmp` 路径。

```bash
/usr/bin/python3 -m pip install --target /tmp/sglang_27b_eval_deps jiwer==4.0.0 rapidfuzz

PYTHONPATH=/tmp/sglang_27b_eval_deps /usr/bin/python3 -u \
  benchmark/beebeeomni/eval_qwen_vl.py \
  --output outputs/benchmarks/qwen27b_vl_sample100 \
  --model-type qwen --model /mnt/afs/share/Qwen3.8-27B \
  --datasets mmstar RealWorldQA ocr chartqa mvbench videomme videommmu \
  --limit 100 --seed 20261020 --concurrency 1
```

BeeBee 自定义模型示例（按 VeOmni MCQ 模板、ratio=4、每对视频帧独立 vision segment）：

```bash
/usr/bin/python3 -u benchmark/beebeeomni/eval_qwen_vl.py \
  --output outputs/benchmarks/beebee_qwen35_mvbench_ratio4 \
  --model-type beebee \
  --model /mnt/afs/share/llava_qwen38_27B_qwen38encoder_qwen3audio_base_qwen38_projector \
  --datasets mvbench --limit 0 --concurrency 1 \
  --downsample-ratio 4 --veomni-resize --veomni-format \
  --pair-vision-segments
```

BeeBee prompt 模板（vision block repeated once per adjacent frame pair when `--pair-vision-segments` is enabled）：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|vision_start|><image><image><|vision_end|>...
<|im_start|>user
{question and options}<|im_end|>
<|im_start|>assistant
<think>

</think>

My best option: (
```

官方 Qwen 不经过上面的手工模板；它仍通过 `/v1/chat/completions` 传 system/user messages，由 Qwen 官方 chat template 处理视频和 thinking。对官方 Qwen 不要传 `--pair-vision-segments`、`--veomni-format` 或 `--veomni-resize`。

`--limit 0` 使用所选文件的全部样本。其他值按固定种子在每个文件内均匀无放回抽样，索引写入 `config.json`。同一命令可以断点续跑；`--retry-errors` 重试失败条目。更改模型、种子、样本量、脚本或并发时使用新的输出目录。

结果包含逐条 JSONL、逐数据集 summary、输入/脚本 SHA256、服务配置和 token 用量。主评分为正确选项比例，同时保存成功样本准确率、失败计错准确率、覆盖率、无效选项、截断及分类统计。沿用原协议，答案取回复中的第一个大写英文字母。

`request_latency_seconds` 从发起 HTTP 请求计时，包含服务端预处理、排队及推理，不含客户端下载、缩放和裁剪；后者单独记录为 `prepare_seconds`。非流式请求不测 TTFT；这些短答案的单并发耗时不能直接当成并发吞吐结果。

这些文件是本地评测集合，抽样分数不是官方完整 benchmark 分数。`ocr` 和 `chartqa` 都是本地改写的选择题，分别不是官方 OCRBench 分数和 ChartQA relaxed accuracy。视频只输入画面，不提供音频、字幕；VideoMMMU 中的 ASR 类题目仍会计分，并在分类统计中单独列出。

全量视频使用 `--limit 0`。本次两卡命令保存在输出目录的 `run_videos_full_gpu0.sh`（MVBench）、`run_videos_full_gpu1.sh`（VideoMME）和 `run_videos_full_mmmu.sh`（VideoMMMU，VideoMME 完成后复用 GPU1）。先并发 1，尾部阶段改为每服务并发 2，延迟按并发分别统计。`prepare_full_videos.py` 保存了原 100 条结果向全量目录迁移的过程；历史预测注明来源和服务地址。

评分复核需要注意：原协议只认最后一个与答案文本匹配的选项，重复选项可能导致等价答案被计错；答案解析取首个大写字母，也可能误取解释性文字中的字母。最终报告保留原规则主分数，并分别给出等价选项和回复格式的复核结果。原始回复始终保留。
