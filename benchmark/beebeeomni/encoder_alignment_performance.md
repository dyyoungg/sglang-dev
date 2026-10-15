# BeeBee encoder alignment performance

Measured on 2026-10-15 using the same dense checkpoint as the alignment investigation.

## Method

- NVIDIA RTX PRO 5000 72GB Blackwell; GPU 1 for TP=1, GPUs 1 and 2 for TP=2. Devices were idle before testing; configurations ran sequentially.
- PyTorch 2.11.0+cu130, BF16, FA2, compiled RoPE disabled, eager execution (no encoder CUDA graph).
- Same real checkpoint weights loaded into both models, with exact state-dict equality checked. Inputs are the same six seeded inputs used by the alignment regression.
- Baseline sources: Git `12630057a38281fb893dfd70a2c8a0f2b099d9fd` plus only Transformers 5 rotary API compatibility and propagation of the noncompiled RoPE flag. Thus this isolates the final numerical-alignment operator changes at the same attention settings. No working-tree files were reverted.
- Each model receives 10 warmup calls per input, then 6 rounds × 10 calls. Before/after order alternates each round. Tables show the median of per-round mean elapsed milliseconds per call, synchronized with CUDA. TP=2 uses the slowest rank. GPU event timings and every sample are retained in JSON.
- DP results include the actual encoder dispatch and output gathering. Inputs and weights are already on GPU. Tokenization, media decoding/preprocessing, model loading, language-model inference, HTTP scheduling, and end-to-end request latency are excluded.
- These are fixed-input encoder latency measurements, not a production throughput benchmark. DP dispatch changes batch shapes; cross-configuration comparisons include that effect. GPU clocks were not locked.

## Same configuration: before vs after

### TP=1

| Input | Before (ms) | After (ms) | Added (ms) | Latency change |
|---|---:|---:|---:|---:|
| Audio, valid length 300 | 6.938 | 7.224 | +0.286 | +4.12% |
| Audio, mixed lengths 51 / 301 | 7.857 | 8.487 | +0.631 | +8.02% |
| Audio, valid length 1500 | 16.640 | 18.032 | +1.392 | +8.37% |
| Vision, single image 32 × 48 | 19.431 | 23.424 | +3.994 | +20.55% |
| Vision, two images 16 × 16 + 16 × 24 | 13.831 | 17.405 | +3.574 | +25.84% |
| Vision, two frames 16 × 16 | 13.669 | 16.798 | +3.129 | +22.89% |

### TP=2, sharded encoders

| Input | Before (ms) | After (ms) | Added (ms) | Latency change |
|---|---:|---:|---:|---:|
| Audio, valid length 300 | 7.351 | 8.509 | +1.158 | +15.75% |
| Audio, mixed lengths 51 / 301 | 8.151 | 9.217 | +1.067 | +13.09% |
| Audio, valid length 1500 | 18.302 | 19.422 | +1.120 | +6.12% |
| Vision, single image 32 × 48 | 20.695 | 24.835 | +4.140 | +20.00% |
| Vision, two images 16 × 16 + 16 × 24 | 16.206 | 19.839 | +3.633 | +22.42% |
| Vision, two frames 16 × 16 | 15.636 | 19.064 | +3.428 | +21.92% |

### TP=2, replicated encoder DP with dispatch

| Input | Before (ms) | After (ms) | Added (ms) | Latency change |
|---|---:|---:|---:|---:|
| Audio, valid length 300 | 7.248 | 7.437 | +0.189 | +2.61% |
| Audio, mixed lengths 51 / 301 | 7.007 | 7.297 | +0.290 | +4.14% |
| Audio, valid length 1500 | 16.908 | 18.251 | +1.342 | +7.94% |
| Vision, single image 32 × 48 | 19.518 | 23.446 | +3.928 | +20.13% |
| Vision, two images 16 × 16 + 16 × 24 | 13.772 | 16.833 | +3.061 | +22.22% |
| Vision, two frames 16 × 16 | 13.737 | 16.808 | +3.071 | +22.35% |

## Configuration change: previous TP=2 sharding to fixed encoder DP

This comparison includes both the numerical-alignment fixes and enabling `--mm-enable-dp-encoder`. The fixed DP configuration passed the encoder equality checks; the fixed sharded TP=2 configuration still has rounding differences.

| Input | Previous TP=2 (ms) | Fixed encoder DP (ms) | Latency change |
|---|---:|---:|---:|
| Audio, valid length 300 | 7.351 | 7.437 | +1.16% |
| Audio, mixed lengths 51 / 301 | 8.151 | 7.297 | -10.47% |
| Audio, valid length 1500 | 18.302 | 18.251 | -0.28% |
| Vision, single image 32 × 48 | 20.695 | 23.446 | +13.30% |
| Vision, two images 16 × 16 + 16 × 24 | 16.206 | 16.833 | +3.87% |
| Vision, two frames 16 × 16 | 15.636 | 16.808 | +7.49% |

Vision adds about 3–4 ms when holding the configuration constant. Audio adds about 0.2–1.4 ms in TP=1 / encoder DP; sharded TP=2 audio adds about 1.1 ms. The dominant vision changes remove fused normalization/activation and packed gate/up GEMMs. This run measures the combined change, not each operator separately.

Compared with the previous sharded TP=2 setup, fixed encoder DP adds 4–13% vision latency on these inputs. Short/long single-request audio latency is approximately unchanged (differences around 1% or less); the two-request mixed-length case is about 10% faster. DP avoids encoder TP communication and can distribute the two requests, but these results should not be generalized to other batch sizes or GPU interconnects.

## Reproduction and raw results

```bash
export PYTHONPATH="/mnt/afs/yangdeyu/VeOmni:python:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=1 \
  benchmark/beebeeomni/benchmark_encoder_alignment.py \
  --output fp8_analysis/torch_sglang_alignment/performance_tp1.json
CUDA_VISIBLE_DEVICES=1,2 /usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmark/beebeeomni/benchmark_encoder_alignment.py \
  --output fp8_analysis/torch_sglang_alignment/performance_tp2.json
CUDA_VISIBLE_DEVICES=1,2 /usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmark/beebeeomni/benchmark_encoder_alignment.py --dp-encoder \
  --output fp8_analysis/torch_sglang_alignment/performance_tp2_dp.json
```

The three JSON files contain source hashes, timings, and metadata. Corresponding `.log` files contain process output. `baseline_audio.py` and `baseline_vision.py` in the same directory preserve the exact compatibility-adjusted baseline source.
