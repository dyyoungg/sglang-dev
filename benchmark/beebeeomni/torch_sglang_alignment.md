# BeeBee dense checkpoint alignment investigation

Validated on 2026-10-15 with PyTorch 2.11.0+cu130, Transformers 5.9.0,
RTX PRO 5000 Blackwell, TP=1/2, BF16, and
`/mnt/afs/share/20260907_beebee_14B_veomni_28k`.
VeOmni imports resolve from `/mnt/afs/yangdeyu/VeOmni`.

## Current result after serving fixes

With the changes in this checkout, `--mm-attention-backend fa2` and
`--no-mm-use-compiled-rope`, both vision and audio encoder outputs are bitwise
equal to VeOmni in the tested TP=1 cases. Disabling compiled RoPE alone was not
sufficient before these code changes. The historical experiments below explain
the causes and retain their original measurements.

| Execution | Audio | Vision |
|---|---|---|
| TP=1, three cases per encoder | All exactly equal | All exactly equal |
| TP=2, sharded encoders | Nonzero differences | Nonzero differences |
| TP=2, replicated DP encoders, direct forward | All exactly equal | All exactly equal |
| TP=2, actual encoder DP dispatch | All exactly equal | All exactly equal |

For every exactly equal case, both mean and maximum absolute difference are 0.
Audio cases cover valid post-convolution lengths `[300]`, `[51,301]`, and
`[1500]`. Vision cases cover one image (`[1,32,48]`), two mixed-size images
(`[1,16,16]`, `[1,16,24]`, downsample ratios 4 and 16), and two temporal frames
(`[2,16,16]`). Seeds are 42, 123, and 2026. DP dispatch references use the same
per-request batch shapes as the dispatched execution, including an idle rank
for single-request cases.

The serving implementation now:

- Projects unquantized audio Q/K/V separately while preserving packed weight
  storage and TP loading. K uses a bias-free GEMM, matching VeOmni.
- Uses reference-style native RMSNorm, BF16 residual-add rounding, separate
  activation/multiply, and separate unquantized gate/up projections in the
  dense Qwen2.5 vision encoder. Splitting gate/up was required for the mixed-size
  image case even though the initial single-image case matched without it.
- Preserves FP32 rotary frequencies and cos/sin. ViT graph workspaces also
  preserve that dtype when allocated or grown.

Quantized projection paths retain their packed GEMMs and were not validated.
The subsequent [performance benchmark](encoder_alignment_performance.md) measures
the cost: at TP=1, vision latency increases 21–26% and audio latency 4–8%; with
TP=2 encoder DP, vision increases 20–22% and audio 3–8% at the same configuration.
The graph workspace dtype was checked, but actual CUDA graph capture/replay and
end-to-end language-model logits/generation have not been validated. These
random-input encoder checks do not establish equality for every input, batching
configuration, checkpoint, or hardware platform.

Reproduce the encoder checks (requires local VeOmni and CUDA):

```bash
export PYTHONPATH="/mnt/afs/yangdeyu/VeOmni:python:$PYTHONPATH"
/usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=1 \
  benchmark/beebeeomni/test_encoder_alignment.py
/usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmark/beebeeomni/test_encoder_alignment.py --dp-encoder
```

The test selects FA2 and disables compiled RoPE itself. For a multi-GPU launch,
the equivalent encoder options are:

```bash
--mm-attention-backend fa2 --no-mm-use-compiled-rope --mm-enable-dp-encoder
```

TP=2 sharded tests (omit `--dp-encoder`) report audio maximum differences
0.09765625–0.1015625 and vision maximum differences 0.125–1.0625 across these
cases. Splitting TP GEMMs and reducing partial sums changes BF16 rounding;
the current fixes do not make that execution bitwise equal to the TP=1 reference.
Replicated encoder DP retains a full encoder on each GPU and distributes requests.

Logs under `fp8_analysis/torch_sglang_alignment/`:
`beebee_alignment_fixed.log` (full checkpoint weight checks and initial encoder
comparison), `encoder_regression_tp1.log`, `encoder_regression_tp2.log`, and
`encoder_regression_tp2_dp.log` (six direct plus six dispatch comparisons).

## Checkpoint corruption in the reference model

VeOmni's `veomni/models/custom/audio_encoder/modeling_whisper.py` used
`module.weight.data.normal_()` and `module.bias.data.zero_()` in `_init_weights`.
Transformers 5 calls missing-key initialization after loading and guards supported
initialization functions using the parameter's `_is_hf_initialized` flag.
The direct `.data` writes bypass that guard and overwrite loaded audio weights.
An empty `missing_keys`/`unexpected_keys` report does not detect this corruption.

A one-layer audio model round trip reproduced 22/28 corrupted tensors. Replacing
the writes with `transformers.initialization.normal_` / `zeros_`, and guarding
embedding padding initialization, reduced this to 0/28. The fix is applied in the
VeOmni checkout. On the real checkpoint, both VeOmni and SGLang subsequently
matched every checkpoint tensor exactly (with SGLang packing/name mappings).

The local regression check covers FP32 and BF16 round trips plus a missing bias
whose initialization must preserve the loaded weight in the same module:

```bash
PYTHONPATH="/mnt/afs/yangdeyu/VeOmni:$PYTHONPATH" /usr/bin/python3 \
  benchmark/beebeeomni/test_whisper_checkpoint_loading.py
```

This is a local integration test requiring VeOmni, not an SGLang CI test.

## Historical diagnosis before serving fixes

Controlled experiments used the real encoder weights and identical random inputs.
Audio input: `[1,128,3000]`, valid post-convolution length 300.
Vision input: `[1536,1176]`, grid `[1,32,48]`.
These are encoder checks, not end-to-end generation or dataset quality results.

| Experiment | Mean absolute difference | Maximum absolute difference |
|---|---:|---:|
| Audio, merged QKV | 0.00495540 | 0.0859375 |
| Audio, three reference-style Q/K/V projections | 0 | 0 |
| Vision patch embedding / Conv3d | 0 | 0 |
| Vision, original diagnostic setup | 0.01510531 | 0.53125 |
| Vision, shared FP32 inv_freq | 0.01442781 | 0.359375 |
| Above + noncompiled RoPE | 0.01442781 | 0.359375 |
| Above + reference FP32 cos/sin | 0.01649001 | 0.53125 |
| Above + reference RMSNorm and residual rounding | 0.01307710 | 0.46875 |
| Above + native SiLU multiplication | 0.01198060 | 0.3125 |
| Above + separate gate/up projections | 0.01198060 | 0.3125 |
| Above + reference FlashAttention instead of default Triton | 0 | 0 |

The vision rows are cumulative substitutions. Individual substitutions need not
monotonically reduce the final error after 32 layers. They establish that the
remaining differences can be explained by the computation paths on this input;
they do not establish production accuracy or performance equivalence.
At that stage, the substitutions lived only in the diagnostic script. They have
since been implemented in the serving model as described above.
Artifacts: `fp8_analysis/torch_sglang_alignment/`.

## Initial SGLang fixes and comparison entry point

- Qwen2.5 BeeBee now forwards `mm_use_compiled_rope` to `VisionAttention`, as the
  Qwen3 vision branch already did. Previously the dense branch ignored the flag.
- Build the Qwen2.5 rotary frequency table directly: Transformers 5 changed the
  imported rotary module's argument from a sequence length to position IDs.
- The comparison entry point checks both models independently against checkpoint
  tensors and stops on weight mismatches before comparing forward outputs.
- Derive vision patch width from config (1176 here, not the Qwen3 width 1536).
- Initialize SGLang like its server loader, with BF16 default parameters and
  explicitly FP32 buffers preserved; blanket `.to(bfloat16)` incorrectly casts
  the nonpersistent RoPE buffer, which a state-dict comparison cannot detect.
- Expose model path, attention backend, and compiled RoPE options in the script.

```bash
export PYTHONPATH="/mnt/afs/yangdeyu/VeOmni:$PYTHONPATH"
/usr/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=1 \
  python/sglang/srt/models/beebee_omni.py \
  --model-path /mnt/afs/share/20260907_beebee_14B_veomni_28k \
  --no-mm-use-compiled-rope --mm-attention-backend fa2
```

This command now uses the corrected serving operators. Disabling compiled RoPE
alone did not establish alignment before those changes, and this dense Conv3d
result does not generalize to the MoE encoder.

## Historical launch backend follow-up: FA2

The saved launch script
`fp8_analysis/tp_prefill_0926_20261010/bf16/launch_bf16.sh` specifies
`--mm-attention-backend fa2 --no-mm-use-compiled-rope`.
Before the serving operator fixes, reran the entry point with those options, the same checkpoint,
seed 42, TP=1, and BF16 server-style initialization. This changes only the
attention backend relative to `beebee_alignment_final.log`; it does not apply
the diagnostic operator substitutions above.

| Encoder | Backend | Mean absolute difference | Maximum absolute difference |
|---|---|---:|---:|
| Vision | Default Triton | 0.012954 | 0.187500 |
| Vision | FA2 | 0.012457 | 0.250000 |
| Audio | Either setting | 0.004625 | 0.095703 |

Both models still match all checkpoint tensors exactly. FA2 alone slightly
reduces the vision mean error on this input but increases the maximum error;
it does not resolve the remaining normalization, activation and RoPE precision
differences. The Whisper encoder calls FlashAttention directly, so this vision
backend option does not change its QKV projection or output.

These measurements describe the earlier implementation, not the current code.
Log: `fp8_analysis/torch_sglang_alignment/beebee_alignment_fa2.log`.
