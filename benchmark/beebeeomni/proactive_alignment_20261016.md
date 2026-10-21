# BeeBee proactive next-token alignment

Input: `112012-2026-08-15-12:33:35_generate.json`, proactive assistant turn 31.
The prompt was rebuilt from the conversation history using the SGLang debug
replay format. Images were ordered by timestamp and filename, paired in the
same two-frame groups as the serving processor, and resized to the serving
dimensions before preprocessing.

The request contained 440 image placeholders and 440 image files. Both sides
expanded the prompt to 7,762 tokens, including 6,822 visual embedding slots.
The checkpoint used for the strict comparison was:

`/mnt/afs/jiayi/code/VeOmni/ckpt/261015_llavaomni_14B_beebee_game_bs10_predata_bw_balance/checkpoints/global_step_578/hf_ckpt`

SGLang was running with `--mm-attention-backend fa2` and
`--no-mm-use-compiled-rope` on port 18004. VeOmni used Transformers 5.9.0,
BF16, and `veomni_flash_attention_2_with_sp`.

With `temperature=0`, `top_k=1`, `top_p=1`, and
`repetition_penalty=1`, VeOmni's next-token top-k was:

```text
我去 24.25
这   23.25
这个 21.875
哇   21.375
我   21.25
好   21.125
```

The SGLang response was:

```text
这
top logprobs: 这=-1.0248929, 我去=-1.7748929, 哇=-2.6498930,
             这个=-3.0248930, 我=-3.1498928, 好=-3.2748930
```

The result is unchanged in VeOmni when `use_cache=False` is changed to
`use_cache=True`, so this is not a Torch-side KV-cache discrepancy. The first
token ranking differs in the SGLang multimodal prefill/LM path even after the
vision encoder fixes; the close Torch scores mean the numerical difference is
large enough to flip the sampled first token for this checkpoint.

For comparison, using the other `/mnt/afs/share/20260907...` checkpoint made
both implementations select `这`; that directory is not tensor-identical to
the jiayi checkpoint and must not be used for this strict comparison.

## Full prefill logits comparison (2026-10-16)

The strict rerun uses the same jiayi checkpoint, turn 31 prompt, and all 440
JPEG images from the saved SGLang request. Torch image tensors were regenerated
from those JPEG bytes using the checkpoint's `Qwen25VLImageProcessorOptimized`
and `patch_reshape_method="torch"`, matching the SGLang processor path. The
expanded prompt has 7,762 positions and 6,822 image slots. Both sides used BF16
weights, FA2, and no compiled RoPE; the comparison is over the raw, pre-sampler
full-vocabulary logits (151,680 entries). SGLang ran TP=1 with CUDA graphs and
overlap disabled. The request was repeated after flushing the radix cache, and
reported zero cached tokens.

| Metric | Torch vs SGLang |
| --- | ---: |
| Max absolute logit difference | 1.5000 |
| Mean absolute logit difference | 0.19365 |
| Median absolute logit difference | 0.15625 |
| 99th percentile absolute difference | 0.66895 |
| Relative L2 difference | 6.38% |
| Cosine similarity | 0.997971 |
| Top-1 / top-5 overlap | 1 / 5 |
| Top-20 / top-100 overlap | 19 / 92 |

With identical JPEG inputs, both implementations select token `这` (ID 43288).
The leading logits are close: SGLang has `这=23.75`, `我去=23.00`; Torch has
`这=23.50`, `我去=22.875`. The old comparison used uncompressed source PIL
images on the Torch side and JPEG-encoded images on the SGLang side; its mean
absolute difference was 0.470 and it flipped top-1 to `我去` in Torch. That
older number is not an apples-to-apples model-forward comparison.

The cache-enabled and cache-flushed SGLang results have the same top-1 and
nearly identical error statistics, so radix-cache reuse of the first three
prompt tokens does not explain this discrepancy. The exact-input run aligns
the ranking but logits are not numerically identical; remaining differences
are in the BF16 multimodal/decoder execution paths and need layer-level
activation comparison to localize further.
