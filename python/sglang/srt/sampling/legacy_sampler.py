"""Legacy top-k/top-p sampling, compatible with LightLLM ``triton_top_kp``.

Keep its zero-based top-k threshold and unshifted top-p mask intentionally:
changing either would change the distribution this backend reproduces.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def sample_legacy_top_kp(
    logits: torch.Tensor,
    sampling_info: "SamplingBatchInfo",
    return_logprob: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return token IDs and, if requested, logprobs in original vocabulary order.

    Filtered sampling returns logprobs of the truncated, renormalized
    distribution. Greedy and unfiltered sampling return full-distribution
    logprobs, matching the corresponding LightLLM branches.
    """
    if sampling_info.need_min_p_sampling:
        raise ValueError("legacy_top_kp does not support min_p > 0")
    if sampling_info.sampling_seed is not None:
        raise ValueError("legacy_top_kp does not support deterministic sampling")

    logits.div_(sampling_info.temperatures)
    if sampling_info.is_all_greedy:
        token_ids = logits.argmax(dim=-1)
        logprobs = torch.log(logits.softmax(dim=-1)) if return_logprob else None
        return token_ids, logprobs

    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        probs = logits.softmax(dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).view(-1)
        return token_ids, torch.log(probs) if return_logprob else None

    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    # LightLLM indexes at k, not k - 1: normally this keeps k + 1 tokens,
    # and also keeps all tokens tied at the threshold.
    top_ks = sampling_info.top_ks.clamp(max=logits.shape[-1] - 1).long()
    thresholds = sorted_logits.gather(1, top_ks.unsqueeze(1))
    sorted_logits.masked_fill_(sorted_logits < thresholds, -float("inf"))

    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    top_p_mask = cumulative_probs > sampling_info.top_ps.unsqueeze(1)
    # Intentionally remove the token crossing top_p, except the first token.
    top_p_mask[:, :1] = False
    sorted_logits.masked_fill_(top_p_mask, -float("inf"))

    probs = sorted_logits.softmax(dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1, replacement=False)
    token_ids = sorted_indices.gather(1, sampled_indices).view(-1)
    logprobs = None
    if return_logprob:
        logprobs = torch.empty_like(probs).scatter_(1, sorted_indices, torch.log(probs))
    return token_ids, logprobs
