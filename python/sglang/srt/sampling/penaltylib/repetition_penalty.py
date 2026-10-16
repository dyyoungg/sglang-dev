import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend


@torch.compile(dynamic=True, backend=get_compiler_backend())
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

        # Pre-fill penalty for custom_penalty_ids (only when input_penalty is enabled)
        for i, req in enumerate(self.orchestrator.reqs()):
            # print("req.sampling_params.input_penalty", req.sampling_params.input_penalty)
            if not getattr(req.sampling_params, 'input_penalty', False):
                continue
            ids = req.sampling_params.custom_penalty_ids
            if ids:
                unique_ids = torch.tensor(
                    list(set(ids)), dtype=torch.long, device=self.orchestrator.device
                )
                self.cumulated_repetition_penalties[i].scatter_(
                    dim=0,
                    index=unique_ids,
                    value=req.sampling_params.repetition_penalty,
                )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.cumulated_repetition_penalties.scatter_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.repetition_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)
