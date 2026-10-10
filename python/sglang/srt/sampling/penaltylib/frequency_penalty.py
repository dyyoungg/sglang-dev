import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    # def _prepare(self):
    #     self.cumulated_frequency_penalties = torch.zeros(
    #         (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
    #         dtype=torch.float32,
    #         device=self.orchestrator.device,
    #     )

    #     self.frequency_penalties = (
    #         torch.tensor(
    #             data=[
    #                 req.sampling_params.frequency_penalty
    #                 for req in self.orchestrator.reqs()
    #             ],
    #             dtype=torch.float32,
    #             device=self.orchestrator.device,
    #         )
    #     ).unsqueeze_(1)
    
    def _prepare(self):
        # 1. 依然初始化累加矩阵 (默认全 0)
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        # 2. 初始化单次惩罚的步长值
        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

        for i, req in enumerate(self.orchestrator.reqs()):
            tokens = getattr(req.sampling_params, "custom_penalty_ids", None)
            req_id = getattr(req, "rid", i)
            
            if tokens is not None and len(tokens) > 0:
                # 把 tokens 转换成 tensor 以便做索引
                token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.orchestrator.device)
                
                # 获取该请求设定的惩罚值 (比如 1.5)
                penalty_val = req.sampling_params.frequency_penalty
                
                # 这样在模型还没生成任何字的时候，这些 token 的惩罚就已经生效了
                self.cumulated_frequency_penalties[i, token_tensor] = penalty_val
                
                # print(f"[DEBUG Penalty] 请求 {req_id}: 已在一开始对特定的 {len(tokens)} 个 Token 施加了初始惩罚值: {penalty_val}")


    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.cumulated_frequency_penalties.scatter_add_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.frequency_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_frequency_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("frequency_penalties", "cumulated_frequency_penalties"):
            if hasattr(self, name):
                delattr(self, name)
