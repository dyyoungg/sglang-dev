"""Install opt-in runtime captures after warmup and graph initialization."""

import logging

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class RuntimeDiagnostics:
    def __init__(self, runner):
        # The logits recorder wraps the runner's sample method, so it needs the
        # runner itself rather than a copy of its configuration.
        if activation_dir := envs.SGLANG_FP8_ACTIVATION_DUMP_DIR.get():
            from sglang.srt.debug_utils.fp8_activation_capture import (
                configure_fp8_activation_capture,
                register_fp8_linear_activation_hooks,
            )

            parallel = get_parallel()
            if parallel.tp_size != 1 or parallel.pp_size != 1:
                raise ValueError("FP8 matched activation capture requires TP=PP=1")
            configure_fp8_activation_capture(
                activation_dir, tp_rank=parallel.tp_rank, pp_rank=parallel.pp_rank
            )
            count = register_fp8_linear_activation_hooks(runner.model)
            logger.info(
                "[FP8 diagnostics] Activation capture enabled: dir=%s, "
                "max_rows_per_request_module=%s, linear_hooks=%s",
                activation_dir,
                envs.SGLANG_FP8_ACTIVATION_MAX_TOKENS.get(),
                count,
            )

        if logits_dir := envs.SGLANG_OUTPUT_LOGITS_DIR.get():
            from sglang.srt.debug_utils.output_logits_capture import (
                install_output_logits_capture,
            )

            install_output_logits_capture(runner, logits_dir)
            logger.info("[Logits diagnostics] Capture enabled: %s", logits_dir)
