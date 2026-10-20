"""Distribution and compatibility checks for the legacy sampler."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.legacy_sampler import (
    sample_legacy_top_kp,
    validate_legacy_sampling_request,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def make_info(top_ks, top_ps, temperatures=None):
    return SimpleNamespace(
        temperatures=torch.tensor(temperatures or [1.0] * len(top_ks)).view(-1, 1),
        top_ks=torch.tensor(top_ks, dtype=torch.int32),
        top_ps=torch.tensor(top_ps),
        is_all_greedy=all(k <= 1 for k in top_ks),
        need_top_k_sampling=any(k != TOP_K_ALL for k in top_ks),
        need_top_p_sampling=any(p != 1.0 for p in top_ps),
        need_min_p_sampling=False,
        sampling_seed=None,
    )


class TestLegacySampler(unittest.TestCase):
    def test_unsupported_options_rejected_at_request_ingress(self):
        """Reject unsupported requests without killing the scheduler's sampler."""
        for kwargs in ({"min_p": 0.1}, {"sampling_seed": 42}):
            with self.subTest(kwargs=kwargs):
                params = SamplingParams(**kwargs)
                with self.assertRaisesRegex(ValueError, "legacy_top_kp"):
                    validate_legacy_sampling_request(params, "legacy_top_kp")
                validate_legacy_sampling_request(params, "pytorch")

    def test_top_k_keeps_k_plus_one_and_restores_vocab_order(self):
        probs = torch.tensor([[0.05, 0.30, 0.10, 0.40, 0.15]])
        with patch("torch.multinomial", return_value=torch.tensor([[1]])) as sample:
            ids, logprobs = sample_legacy_top_kp(
                probs.log(), make_info([2], [1.0]), True
            )
        self.assertEqual(ids.item(), 1)
        torch.testing.assert_close(
            sample.call_args.args[0],
            torch.tensor([[0.40 / 0.85, 0.30 / 0.85, 0.15 / 0.85, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            logprobs.exp(),
            torch.tensor([[0.0, 0.30 / 0.85, 0.0, 0.40 / 0.85, 0.15 / 0.85]]),
        )

    def test_top_p_renormalizes_after_top_k_and_removes_crossing_token(self):
        probs = torch.tensor([[0.40, 0.30, 0.15, 0.10, 0.05]])
        ids, logprobs = sample_legacy_top_kp(probs.log(), make_info([2], [0.8]), True)
        self.assertEqual(ids.item(), 0)
        torch.testing.assert_close(logprobs.exp(), torch.tensor([[1.0, 0, 0, 0, 0]]))

    def test_top_p_only_keeps_first_token_below_threshold(self):
        probs = torch.tensor([[0.40, 0.30, 0.15, 0.10, 0.05]])
        for top_p, expected in [
            (0.8, [4 / 7, 3 / 7, 0, 0, 0]),
            (0.01, [1.0, 0, 0, 0, 0]),
        ]:
            with self.subTest(top_p=top_p):
                _, logprobs = sample_legacy_top_kp(
                    probs.log(), make_info([TOP_K_ALL], [top_p]), True
                )
                torch.testing.assert_close(logprobs.exp(), torch.tensor([expected]))

    def test_top_k_threshold_includes_ties(self):
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.2]])
        _, logprobs = sample_legacy_top_kp(probs.log(), make_info([2], [1.0]), True)
        torch.testing.assert_close(logprobs.exp(), probs)

    def test_greedy_logprobs_include_temperature(self):
        probs = torch.tensor([[0.2, 0.8]])
        ids, logprobs = sample_legacy_top_kp(
            probs.log(), make_info([1], [1.0], [0.5]), True
        )
        self.assertEqual(ids.item(), 1)
        torch.testing.assert_close(logprobs.exp(), torch.tensor([[1 / 17, 16 / 17]]))

    def test_mixed_batch_preserves_legacy_top_k_one_behavior(self):
        probs = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
        # Unlike the all-greedy shortcut, index 1 keeps two candidates here.
        # Use top_p=1 for this row to retain both after the top-p stage.
        _, logprobs = sample_legacy_top_kp(
            probs.log(), make_info([1, TOP_K_ALL], [1.0, 0.99]), True
        )
        torch.testing.assert_close(
            logprobs.exp(), torch.tensor([[2 / 3, 1 / 3, 0]] * 2)
        )

    def test_unfiltered_sampling_and_optional_logprobs(self):
        probs = torch.tensor([[0.2, 0.8]])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            expected = torch.multinomial(probs, num_samples=1).view(-1)
            torch.manual_seed(123)
            ids, logprobs = sample_legacy_top_kp(
                probs.log(), make_info([TOP_K_ALL], [1.0]), True
            )
        torch.testing.assert_close(ids, expected)
        torch.testing.assert_close(logprobs.exp(), probs)
        _, logprobs = sample_legacy_top_kp(probs.log(), make_info([2], [0.9]), False)
        self.assertIsNone(logprobs)

    def test_unsupported_sampling_parameters(self):
        for name, value, message in [
            ("need_min_p_sampling", True, "min_p"),
            ("sampling_seed", torch.tensor([42]), "deterministic"),
        ]:
            with self.subTest(name=name):
                info = make_info([2], [0.9])
                setattr(info, name, value)
                with self.assertRaisesRegex(ValueError, message):
                    sample_legacy_top_kp(torch.zeros(1, 5), info, True)


if __name__ == "__main__":
    unittest.main()
