"""Regression checks for the VeOmni Whisper Transformers 5 loader fix."""

import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from veomni.models.custom.llava_qwen3moe.modeling_audio_encoder import (
    BeeBeeAudioModelConfig,
    BeeBeeVLAudioModel,
)


class WhisperLoadingTest(unittest.TestCase):
    def test_checkpoint_roundtrip_and_partial_load(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                model = BeeBeeVLAudioModel(
                    BeeBeeAudioModelConfig(
                        d_model=16,
                        encoder_layers=1,
                        encoder_attention_heads=2,
                        encoder_ffn_dim=32,
                        max_source_positions=20,
                        output_size=16,
                    )
                ).to(dtype)
                with torch.no_grad():
                    for param in model.parameters():
                        param.uniform_(-0.5, 0.5)
                expected = {k: v.clone() for k, v in model.state_dict().items()}
                model.save_pretrained(directory)
                loaded, info = BeeBeeVLAudioModel.from_pretrained(
                    directory,
                    dtype=dtype,
                    output_loading_info=True,
                )
                self.assertFalse(info["missing_keys"])
                self.assertFalse(info["unexpected_keys"])
                for name, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]), name)
                # Missing bias must be initialized without changing the loaded
                # weight in the very same module.
                path = directory + "/model.safetensors"
                weights = load_file(path)
                del weights["conv1.bias"]
                save_file(weights, path)
                loaded, info = BeeBeeVLAudioModel.from_pretrained(
                    directory,
                    dtype=dtype,
                    output_loading_info=True,
                )
                self.assertEqual(set(info["missing_keys"]), {"conv1.bias"})
                self.assertEqual(torch.count_nonzero(loaded.conv1.bias).item(), 0)
                for name, value in loaded.state_dict().items():
                    if name != "conv1.bias":
                        self.assertTrue(torch.equal(value, expected[name]), name)


if __name__ == "__main__":
    unittest.main()
