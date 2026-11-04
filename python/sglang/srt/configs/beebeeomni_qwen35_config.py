"""Configuration for Llava Qwen3.5 Omni checkpoints."""

from transformers import AutoConfig, PretrainedConfig

from sglang.srt.configs.beebeeomni_moe_config import BeeBeeMoEVisionConfig, _init_config
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig


class BeeBeeOmniQwen35Config(PretrainedConfig):
    model_type = "llavaqwen35_omni"
    sub_configs = {"foundation_config": AutoConfig}

    def __init__(self, foundation_config=None, encoder_config=None, **kwargs):
        if isinstance(foundation_config, dict) and foundation_config.get("model_type") == "qwen3_5_text":
            self.foundation_config = Qwen3_5TextConfig(
                **{k: v for k, v in foundation_config.items() if k != "model_type"}
            )
        else:
            self.foundation_config = _init_config(foundation_config)
        encoder_config = encoder_config or {}
        image_config = encoder_config.get("image_config")
        audio_config = encoder_config.get("audio_config")
        self.vision_config = (
            image_config
            if isinstance(image_config, PretrainedConfig)
            else BeeBeeMoEVisionConfig(**{k: v for k, v in (image_config or {}).items() if k != "model_type"})
        )
        self.audio_config = _init_config(audio_config)
        if self.audio_config is not None and not hasattr(self.audio_config, "audio_downsample_ratio"):
            self.audio_config.audio_downsample_ratio = getattr(self.audio_config, "audio_downsample_size", 2)
        self.image_token_id = kwargs.pop("image_token_id", 151655)
        self.video_token_id = kwargs.pop("video_token_id", 151656)
        self.audio_token_id = kwargs.pop("audio_token_id", 151656)
        self.vision_start_token_id = kwargs.pop("vision_start_token_id", 151652)
        self.vision_end_token_id = kwargs.pop("vision_end_token_id", 151653)
        self.tie_word_embeddings = getattr(self.foundation_config, "tie_word_embeddings", False)
        kwargs["architectures"] = ["BeeBeeOmniQwen35ForConditionalGeneration"]
        super().__init__(**kwargs)

    @property
    def text_config(self):
        return self.foundation_config

    def get_text_config(self, decoder=None, encoder=None):
        return self.foundation_config


AutoConfig.register("llavaqwen35_omni", BeeBeeOmniQwen35Config, exist_ok=True)
