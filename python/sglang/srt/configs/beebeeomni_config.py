# coding=utf-8
"""
Configuration for BeeBeeLlavaQwen2ForConditionalGeneration.

Follows the same flat sub-config pattern as Qwen2_5_VLConfig:
  vision_config  ->  BeeBeeVisionConfig   (extends Qwen2_5_VLVisionConfig)
  audio_config   ->  BeeBeeAudioConfig    (extends WhisperConfig)
  text_config    ->  any PretrainedConfig (e.g. Qwen2Config)
"""

import inspect
from typing import Any, Dict, Optional, Union

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig


class BeeBeeVisionConfig(Qwen2_5_VLVisionConfig):
    """
    Qwen2.5-VL ViT config + BeeBee DynamicAvgPool projector fields.

    Extra args vs Qwen2_5_VLVisionConfig
    ─────────────────────────────────────
    image_projector_type  : "dynamic_avgpool"
    image_downsample_size : adaptive-avgpool scale factor  (default 8)
    output_size           : projected dim = LLM hidden size (e.g. 6144)
    freeze_vision_merger  : freeze patch-merger during training
    train_vision_projector: train projector only
    return_hidden_states  : return intermediate ViT states
    """

    model_type = "beebee_vision_model"  # must match training checkpoint

    def __init__(
        self,
        image_projector_type: str   = "dynamic_avgpool",
        image_downsample_ratio: int  = 8,
        out_hidden_size: int            = 6144,
        return_hidden_states: bool  = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_projector_type   = image_projector_type
        self.image_downsample_ratio  = image_downsample_ratio
        self.out_hidden_size            = out_hidden_size
        self.return_hidden_states   = return_hidden_states


class BeeBeeAudioConfig(WhisperConfig):
    """
    Whisper encoder config + BeeBee AudioConvUpScale projector fields.

    Extra args vs WhisperConfig
    ────────────────────────────
    audio_projector_type  : "conv_channel_upscale"
    audio_downsample_ratio: total time-compression factor applied by the projector
                            (training used the name audio_downsample_size; both accepted)
    output_size           : projected dim = LLM hidden size (e.g. 6144)
    train_audio_projector : train projector only
    return_hidden_states  : return intermediate encoder states
    """

    model_type = "beebee_audio_model"  # must match training checkpoint

    def __init__(
        self,
        audio_projector_type: str    = "conv_channel_upscale",
        audio_downsample_ratio: int  = 10,
        output_hidden_size: int             = 6144,
        return_hidden_states: bool   = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_downsample_ratio = audio_downsample_ratio
        self.audio_projector_type   = audio_projector_type
        self.audio_downsample_ratio = audio_downsample_ratio
        self.output_hidden_size            = output_hidden_size
        self.return_hidden_states   = return_hidden_states


class BeeBeeOmniConfig(PretrainedConfig):
  
    model_type = "beebee_omni"
    sub_configs = {
        "vision_config": BeeBeeVisionConfig,
        "audio_config":  BeeBeeAudioConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ── Sub-configs ───────────────────────────────────────────────────
        vision_config: Optional[Union[Dict[str, Any], BeeBeeVisionConfig]] = None,
        audio_config:  Optional[Union[Dict[str, Any], BeeBeeAudioConfig]]  = None,
        text_config:   Optional[Union[Dict[str, Any], PretrainedConfig]]   = None,

        # ── Vision token IDs ─────────────────────────────────────────────
        image_token_id:        int = 151655,
        video_token_id:        int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id:   int = 151653,
        # ── Misc ──────────────────────────────────────────────────────────
        tie_word_embeddings: bool  = False,
        initializer_range: float   = 0.02,
        **kwargs,
    ):
        # ── vision_config ─────────────────────────────────────────────────
        if isinstance(vision_config, dict):
            self.vision_config = BeeBeeVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = BeeBeeVisionConfig()
        else:
            self.vision_config = vision_config

        # ── audio_config ──────────────────────────────────────────────────
        if isinstance(audio_config, dict):
            self.audio_config = BeeBeeAudioConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = BeeBeeAudioConfig()
        else:
            self.audio_config = audio_config

        # ── text_config ───────────────────────────────────────────────────
        # Mirrors Qwen2_5_VLConfig: if text_config is None we try to pick up
        # known text-backbone kwargs that were passed in as flat kwargs, then
        # delegate to AutoConfig so any Qwen2/Qwen3/… backbone works.
        if isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            model_type = text_config.get("model_type", "")
            self.text_config = (
                AutoConfig.for_model(model_type, **{k: v for k, v in text_config.items() if k != "model_type"})
                if model_type else PretrainedConfig(**text_config)
            )
        else:
            # text_config is None – absorb any LLM kwargs that were passed flat
            # (same trick Qwen2_5_VLConfig uses for hub-saved flat dicts)
            known_llm_extras = ["rope_scaling", "rope_theta"]
            text_params = (
                list(inspect.signature(PretrainedConfig.__init__).parameters.keys())
                + known_llm_extras
            )
            absorbed = {k: kwargs.pop(k) for k in list(kwargs) if k in text_params}
            self.text_config = PretrainedConfig(**absorbed) if absorbed else PretrainedConfig()

        # ── token IDs ─────────────────────────────────────────────────────
        self.image_token_id        = image_token_id
        self.video_token_id        = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id   = vision_end_token_id
  
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range   = initializer_range

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            architectures=kwargs.pop(
                "architectures",
                ["BeeBeeOmniForConditionalGeneration"],
            ),
            **kwargs,
        )

    def get_text_config(self) -> PretrainedConfig:
        """Return the LLM backbone config (used by SGLang model loader)."""
        return self.text_config


# =============================================================================
# AutoConfig registration
# =============================================================================

AutoConfig.register("beebee_vision_model", BeeBeeVisionConfig,  exist_ok=True)
AutoConfig.register("beebee_audio_model",      BeeBeeAudioConfig,   exist_ok=True)
AutoConfig.register("beebee_omni",         BeeBeeOmniConfig,    exist_ok=True)



if __name__ == "__main__":
    import tempfile
    from transformers import Qwen2Config

    image_cfg = BeeBeeVisionConfig(
        hidden_size=1280,
        num_heads=16,
        depth=32,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        image_downsample_size=8,
        output_size=6144,
    )

    audio_cfg = BeeBeeAudioConfig(
        d_model=1280,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        max_source_positions=1500,
        audio_downsample_ratio=4,
        output_size=6144,
    )

    text_cfg = Qwen2Config(
        vocab_size=151936,
        hidden_size=6144,
        intermediate_size=16384,
        num_hidden_layers=28,
        num_attention_heads=48,
        num_key_value_heads=8,
        tie_word_embeddings=False,
    )

    cfg = BeeBeeOmniConfig(
        vision_config=image_cfg,
        audio_config=audio_cfg,
        text_config=text_cfg,
    )

    # round-trip
    with tempfile.TemporaryDirectory() as d:
        cfg.save_pretrained(d)
        loaded = BeeBeeOmniConfig.from_pretrained(d)

    assert loaded.model_type                          == "beebee_omni"
    assert loaded.vision_config.model_type            == "beebee_vision_model"
    assert loaded.audio_config.model_type             == "beebee_audio_model"
    assert loaded.vision_config.image_downsample_ratio == 8      # alias
    assert loaded.audio_config.audio_downsample_ratio  == 4
    assert loaded.image_token_id                      == 151655
    assert loaded.get_text_config().hidden_size       == 6144
    assert loaded.architectures == ["BeeBeeOmniForConditionalGeneration"]

    # training-side alias: audio_downsample_size= should map to audio_downsample_ratio
    compat = BeeBeeAudioConfig(audio_downsample_size=10, output_size=6144)
    assert compat.audio_downsample_ratio == 10

    print("All assertions passed ✓")
    print(f"  vision  model_type : {loaded.vision_config.model_type}")
    print(f"  audio   model_type : {loaded.audio_config.model_type}")
    print(f"  text    model_type : {loaded.text_config.model_type}")
    print(f"  architectures      : {loaded.architectures}")