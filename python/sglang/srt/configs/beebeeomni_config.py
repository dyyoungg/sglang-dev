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
from copy import deepcopy

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
        image_downsample_size: int  = 16,
        output_size: int            = 5120,
        return_hidden_states: bool  = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_projector_type   = image_projector_type
        self.image_downsample_ratio  = image_downsample_size
        self.output_size            = output_size
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
        audio_downsample_size: int  = 10,
        output_size: int             = 5120,
        return_hidden_states: bool   = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_downsample_ratio = audio_downsample_size
        self.audio_projector_type   = audio_projector_type
        self.output_size            = output_size
        self.return_hidden_states   = return_hidden_states


def _init_config(config_dict: Optional[Dict[str, Any] | PretrainedConfig]) -> Optional["PretrainedConfig"]:
    """
    Initialize a Hugging Face PretrainedConfig from a plain dictionary using AutoConfig.
    Returns a bare PretrainedConfig if input is None or the model_type is empty.
    """
    if config_dict is None:
        return PretrainedConfig()
    if isinstance(config_dict, PretrainedConfig):
        return config_dict

    config_copy = deepcopy(config_dict)
    model_type = config_copy.pop("model_type", "")
    if model_type == "":
        return PretrainedConfig()
    return AutoConfig.for_model(model_type, **config_copy)

# ── Encoder wrapper ───────────────────────────────────────────────────────────

class BeeBeeOmniEncoderConfig(PretrainedConfig):
  
    model_type = "llavaqwen2_encoder"
    sub_configs = {
        "image_config": BeeBeeVisionConfig,
        "audio_config": BeeBeeAudioConfig,
    }

    def __init__(
        self,
        image_config: Optional[Union[Dict[str, Any], BeeBeeVisionConfig]] = None,
        audio_config: Optional[Union[Dict[str, Any], BeeBeeAudioConfig]]  = None,
        encode_input:  bool  = True,
        encode_output: bool  = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Pass dedicated defaults so bare dicts get the right class
        self.image_config = _init_config(image_config)
        self.audio_config = _init_config(audio_config)
        self.encode_input      = encode_input
        self.encode_output     = encode_output
        self.initializer_range = initializer_range
        super().__init__(**kwargs)



class BeeBeeOmniConfig(PretrainedConfig):
  
    model_type = "llavaqwen2_omni"
    sub_configs  = {
        "encoder_config":    BeeBeeOmniEncoderConfig,
        "foundation_config": AutoConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ── Sub-configs ───────────────────────────────────────────────────
        encoder_config:    Optional[Union[Dict[str, Any], BeeBeeOmniEncoderConfig]] = None,
        foundation_config: Optional[Union[Dict[str, Any], PretrainedConfig]]=None,   
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
        if isinstance(encoder_config, BeeBeeOmniEncoderConfig):
           encoder_config = encoder_config
        elif isinstance(encoder_config, dict):
            encoder_config = BeeBeeOmniEncoderConfig(**encoder_config)
        else:
            # encoder_config is None → bare defaults
            encoder_config = BeeBeeOmniEncoderConfig()

        self.vision_config = encoder_config.image_config
   
        self.audio_config = encoder_config.audio_config

        # ── text_config ───────────────────────────────────────────────────
        # Mirrors Qwen2_5_VLConfig: if text_config is None we try to pick up
        # known text-backbone kwargs that were passed in as flat kwargs, then
        # delegate to AutoConfig so any Qwen2/Qwen3/… backbone works.
        if isinstance(foundation_config, PretrainedConfig):
            self.text_config = foundation_config
        elif isinstance(foundation_config, dict):
            model_type = foundation_config.get("model_type", "")
            self.text_config = (
                AutoConfig.for_model(model_type, **{k: v for k, v in foundation_config.items() if k != "model_type"})
                if model_type else PretrainedConfig(**foundation_config)
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
        kwargs.pop("architectures", None)  # 先把原有的删掉
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            architectures=["BeeBeeOmniForConditionalGeneration"],
            **kwargs,
        )

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        """Return the LLM backbone config (used by SGLang model loader)."""
        return self.text_config


# =============================================================================
# AutoConfig registration
# =============================================================================

AutoConfig.register("beebee_vision_model", BeeBeeVisionConfig,  exist_ok=True)
AutoConfig.register("beebee_audio_model",      BeeBeeAudioConfig,   exist_ok=True)
AutoConfig.register("llavaqwen2_omni",         BeeBeeOmniConfig,    exist_ok=True)

