# coding=utf-8
import inspect
from typing import Any, Dict, Optional, Union
from copy import deepcopy

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig

class BeeBeeMoEVisionConfig(Qwen3_5MoeVisionConfig):

    model_type = "beebee_qwen35moe_vision_model"  # must match training checkpoint

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


class BeeBeeQwen3AudioConfig(PretrainedConfig):
    """
    Qwen3 Audio Encoder config + BeeBee projector fields.

    Architecture: 3x Conv2d (8x temporal downsample) + sinusoidal PE
                  + N transformer encoder layers + LN + projector.
    """

    model_type = "beebee_qwen3_audio_model"

    def __init__(
        self,
        # ── Encoder architecture ──────────────────────────────────────────
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        n_window: int = 50,
        n_window_infer: int = 800,
        downsample_hidden_size: int = 480,
        max_source_positions: int = 1500,
        conv_chunksize: int = 500,
        # ── Projector ─────────────────────────────────────────────────────
        audio_projector_type: str = "multi_conv",
        audio_downsample_size: int = 2,
        output_size: int = 5120,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.downsample_hidden_size = downsample_hidden_size
        self.max_source_positions = max_source_positions
        self.conv_chunksize = conv_chunksize
        # Projector fields
        self.audio_downsample_ratio = audio_downsample_size
        self.audio_projector_type = audio_projector_type
        self.output_size = output_size
        self.return_hidden_states = return_hidden_states


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

class BeeBeeMoEOmniEncoderConfig(PretrainedConfig):

    model_type = "llavaqwen2_encoder"
    sub_configs = {
        "image_config": BeeBeeMoEVisionConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        image_config: Optional[Union[Dict[str, Any], BeeBeeMoEVisionConfig]] = None,
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



class BeeBeeMoEOmniConfig(PretrainedConfig):
  
    model_type = "llavaqwen3moe_omni"
    sub_configs  = {
        "encoder_config":    BeeBeeMoEOmniEncoderConfig,
        "foundation_config": AutoConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ── Sub-configs ───────────────────────────────────────────────────
        encoder_config:    Optional[Union[Dict[str, Any], BeeBeeMoEOmniEncoderConfig]] = None,
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
        if isinstance(encoder_config, BeeBeeMoEOmniEncoderConfig):
           encoder_config = encoder_config
        elif isinstance(encoder_config, dict):
            encoder_config = BeeBeeMoEOmniEncoderConfig(**encoder_config)
        else:
            # encoder_config is None → bare defaults
            encoder_config = BeeBeeMoEOmniEncoderConfig()

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
            architectures=["BeeBeeMoEOmniForConditionalGeneration"],
            **kwargs,
        )

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        """Return the LLM backbone config (used by SGLang model loader)."""
        return self.text_config


AutoConfig.register("beebee_qwen35moe_vision_model", BeeBeeMoEVisionConfig,  exist_ok=True)
AutoConfig.register("beebee_audio_model",      BeeBeeAudioConfig,   exist_ok=True)
AutoConfig.register("beebee_qwen3_audio_model", BeeBeeQwen3AudioConfig, exist_ok=True)
AutoConfig.register("llavaqwen3moe_omni",         BeeBeeMoEOmniConfig,    exist_ok=True)


