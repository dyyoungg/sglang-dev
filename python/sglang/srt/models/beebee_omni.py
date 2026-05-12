
import logging
import re
from functools import partial, lru_cache
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np

from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)
from transformers.models.whisper.configuration_whisper import WhisperConfig
from flash_attn import flash_attn_func, flash_attn_varlen_func

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.configs.beebeeomni_config import BeeBeeOmniConfig, BeeBeeAudioConfig, BeeBeeVisionConfig
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, permute_inv
from sglang.srt.multimodal.mm_utils import run_dp_sharded_beebee_vision_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


class WhisperAttention(nn.Module):
    """
    Multi-head self-attention for Whisper encoder layers.
    Uses Flash Attention 2 exclusively (no FA3, no eager fallback, no SP).
    """
 
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
 
        # Whisper encoder uses bias=False on k_proj, bias=True on q/v/out
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q:  Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q:  Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Two code paths:
          - Padded batch  (cu_seqlens_q is None) : hidden_states [B, T, D]
          - Varlen batch  (cu_seqlens_q provided): hidden_states [total_tokens, D]
        """
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
 
        if cu_seqlens_q is None:
            # ── padded-batch path ─────────────────────────────────────────
            B, T, _ = hidden_states.shape
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
            out = flash_attn_func(q, k, v, softmax_scale=self.scaling, causal=False)
            out = out.reshape(B, T, self.embed_dim)
        else:
            # ── varlen (packed) path ──────────────────────────────────────
            N = hidden_states.shape[0]
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_heads, self.head_dim)
            v = v.view(N, self.num_heads, self.head_dim)
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=self.scaling,
                causal=False,
            )
            out = out.reshape(N, self.embed_dim).contiguous()
 
        return self.out_proj(out)
 
 
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        d = config.d_model
        self.self_attn = WhisperAttention(d, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(d, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q:  Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q:  Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens_q=cu_seqlens_q,  cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,  max_seqlen_kv=max_seqlen_kv,
        )
        hidden_states = residual + hidden_states
 
        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
 
        # fp16 overflow guard
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp, max=clamp)
 
        return hidden_states
 
 
class WhisperEncoder(nn.Module):
    """
    Whisper audio encoder (inference-only).
 
    Input  : input_features [B, num_mel_bins, T]   mel spectrogram, channels-first
             input_seq_lens [B]  optional – valid mel-frame count per sample
                                            (enables varlen flash-attn; avoids
                                             attending to padding frames)
 
    Output : hidden_states  [B, T_out, d_model]
             where T_out = ceil(T / 2)  (conv2 has stride=2)
 
    Checkpoint prefix: audio_encoder.encoder.*
    """
 
    NUM_MEL_BINS = 128  # standard Whisper mel bins
 
    def __init__(self, config: WhisperConfig):
        super().__init__()
        d = config.d_model
        self.max_source_positions = config.max_source_positions
 
        # Two 1-D convolutions: conv2 has stride=2 → 2× time downsampling
        self.conv1 = nn.Conv1d(self.NUM_MEL_BINS, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, d)
 
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d)
 
    # ── device / dtype helpers ───────────────────────────────────────────
 
    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype
 
    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device
 
    # ── forward ─────────────────────────────────────────────────────────
 
    def forward(
        self,
        input_features: torch.Tensor,                    # [B, num_mel_bins, T]
        input_seq_lens: Optional[torch.Tensor] = None,   # [B] int64
    ) -> torch.Tensor:
        # ── Conv feature extraction ──────────────────────────────────────
        # [B, D, T] after both convolutions;  conv2 halves T
        x = F.gelu(self.conv1(input_features))
        x = F.gelu(self.conv2(x))
        T_out = x.shape[-1]
        x = x.permute(0, 2, 1)                        # [B, T_out, D]
        x = x + self.embed_positions.weight[:T_out]   # fixed sinusoidal PE
 
        # ── Build varlen args when seq lengths are supplied ───────────────
        # input_seq_lens are counts of *mel frames* (before this encoder).
        # After conv1 (kernel=3,stride=1) + conv2 (kernel=3,stride=2) the
        # output length per sample is  ceil((mel_len - 1) / 2 + 1) ≈ mel_len // 2.
        # We compute it the same way Whisper's _get_feat_extract_output_lengths does:
        #   out_len = (mel_len - 1) // 2 + 1
        if input_seq_lens is not None:
            conv_lens = ((input_seq_lens.long() - 1) // 2 + 1).clamp(max=T_out)
            B = x.shape[0]
            cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=x.device)
            cu_seqlens[1:] = conv_lens.cumsum(0).to(torch.int32)
            max_seqlen = int(conv_lens.max().item())
 
            # Pack valid frames into flat [total_tokens, D]
            chunks = [x[i, : int(conv_lens[i].item()), :] for i in range(B)]
            x = torch.cat(chunks, dim=0)
        else:
            cu_seqlens = None
            max_seqlen = None
 
        # ── Transformer layers ────────────────────────────────────────────
        for layer in self.layers:
            x = layer(
                x,
                cu_seqlens_q=cu_seqlens,   cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,   max_seqlen_kv=max_seqlen,
            )
 
        # ── Unpack back to padded batch ───────────────────────────────────
        if cu_seqlens is not None:
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            x = torch.split(x, [int(s) for s in split_sizes], dim=0)
            x = pad_sequence(x, batch_first=True)   # [B, max_valid_T, D]
 
        x = self.layer_norm(x)
        return x   # [B, T_out, d_model]
 
 
class AudioConvUpScaleProjector(nn.Module):
    """
    Maps Whisper hidden states to the LLM embedding dimension.
 
    Steps
    -----
    1. Conv1d (kernel=2, stride=2)  : another 2× time compression
       → combined with Whisper's conv2, total compression = audio_downsample_ratio
    2. Reshape  : group `linear_compress_ratio` consecutive frames into one vector
    3. 2-layer MLP  : project to `out_hidden`
 
    Checkpoint keys
    ---------------
    audio_encoder.mm_projector.afeat_1d_conv.*
    audio_encoder.mm_projector.linear1.*
    audio_encoder.mm_projector.linear2.*
    """
 
    def __init__(self, encoder_hidden: int, out_hidden: int, downsample_ratio: int = 4):
        super().__init__()
        self.audio_downsample_ratio = downsample_ratio
        # conv already does 2×; MLP handles the remainder
        self.linear_compress_ratio = downsample_ratio // 2
 
        self.afeat_1d_conv = nn.Conv1d(
            encoder_hidden, encoder_hidden, kernel_size=2, stride=2, padding=0
        )
        self.linear1 = nn.Linear(encoder_hidden * self.linear_compress_ratio, out_hidden, bias=True)
        self.gelu    = nn.GELU()
        self.linear2 = nn.Linear(out_hidden, out_hidden, bias=True)
 
    def forward(
        self,
        x: torch.Tensor,              # [B, T, encoder_hidden]
        feature_lengths: List[int],   # valid *mel-frame* counts per sample
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns
        -------
        out        : [N_total_audio_tokens, out_hidden]  (all samples concatenated)
        num_tokens : per-sample token counts
        """
        # Conv1d: [B, D, T] -> [B, D, T//2]
        x = self.afeat_1d_conv(x.transpose(1, 2)).transpose(1, 2)
        B, seq_len, D = x.shape
 
        # Pad seq_len to a multiple of linear_compress_ratio
        remainder = seq_len % self.linear_compress_ratio
        if remainder:
            pad_len = self.linear_compress_ratio - remainder
            x = torch.cat(
                [x, torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)], dim=1
            )
            seq_len = seq_len + pad_len
 
        # Group frames: [B, seq_len // lcr, D * lcr]
        x = x.reshape(B, seq_len // self.linear_compress_ratio,
                       D * self.linear_compress_ratio)
 
        # MLP projection
        x = self.linear2(self.gelu(self.linear1(x)))   # [B, compressed_T, out_hidden]
 
        # Compute valid output-token count per sample and flatten
        # feature_lengths are original mel-frame counts;
        # total downsampling = audio_downsample_ratio
        num_tokens = [
            math.ceil(l / self.audio_downsample_ratio) for l in feature_lengths
        ]
        parts = [x[i, : num_tokens[i], :] for i in range(B)]
        out = torch.cat(parts, dim=0)   # [total_tokens, out_hidden]
        return out, num_tokens
 
 
# =============================================================================
# Part 3 – Full Audio Encoder
# Checkpoint prefix: audio_encoder.*
# =============================================================================
 
class BeeBeeAudioEncoder(nn.Module):
    """
    Full audio encoding pipeline:
 
      input_features [B, num_mel_bins, T]
          -> WhisperEncoder           -> [B, T//2, whisper_d_model]
          -> AudioConvUpScaleProjector -> [N_total_tokens, llm_hidden]
 
    The module mirrors the checkpoint layout of BeeBeeVLAudioModel:
      audio_encoder.encoder.*        <- WhisperEncoder
      audio_encoder.mm_projector.*   <- AudioConvUpScaleProjector
    """
 
    def __init__(self, audio_config: BeeBeeAudioConfig, out_hidden_size: int):
        super().__init__()
        whisper_hidden: int    = audio_config.d_model
        downsample_ratio: int  = getattr(audio_config, "audio_downsample_ratio", 10)
 
        self.encoder      = WhisperEncoder(audio_config)
        self.audio_projector = AudioConvUpScaleProjector(
            encoder_hidden=whisper_hidden,
            out_hidden=out_hidden_size,
            downsample_ratio=downsample_ratio,
        )
 
    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.dtype
 
    @property
    def device(self) -> torch.device:
        return self.encoder.device
 
    def forward(
        self,
        input_features: torch.Tensor,    # [B, num_mel_bins, T]  channels-first
        feature_lengths: List[int],       # valid mel-frame count per sample
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns
        -------
        audio_embeds : [N_total_audio_tokens, out_hidden]
        num_tokens   : per-sample audio token counts
        """
        input_features = input_features.to(self.device, self.dtype)
 
        # Convert mel lengths to Whisper-encoder output lengths for varlen attention
        # WhisperEncoder internally computes conv_len = (mel_len - 1) // 2 + 1
        input_seq_lens = torch.tensor(feature_lengths, dtype=torch.long, device=self.device)
 
        # WhisperEncoder: [B, mel, T] -> [B, T//2, d_model]
        whisper_out = self.encoder(input_features, input_seq_lens=input_seq_lens)
 
        # AudioConvUpScaleProjector: [B, T//2, d_model] -> [N_tokens, llm_hidden]
        audio_embeds, num_tokens = self.mm_projector(whisper_out, feature_lengths)
        return audio_embeds, num_tokens




class Qwen2_5_VLMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.hidden_act = hidden_act
        if self.hidden_act == "silu":
            self.act = SiluAndMul()
        else:
            base_act = ACT2FN[self.hidden_act]

            def _act_fn(x: torch.Tensor) -> torch.Tensor:
                gate, up = x.chunk(2, dim=-1)
                return base_act(gate) * up

            self.act = _act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act(gate_up)
        x_down, _ = self.down_proj(x)
        return x_down


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-6,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_ws=None,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            output_ws=output_ws,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class BeeBeeVLPatchMerger(nn.Module):
    """
    Norm-only merger: RMSNorm → reshape patches into spatial_merge_unit groups.
    Weight: image_encoder.merger.ln_q.*
    """

    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, B, D = x.shape
        x2d = x.reshape(-1, D)
        x2d = self.ln_q(x2d)
        x2d = x2d.view(-1, self.hidden_size)
        return x2d
    

@lru_cache(maxsize=200)
def _adaptive_pool_size(h: int, w: int, scale: int = 20) -> Tuple[int, int]:
    r = 1.0 / math.sqrt(scale)
    return max(1, int(np.round(h * r))), max(1, int(np.round(w * r)))

class DynamicAvgPoolProjector(nn.Module):
    def __init__(
        self,
        encoder_hidden: int,
        out_hidden: int,
        downsample_ratio: int = 16,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.mm_downsample_ratio = downsample_ratio
        self.encoder_hidden = encoder_hidden
        self.merge_size = 2 

        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()

        self.mlp = nn.ModuleList([
            ColumnParallelLinear(
                encoder_hidden,
                encoder_hidden,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("mlp.0", prefix),
                tp_size=tp_size,
                tp_rank=tp_rank,
            ),
            nn.GELU(),
            RowParallelLinear(
                encoder_hidden,
                out_hidden,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("mlp.2", prefix),
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        ])

    def forward(
        self,
        images_feature: torch.Tensor,
        images_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            images_feature: [N_merged_patches, encoder_hidden]
            images_thw: [n_images, 3] -> (t, h_merged, w_merged)
        """
    
        outputs = []
        seq_len_list: List[int] = []
        start = 0
        hidden_size = images_feature.shape[-1]

        for thw in images_thw:
            t, h_m, w_m = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            length = t * h_m * w_m
            
            img_seq = images_feature[start : start + length]
            start += length

            # [t, h_m, w_m, D] -> [D, t, h_m, w_m]
            img_feat = img_seq.view(t, h_m, w_m, -1).permute(3, 0, 1, 2)
            
            Mh, Nw = _adaptive_pool_size(h_m, w_m, scale=self.mm_downsample_ratio)
            
            pooled = F.adaptive_avg_pool2d(img_feat, (Mh, Nw))
            
            # [D, t, Mh, Nw] -> [t, Mh, Nw, D] -> [t*Mh*Nw, D]
            pooled = pooled.permute(1, 2, 3, 0).contiguous().view(-1, hidden_size)

            tokens_per_frame = pooled.shape[0] // t
            seq_len_list.extend([tokens_per_frame] * t)
            outputs.append(pooled)

        hidden_states = torch.cat(outputs, dim=0)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        
        hidden_states, _ = mlp_fc1(hidden_states)
        hidden_states = mlp_act(hidden_states)
        hidden_states, _ = mlp_fc2(hidden_states)

        return hidden_states, seq_len_list

class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [S, B, context_dim]
        S, B, D = x.shape
        x2d = x.reshape(-1, D)
        x2d = self.ln_q(x2d)  # RMSNorm expects 2D
        x2d = x2d.view(-1, self.hidden_size)  # group into spatial_merge_unit
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x2d)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class BeeBeeVisionTransformer(nn.Module, RotaryPosMixin):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        downsample_ratio: int = 16,
        use_data_parallel: bool = False,
        max_context_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = ((vision_config.intermediate_size + 7) // 8) * 8
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for i in range(depth)
            ]
        )

        self.merger = BeeBeeVLPatchMerger(
            context_dim=hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )

        self.embed_dim = hidden_size * (self.spatial_merge_unit**2)
        self.mm_projector = DynamicAvgPoolProjector(
            encoder_hidden=self.embed_dim,
            out_hidden=vision_config.out_hidden_size,
            downsample_ratio=downsample_ratio,
            quant_config=quant_config,
            prefix=add_prefix("mm_projector", prefix),
        )

        # Resource prepared for vit cuda graph
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.max_context_len = max_context_len
        self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()

        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = None
        if self.enable_cg:
            self.cuda_graph_runner = ViTCudaGraphRunner(self)

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if self.enable_cg:
            return self.forward_with_cuda_graph(x, grid_thw)

        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Move window_index to the same device as x before using it to index x
        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)

        # Ensure rotary_pos_emb is on the same device/dtype as x
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        # After building position_embeddings, make sure both cos and sin are on the same device/dtype as the attention input
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")
            cu_window_seqlens = cu_window_seqlens.to("cpu")
        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            fullatt_indexes = self.fullatt_block_indexes
            if isinstance(fullatt_indexes, torch.Tensor):
                fullatt_indexes = fullatt_indexes.tolist()
            if layer_num in fullatt_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)
        x = x[reverse_indices, :]

        proj_thw = grid_thw.clone()
        proj_thw[:, 1] = grid_thw[:, 1] // self.spatial_merge_size
        proj_thw[:, 2] = grid_thw[:, 2] // self.spatial_merge_size
        features, seq_lens = self.mm_projector(x, proj_thw)
        return features

    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        # patch token num
        seq_len, _ = x.size()

        # [G, M, hidden]
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]  # [G, M, hidden]
        x = x.reshape(seq_len, -1)  # [seq_len, hidden]

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        # After building position_embeddings, make sure both cos and sin are on
        # the same device/dtype as the attention input
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )

        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

       
        x = self.cuda_graph_runner.run(
                x=x,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                output_indices=reverse_indices, # Graph Runner 内部会自动帮你做 x = x[reverse_indices, :]
            )

        proj_thw = grid_thw.clone()
        proj_thw[:, 1] = grid_thw[:, 1] // self.spatial_merge_size
        proj_thw[:, 2] = grid_thw[:, 2] // self.spatial_merge_size
        
        features, seq_lens = self.mm_projector(x, proj_thw)
        
        return features



class BeeBeeOmniForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # To ensure correct weight loading and mapping.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attn.qkv": "attn.qkv_proj",
        },
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )

    def __init__(
        self,
        config: BeeBeeOmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.text_config = self.config.text_config
        self.vision_config = self.config.vision_config
        self.audio_config = self.config.audio_config
        self.model = Qwen2Model(
            self.text_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.text_config.vocab_size,
                    self.text_config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()
        
        self.image_encoder = BeeBeeVisionTransformer(
            self.vision_config,
            norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("image_encoder", prefix),
            use_data_parallel=self.use_data_parallel,
            max_context_len=self.config.max_position_embeddings,
        )

  
        self.audio_encoder = BeeBeeAudioEncoder(
            audio_config=self.audio_config,
            out_hidden_size=self.text_config.hidden_size,
        )
        
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.image_encoder.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        expected_dim = getattr(self.image_encoder, "embed_dim", -1)

        raw_patch_dim = 1176

        if pixel_values.dim() == 2:
            current_dim = pixel_values.shape[-1]
            if current_dim == expected_dim:
                return pixel_values
            if current_dim != raw_patch_dim:
                return pixel_values

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if self.use_data_parallel:
            return run_dp_sharded_beebee_vision_model(
                self.image_encoder, 
                pixel_values, 
                image_grid_thw.tolist(), 
                merge_size=2, 
                downsample_ratio=getattr(self.vision_config, "mm_downsample_ratio", 16)
            )
        else:
            image_embeds = self.image_encoder(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    _lora_pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)$"
    )

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self._lora_pattern.match(module_name))

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        if self.audio_encoder is None:
            raise ValueError("Audio tokens present but audio_encoder was not initialized.")

        if not items:
            return torch.empty(0, device=self.audio_encoder.device)

        all_mel_chunks = []
        all_chunk_lengths = []
        chunks_per_item = []

        for item in items:
            # mel_chunks shape: [num_chunks, 128, 3000]
            mel_chunks = item.feature.to(self.audio_encoder.device, self.audio_encoder.dtype)
            chunk_lengths = item.model_specific_data["audio_length"]

            all_mel_chunks.append(mel_chunks)
            all_chunk_lengths.extend(chunk_lengths) # 注意这里用 extend 打平
            chunks_per_item.append(len(chunk_lengths)) # 记录这条音频贡献了多少个 chunk

        # batched_mel_chunks shape: [total_chunks, 128, 3000]
        batched_mel_chunks = torch.cat(all_mel_chunks, dim=0)

        chunk_embeds, chunk_token_nums = self.audio_encoder(batched_mel_chunks, all_chunk_lengths)

        # 4. 分发与重组阶段：按原始归属切分回连续的音频长序列
        all_audio_embeds = []
        current_chunk_idx = 0

        for num_chunks in chunks_per_item:
            valid_chunk_embeds = []
            
            # 取出属于当前音频的 num_chunks 个计算结果
            for _ in range(num_chunks):
                token_len = chunk_token_nums[current_chunk_idx]
                if token_len > 0:
                    valid_chunk_embeds.append(chunk_embeds[current_chunk_idx, :token_len, :])
                current_chunk_idx += 1
            
        
            if valid_chunk_embeds:
                single_audio_embed = torch.cat(valid_chunk_embeds, dim=0)
            else:
                single_audio_embed = torch.empty((0, chunk_embeds.shape[-1]), device=chunk_embeds.device)
                
            all_audio_embeds.append(single_audio_embed)

        return torch.cat(all_audio_embeds, dim=0)


    def post_process(
        self,
        inputs_embeds,
        modalities: List[Modality],
        embeddings: List[torch.Tensor],
        indices: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Placeholder for post_process
        new_embeddings = []
        for i, (modality, embedding, index) in enumerate(
            zip(modalities, embeddings, indices)
        ):
            if embedding is None or index is None:
                continue

            new_embeddings.append(embedding)
        return new_embeddings, forward_batch

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Custom weight loader for SGLang/vLLM backend.
        Handles PP layer filtering, TP tensor sharding, and module name mapping.
        """
        # 定义需要合并 (Concat) 的参数映射 (用于张量并行 TP)
        # 格式: (代码中的合并参数名, 权重文件中的独立参数名, shard_id)
        stacked_params_mapping = [
            # LLM Attention Q/K/V 合并
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # LLM & Vision MLP Gate/Up 合并 (SwiGLU 结构)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "model")
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # 3. GPTQ/AWQ 量化模型兼容
            # 跳过权重中残留但当前网络结构不需要的 bias
            if name.endswith(".bias") and name not in params_dict:
                continue

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)

            # 5. Vision Encoder 命名映射
            if name.startswith("image_encoder."):
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

            # 6. Audio Encoder 结构层级注入
            # safetensor 叫 audio_encoder.conv1，代码中叫 audio_encoder.encoder.conv1
            if name.startswith("audio_encoder.") and not name.startswith("audio_encoder.audio_projector."):
                name = name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

            # 7. 处理分片参数合并 (Stacked Params)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                
                if f".{weight_name}." not in name and not name.endswith(f".{weight_name}"):
                    continue
                
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                    
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 传入 shard_id 告诉 SGLang 的 ParallelLinear 应该把这个切片拼到哪个位置
                weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break
            
            if is_stacked:
                continue

       
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                
                logger.warning(f"Skipped unmapped safetensor key: {name}")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = BeeBeeOmniForConditionalGeneration


if __name__ == "__main__":
    import os
    import glob
    import torch
    from safetensors import safe_open
    
  
    MODEL_PATH = "mnt/afs/share/llava_qwen2_14B-veomni-down16" 
    
   
    from sglang.srt.server_args import ServerArgs
    dummy_args = ServerArgs(model_path=MODEL_PATH, mm_enable_dp_encoder=False)
   

    print("Initializing models...")
    
    from veomni.models.custom.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM as TrainBeeBeeOmni
    
    
    config = BeeBeeOmniConfig.from_pretrained(MODEL_PATH)
    sglang_model = BeeBeeOmniForConditionalGeneration(config).to(torch.bfloat16).cuda()
    sglang_model.eval()

    # 初始化原始模型
    train_model = TrainBeeBeeOmni.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda()
    train_model.eval()

    # ---------------------------------------------------------
    # 3. 为 SGLang 模型加载 Safetensors 权重
    # ---------------------------------------------------------
    print("Loading weights into SGLang model...")
    safetensors_files = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No .safetensors files found in {MODEL_PATH}")

    weights_iterator = []
    for f in safetensors_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                weights_iterator.append((k, st.get_tensor(k)))
    
    # 调用我们刚刚重写的 load_weights 逻辑
    sglang_model.load_weights(weights_iterator)
    
    # 将加载好权重的 SGLang 模型移到 GPU
    sglang_model = sglang_model.cuda()

    # ---------------------------------------------------------
    # 4. 测试 1：视觉编码器 (Image Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Vision Encoder ---")
    # 构造 dummy vision 输入 (参照 Qwen2.5-VL 规范)
    # 假设一张图被 patchify 成了 256 个 patch，每个 patch 维度 1176 (3通道 * 14 * 14 * 2合并)
    dummy_pixel_values = torch.randn(256, 1176, dtype=torch.bfloat16, device="cuda")
    # 对应网格 T=1, H=16, W=16 (16*16 = 256)
    dummy_image_grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        # SGLang 视觉前向
        sgl_vision_out = sglang_model.image_encoder(dummy_pixel_values, dummy_image_grid_thw)
        
        # 原始模型视觉前向 (此处根据你原始模型的方法名可能需要微调，比如 train_model.visual)
        # 假设原始模型的视觉提取入口和 SGLang 保持一致
        orig_vision_out = train_model.image_encoder(dummy_pixel_values, dummy_image_grid_thw)

    v_max_diff = torch.max(torch.abs(sgl_vision_out - orig_vision_out)).item()
    v_mean_diff = torch.mean(torch.abs(sgl_vision_out - orig_vision_out)).item()
    print(f"Vision Output Shape: {sgl_vision_out.shape}")
    print(f"Vision Max Diff:  {v_max_diff:.6f}")
    print(f"Vision Mean Diff: {v_mean_diff:.6f}")
    if v_max_diff < 1e-3:
        print("✅ Vision Encoder weights loaded perfectly!")
    else:
        print("❌ Vision Encoder has significant precision differences.")

    # ---------------------------------------------------------
    # 5. 测试 2：音频编码器 (Audio Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Audio Encoder ---")
    # 构造 dummy audio 输入 (参照 Whisper 规范)
    # 假设输入为 1 条音频，包含 1 个 chunk，128个mel bins，长度为 3000
    dummy_mel = torch.randn(1, 128, 3000, dtype=torch.bfloat16, device="cuda")
    dummy_mel_lengths = [3000]

    with torch.no_grad():
        # SGLang 音频前向
        sgl_audio_out, _ = sglang_model.audio_encoder(dummy_mel, dummy_mel_lengths)
        
        # 原始模型音频前向 
        # 注意：这里需要调用你原始模型中真正跑音频 forward 的逻辑
        orig_audio_out, _ = train_model.audio_encoder(dummy_mel, dummy_mel_lengths)

    a_max_diff = torch.max(torch.abs(sgl_audio_out - orig_audio_out)).item()
    a_mean_diff = torch.mean(torch.abs(sgl_audio_out - orig_audio_out)).item()
    print(f"Audio Output Shape: {sgl_audio_out.shape}")
    print(f"Audio Max Diff:  {a_max_diff:.6f}")
    print(f"Audio Mean Diff: {a_mean_diff:.6f}")
    if a_max_diff < 1e-3:
        print("✅ Audio Encoder weights loaded perfectly!")
    else:
        print("❌ Audio Encoder has significant precision differences.")

    print("\nAll tests completed.")