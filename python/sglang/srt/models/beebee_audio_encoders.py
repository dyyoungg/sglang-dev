import logging
from typing import  List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

from transformers.activations import ACT2FN
from transformers.models.whisper.configuration_whisper import WhisperConfig
from flash_attn import flash_attn_varlen_func

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear
)
from sglang.srt.configs.beebeeomni_config import BeeBeeAudioConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_npu


_is_cuda = is_cuda()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


class WhisperAttention(nn.Module):
    """
    Multi-head self-attention for Whisper encoder layers.
 
    Reuses sglang's TP infrastructure (identical to whisper.py's encoder path):
      - QKVParallelLinear  : merges q/k/v and shards across TP ranks by head
      - RowParallelLinear  : out_proj, gathers across TP ranks after projection
 
    Two forward code paths:
      - Padded batch  (cu_seqlens_q is None) : hidden_states [B, T, D]
        Uses F.scaled_dot_product_attention with TP-local head counts.
      - Varlen batch  (cu_seqlens_q provided): hidden_states [total_tokens, D]
        Uses flash_attn_varlen_func with TP-local head counts.
 
    Weight-loading note: Whisper encoder k_proj has bias=False while q/v have
    bias=True. With bias=True on the merged QKVParallelLinear the k bias slice
    is never written by load_weights (no k_proj.bias key in the checkpoint) and
    stays at the PyTorch default zero initialisation — which is correct.
    """
 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        tp_size = 1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        tp_rank = 0 if self.use_data_parallel else get_tensor_model_parallel_rank()
        
        assert num_heads % tp_size == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        )
        # Number of heads on *this* TP rank
        self.num_heads = num_heads // tp_size
 
        # QKVParallelLinear shards the head dimension across TP ranks.
        # bias=True: q and v biases are loaded from the checkpoint;
        #            k bias stays at zero (Whisper encoder has no k_proj.bias).
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # RowParallelLinear reduces across TP ranks after the projection.
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q:  Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q:  Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
    ) -> torch.Tensor:
        # QKVParallelLinear returns (output, bias); output is already TP-local.
        qkv, _ = self.qkv_proj(hidden_states)
        # Split along last dim: each chunk is [*, num_heads_local * head_dim]
        q, k, v = qkv.chunk(3, dim=-1)
 
        if cu_seqlens_q is None:
            # ── padded-batch path [B, T, D] ───────────────────────────────
            B, T, _ = hidden_states.shape
            # Reshape to [B, num_heads_local, T, head_dim] for sdpa
            q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # scale=1.0 because we already applied self.scaling above via
            # QKVParallelLinear; match whisper.py encoder path exactly.
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
            out = out.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        else:
            # ── varlen (packed) path [total_tokens, D] ────────────────────
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
            out = out.reshape(N, self.num_heads * self.head_dim).contiguous()
 
        # RowParallelLinear gathers across TP ranks; returns (output, bias).
        out, _ = self.out_proj(out)
        return out
 
 
 
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        d = config.d_model
        self.self_attn = WhisperAttention(d, config.encoder_attention_heads, quant_config=quant_config)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.tp_size = 1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        self.tp_rank = 0 if self.use_data_parallel else get_tensor_model_parallel_rank()
        self.fc1 = ColumnParallelLinear(d, config.encoder_ffn_dim, quant_config=quant_config, tp_rank=self.tp_rank, tp_size=self.tp_size)
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, d, quant_config=quant_config, tp_rank=self.tp_rank, tp_size=self.tp_size)
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
 
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
 
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
 
    def __init__(self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        d = config.d_model
        self.max_source_positions = config.max_source_positions
        self.dropout = config.dropout
     
        self.conv1 = nn.Conv1d(self.NUM_MEL_BINS, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, d)
 
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config, quant_config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d)

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
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        T_out = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.permute(0, 2, 1)                        # [B, T_out, D]
        hidden_states = inputs_embeds + self.embed_positions.weight[:T_out]   # fixed sinusoidal PE
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
       
        if input_seq_lens is not None:
            # build cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv and q, k, v by seq_lens
            B, S, D = hidden_states.shape
            cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32).to(
                hidden_states.device
            )
            cu_seqlens_kv = torch.zeros(B + 1, dtype=torch.int32).to(
                hidden_states.device
            )
            max_seqlen_q = torch.max(input_seq_lens).to(hidden_states.device)
            max_seqlen_kv = torch.max(input_seq_lens).to(hidden_states.device)
 
            hidden_states_collect = []
            for i in range(B):
                hidden_states_collect.append(hidden_states[i, : input_seq_lens[i], :])
                cu_seqlens_q[i + 1] = cu_seqlens_q[i] + input_seq_lens[i]
                cu_seqlens_kv[i + 1] = cu_seqlens_kv[i] + input_seq_lens[i]
                max_seqlen_q = max(max_seqlen_q, input_seq_lens[i])
                max_seqlen_kv = max(max_seqlen_kv, input_seq_lens[i])
 
            hidden_states = torch.cat(hidden_states_collect, dim=0)
 
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None
            max_seqlen_q = None
            max_seqlen_kv = None
        # ── Transformer layers ────────────────────────────────────────────
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens_q=cu_seqlens_q,   
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,   
                max_seqlen_kv=max_seqlen_kv,
            )
 
        # ── Unpack back to padded batch ───────────────────────────────────
        if input_seq_lens is not None:
            # recover the hidden_states, split by seq_lens and then pad&cat
            hidden_states = torch.split(
                hidden_states, input_seq_lens.detach().cpu().numpy().tolist(), dim=0
            )
            hidden_states = pad_sequence(hidden_states, batch_first=True)
 
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states   # [B, T_out, d_model]

 
 
class AudioConvUpScaleProjector(nn.Module):
  
    def __init__(self, encoder_hidden: int, out_hidden: int, downsample_ratio: int = 10):
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
        target_seq_len = math.ceil((seq_len + self.audio_downsample_ratio - 1) / self.audio_downsample_ratio) * self.audio_downsample_ratio
        pad_len = target_seq_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度 padding
 
        new_seq_len = target_seq_len // self.linear_compress_ratio
        x = x.reshape(B, new_seq_len, D * self.linear_compress_ratio)
        x = self.linear2(self.gelu(self.linear1(x)))   # [B, compressed_T, out_hidden]
 
        num_tokens = [(l + self.audio_downsample_ratio - 1)// self.audio_downsample_ratio for l in feature_lengths]
        parts = [x[i, : num_tokens[i], :] for i in range(B)]
        out = torch.cat(parts, dim=0)   # [total_tokens, out_hidden]
        return out, num_tokens
 

class AudioMLPChannelProjector(nn.Module):
    """Pure MLP channel downsampler: reshape adjacent frames into channel dim + MLP mapping."""

    def __init__(self, encoder_hidden: int, out_hidden: int, downsample_ratio: int = 10):
        super().__init__()
        self.audio_downsample_ratio = downsample_ratio
        self.linear1 = nn.Linear(encoder_hidden * downsample_ratio, out_hidden, bias=True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_hidden, out_hidden, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        feature_lengths: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        B, seq_len, D = x.size()
        ratio = self.audio_downsample_ratio

        target_seq_len = math.ceil(seq_len / ratio) * ratio
        pad_len = target_seq_len - seq_len
        if pad_len > 0:
            x = torch.cat(
                [x, torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)], dim=1
            )

        new_seq_len = target_seq_len // ratio
        x = x.reshape(B, new_seq_len, D * ratio)
        x = self.linear2(self.gelu(self.linear1(x)))

        num_tokens = [(l + ratio - 1) // ratio for l in feature_lengths]
        parts = [x[i, : num_tokens[i], :] for i in range(B)]
        out = torch.cat(parts, dim=0)
        return out, num_tokens


class AudioMultiConvProjector(nn.Module):
    """Multi-layer Conv1d progressive downsampler + MLP dimension mapping."""

    def __init__(self, encoder_hidden: int, out_hidden: int, downsample_ratio: int = 2):
        super().__init__()
        self.audio_downsample_ratio = downsample_ratio
        n_conv_layers = int(math.log2(downsample_ratio))
        assert 2 ** n_conv_layers == downsample_ratio, (
            f"downsample_ratio must be power of 2, got {downsample_ratio}"
        )

        convs: List[nn.Module] = []
        for _ in range(n_conv_layers):
            convs.append(nn.Conv1d(encoder_hidden, encoder_hidden, kernel_size=3, stride=2, padding=1))
            convs.append(nn.GELU())
        self.convs = nn.Sequential(*convs)

        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden, out_hidden, bias=True),
            nn.GELU(),
            nn.Linear(out_hidden, out_hidden, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_lengths: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        B, seq_len, _ = x.size()
        ratio = self.audio_downsample_ratio

        # Conv1d expects [B, D, T]
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)  # [B, T', D]
        x = self.mlp(x)

        num_tokens = [(int(l) + ratio - 1) // ratio for l in feature_lengths]
        parts = [x[i, : num_tokens[i], :] for i in range(B)]
        out = torch.cat(parts, dim=0)
        return out, num_tokens


def build_audio_projector(
    projector_type: str,
    encoder_hidden: int,
    out_hidden: int,
    downsample_ratio: int,
) -> nn.Module:
    if projector_type == "conv_channel_upscale":
        return AudioConvUpScaleProjector(encoder_hidden, out_hidden, downsample_ratio)
    elif projector_type == "multi_conv":
        return AudioMultiConvProjector(encoder_hidden, out_hidden, downsample_ratio)
    elif projector_type == "mlp_channel":
        return AudioMLPChannelProjector(encoder_hidden, out_hidden, downsample_ratio)
    else:
        raise NotImplementedError(f"Unknown audio projector type: {projector_type}")


class BeeBeeAudioEncoder(nn.Module):

    def __init__(self, audio_config: BeeBeeAudioConfig, out_hidden_size: int):
        super().__init__()
        whisper_hidden: int    = audio_config.d_model
        downsample_ratio: int  = getattr(audio_config, "audio_downsample_ratio", 10)
        projector_type: str    = getattr(audio_config, "audio_projector_type", "conv_channel_upscale")
        self.out_hidden_size = out_hidden_size
        self.encoder      = WhisperEncoder(audio_config)
        self.audio_projector = build_audio_projector(
            projector_type=projector_type,
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
        nvtx.range_push("BeeBeeAudioEncoder.forward")
        input_features = input_features.to(self.device, self.dtype)

        if isinstance(feature_lengths, List):
            input_seq_lens = torch.tensor(feature_lengths, dtype=torch.long, device=self.device)
        elif isinstance(feature_lengths, torch.Tensor):
            input_seq_lens = feature_lengths.to(dtype=torch.long, device=self.device)

        # WhisperEncoder: [B, mel, T] -> [B, T//2, d_model]
        nvtx.range_push("whisper_encoder")
        whisper_out = self.encoder(input_features, input_seq_lens=input_seq_lens)
        nvtx.range_pop()  # whisper_encoder

        # AudioConvUpScaleProjector: [B, T//2, d_model] -> [N_tokens, llm_hidden]
        nvtx.range_push("audio_projector")
        audio_embeds, num_tokens = self.audio_projector(whisper_out, feature_lengths)
        nvtx.range_pop()  # audio_projector

        nvtx.range_pop()  # BeeBeeAudioEncoder.forward
        return audio_embeds, num_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3 Audio Encoder
# ═══════════════════════════════════════════════════════════════════════════════


class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding (not learned)."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        assert channels % 2 == 0, "SinusoidsPositionEmbedding needs even channels"
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
        position_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("positional_embedding", position_embedding, persistent=False)

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# ── Audio length utilities ────────────────────────────────────────────────────

def _get_feat_extract_output_lengths(input_lengths: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Output length after conv stack + chunking."""
    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13


def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
    """Length after three (k=3, s=2, p=1) convolutions; zero stays zero."""
    for _ in range(3):
        lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
    return lengths


def _chunk_and_pad_features(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split audio mel features into fixed-size chunks and pad."""
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.full(
        (chunk_num.sum().item(),), n_window * 2,
        dtype=torch.long, device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    return padded_feature, chunk_lengths


def _get_valid_indices(chunk_lengths: torch.Tensor, n_window: int) -> torch.Tensor:
    """Flat indices of valid (non-padding) positions after CNN."""
    feature_lens_after_cnn = _post_cnn_length(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=chunk_lengths.device) < feature_lens_after_cnn.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def _get_audio_cu_seqlens(
    chunk_lengths: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window_infer: int,
    n_window: int,
) -> torch.Tensor:
    """Cumulative sequence lengths for windowed flash attention."""
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens, n_window)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths, n_window)
    max_len_after_cnn = feature_lens_after_cnn.max().item()

    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio

    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len_val = cnn_len.item() if hasattr(cnn_len, "item") else int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len_val // window_aftercnn)
        remainder = cnn_len_val % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]

    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


# ── Qwen3 Audio Attention & Encoder Layer ─────────────────────────────────────

class Qwen3AudioAttention(nn.Module):
    """Multi-head attention using flash_attn_varlen_func for Qwen3 audio encoder."""

    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = attention_dropout
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [total_tokens, embed_dim]
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )
        return self.out_proj(attn_output.reshape(seq_length, -1).contiguous())


class Qwen3AudioEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer for Qwen3 audio."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = Qwen3AudioAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


# ── Qwen3 Audio Encoder Core ─────────────────────────────────────────────────

class Qwen3AudioEncoderCore(nn.Module):
    """
    Qwen3 Audio Encoder: 3x Conv2d (8x temporal downsample) + sinusoidal PE
    + N transformer encoder layers + LN.

    Input  : input_features  [B, mel_bins, T]  or  [mel_bins, total_frames] packed
             feature_lens    [B] valid mel-frame counts
    Output : hidden_states   [total_valid_tokens, d_model]
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if getattr(config, "scale_embedding", False) else 1.0
        self.n_window = getattr(config, "n_window", 50)
        self.n_window_infer = getattr(config, "n_window_infer", 800)
        self.conv_chunksize = getattr(config, "conv_chunksize", 500)
        self.dropout_rate = getattr(config, "dropout", 0.0)

        # Positional embedding
        pos_emb_len = getattr(config, "max_source_positions", None) or getattr(config, "max_position_embeddings", 1500)
        self.positional_embedding = SinusoidsPositionEmbedding(pos_emb_len, embed_dim)

        # Conv2d stem: 3 layers, each stride 2 → 8x temporal downsample
        dhs = getattr(config, "downsample_hidden_size", 480)
        num_mel_bins = getattr(config, "num_mel_bins", 128)
        self.conv2d1 = nn.Conv2d(1, dhs, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(dhs, dhs, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(dhs, dhs, 3, 2, padding=1)
        freq_after_conv = (((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(dhs * freq_after_conv, embed_dim, bias=False)

        # Transformer layers
        encoder_layers = getattr(config, "encoder_layers", 32)
        num_heads = getattr(config, "encoder_attention_heads", 20)
        ffn_dim = getattr(config, "encoder_ffn_dim", 5120)
        dropout = getattr(config, "dropout", 0.0)
        attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.layers = nn.ModuleList([
            Qwen3AudioEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, attention_dropout)
            for _ in range(encoder_layers)
        ])
        self.ln_post = nn.LayerNorm(embed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv2d1.weight.device

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns packed hidden_states: [total_valid_tokens, d_model]
        """
        # Normalize input: [B, mel, T] → packed [mel, total_frames]
        if input_features.ndim == 3:
            batch_size = input_features.shape[0]
            parts = []
            for i in range(batch_size):
                length = feature_lens[i].item()
                parts.append(input_features[i, :, :length])
            input_features = torch.cat(parts, dim=1)  # [mel, total_frames]
        elif input_features.ndim != 2:
            raise ValueError(f"Unexpected input_features ndim={input_features.ndim}")

        # Chunk and pad
        padded_feature, chunk_lengths = _chunk_and_pad_features(
            input_features, feature_lens, self.n_window
        )
        valid_indices = _get_valid_indices(chunk_lengths, self.n_window)
        cu_seqlens = _get_audio_cu_seqlens(
            chunk_lengths, feature_lens, self.n_window_infer, self.n_window
        )
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # Conv2d: [num_chunks, 1, mel, chunk_len]
        padded_feature = padded_feature.unsqueeze(1).to(dtype=self.conv2d1.weight.dtype)

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embedding
        pos_emb = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + pos_emb

        # Select valid positions → packed sequence
        hidden_states = torch.index_select(
            padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices
        )

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)

        hidden_states = self.ln_post(hidden_states)
        return hidden_states  # [total_valid_tokens, d_model]


# ── BeeBee Qwen3 Audio Encoder Wrapper ────────────────────────────────────────

class BeeBeeQwen3AudioEncoder(nn.Module):
    """
    Qwen3 audio encoder + pluggable projector.
    Same forward interface as BeeBeeAudioEncoder:
        forward(input_features [B, mel, T], feature_lengths [list/tensor]) -> (packed_embeds, num_tokens)
    """

    def __init__(self, audio_config, out_hidden_size: int):
        super().__init__()
        encoder_hidden: int = audio_config.d_model
        downsample_ratio: int = getattr(audio_config, "audio_downsample_ratio", 2)
        projector_type: str = getattr(audio_config, "audio_projector_type", "multi_conv")
        self.n_window = getattr(audio_config, "n_window", 50)
        self.out_hidden_size = out_hidden_size
        self.encoder = Qwen3AudioEncoderCore(audio_config)
        self.audio_projector = build_audio_projector(
            projector_type=projector_type,
            encoder_hidden=encoder_hidden,
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
        input_features: torch.Tensor,   # [B, mel_bins, T]
        feature_lengths: List[int],      # valid mel-frame count per sample
    ) -> Tuple[torch.Tensor, List[int]]:
        nvtx.range_push("BeeBeeQwen3AudioEncoder.forward")
        input_features = input_features.to(self.device, self.dtype)

        if isinstance(feature_lengths, list):
            feature_lens = torch.tensor(feature_lengths, dtype=torch.long, device=self.device)
        elif isinstance(feature_lengths, torch.Tensor):
            feature_lens = feature_lengths.to(dtype=torch.long, device=self.device)
        else:
            feature_lens = feature_lengths

        # Encoder → packed [total_valid_tokens, d_model]
        nvtx.range_push("qwen3_audio_encoder")
        hidden_states = self.encoder(input_features, feature_lens)
        nvtx.range_pop()

        # Compute per-sample output lengths from encoder conv stack
        per_sample_lens = _get_feat_extract_output_lengths(feature_lens, self.n_window)
        per_sample_lens_list = per_sample_lens.tolist()

        # Unpack to batched [B, max_seq, d_model] for projector
        batch_size = input_features.shape[0] if input_features.ndim == 3 else len(feature_lengths)
        max_seq_len = max(per_sample_lens_list)
        batched = torch.zeros(
            batch_size, max_seq_len, hidden_states.shape[-1],
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        offset = 0
        for i, length in enumerate(per_sample_lens_list):
            length = int(length)
            batched[i, :length, :] = hidden_states[offset : offset + length]
            offset += length

        # Projector → packed [total_proj_tokens, out_hidden]
        nvtx.range_push("audio_projector")
        audio_embeds, num_tokens = self.audio_projector(batched, per_sample_lens_list)
        nvtx.range_pop()

        nvtx.range_pop()  # BeeBeeQwen3AudioEncoder.forward
        return audio_embeds, num_tokens
