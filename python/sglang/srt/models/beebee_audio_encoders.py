import logging
from typing import  List, Optional, Tuple

import torch
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
 

class BeeBeeAudioEncoder(nn.Module):

    def __init__(self, audio_config: BeeBeeAudioConfig, out_hidden_size: int):
        super().__init__()
        whisper_hidden: int    = audio_config.d_model
        downsample_ratio: int  = getattr(audio_config, "audio_downsample_ratio", 10)
        self.out_hidden_size = out_hidden_size
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

        if isinstance(feature_lengths, List):
            input_seq_lens = torch.tensor(feature_lengths, dtype=torch.long, device=self.device)
        elif isinstance(feature_lengths, torch.Tensor):
            input_seq_lens = feature_lengths.to(dtype=torch.long, device=self.device)
 
        # WhisperEncoder: [B, mel, T] -> [B, T//2, d_model]
        whisper_out = self.encoder(input_features, input_seq_lens=input_seq_lens)
 
        # AudioConvUpScaleProjector: [B, T//2, d_model] -> [N_tokens, llm_hidden]
        audio_embeds, num_tokens = self.audio_projector(whisper_out, feature_lengths)
        return audio_embeds, num_tokens
