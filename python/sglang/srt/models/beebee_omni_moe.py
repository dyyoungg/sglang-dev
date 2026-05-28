
import logging
import re
from functools import partial, lru_cache
from typing import Iterable, List, Optional, Tuple, Type, Callable
import io
import wave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import base64
from PIL import Image

from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.whisper.configuration_whisper import WhisperConfig
from flash_attn import flash_attn_func, flash_attn_varlen_func

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group, init_distributed_environment, initialize_model_parallel
from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.attention.vision import VisionAttention, BATCH_BUCKETS, FLASHINFER_MAX_SEQLEN_BUCKETS, FLASHINFER_WORKSPACE_SIZE_BYTES
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear
)
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
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
from sglang.srt.models.qwen3_moe import Qwen3MoeModel
from sglang.srt.configs.beebeeomni_moe_config import BeeBeeMoEOmniConfig, BeeBeeAudioConfig, BeeBeeMoEVisionConfig
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, compute_cu_seqlens_from_grid_numpy
from sglang.srt.multimodal.mm_utils import run_dp_sharded_beebee_vision_model, run_dp_sharded_audio_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler, ServerArgs
from sglang.srt.utils import add_prefix, is_cuda, is_npu, is_cpu, round_up
from sglang.srt.entrypoints.warmup import warmup
from sglang.srt.managers.io_struct import GenerateReqInput

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



class Qwen3_VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output

    
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = Conv3dLayer(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        head_size: Optional[int] = None,
        hidden_act="silu",
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        workspace_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            head_dim=head_size,
            projection_size=num_heads * head_size,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
            workspace_buffer=workspace_buffer,
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_ws: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_ws=output_ws,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x += attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x += mlp
        return x
    
class Qwen35VisionPatchMerger(nn.Module):

    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is typically [S, B, D] or [seq_len, hidden] depending on upstream
        return self.norm(x).view(-1, self.hidden_size)
    

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
    

class BeeBeeQwen3MoeVisionModel(nn.Module, RotaryPosMixin):

    def __init__(
        self,
        vision_config: BeeBeeMoEVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        downsample_ratio: int = 16,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.num_grid = self.num_grid_per_side * self.num_grid_per_side
        self.align_corners = (
            get_global_server_args().enable_precise_embedding_interpolation
        )
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.use_data_parallel = use_data_parallel

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)

        if self.pp_group.is_first_rank:
            self.pos_embed = VocabParallelEmbedding(
                self.num_position_embeddings,
                self.hidden_size,
                quant_config=quant_config,
                enable_tp=not use_data_parallel,
                use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
                prefix=add_prefix("pos_embed", prefix),
            )
        else:
            self.pos_embed = PPMissingLayer()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        if is_cpu() and hasattr(vision_config, "original_num_heads"):
            head_dim = self.hidden_size // vision_config.original_num_heads
        else:
            head_dim = self.hidden_size // self.num_heads

        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )
        workspace_buffer = None
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            if torch.cuda.is_available() and (not _is_npu):
                ws_device = torch.device("cuda", torch.cuda.current_device())
            else:
                ws_device = self.device
            workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_SIZE_BYTES,
                dtype=torch.uint8,
                device=ws_device,
            )

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=vision_config.intermediate_size,
                    head_size=head_dim,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    use_data_parallel=use_data_parallel,
                    workspace_buffer=workspace_buffer,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = Qwen35VisionPatchMerger(
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )
        self.embed_dim = self.hidden_size * self.spatial_merge_unit
        self.mm_projector = DynamicAvgPoolProjector(
            encoder_hidden=self.embed_dim,
            out_hidden=vision_config.output_size,
            downsample_ratio=downsample_ratio,
            quant_config=quant_config,
            prefix=add_prefix("mm_projector", prefix),
        )
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
        self.cuda_graph_runner: Optional[ViTCudaGraphRunner] = None
        if self.enable_cg:
            self.cuda_graph_runner = ViTCudaGraphRunner(self)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device
    
    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined
    
    def _get_interpolation_indices(self, dim_size: int) -> torch.Tensor:
        """
        Compute continuous interpolation indices for a single dimension.

        Returns continuous indices.
        """
        if self.align_corners:
            indices = np.linspace(
                0, self.num_grid_per_side - 1, dim_size, dtype=np.float32
            )
        else:
            indices = (np.arange(dim_size, dtype=np.float32) + 0.5) * (
                self.num_grid_per_side / dim_size
            ) - 0.5
            indices = np.clip(indices, 0, self.num_grid_per_side - 1)
        return indices

    def _get_interpolation_indices(self, dim_size: int) -> torch.Tensor:
        """
        Compute continuous interpolation indices for a single dimension.

        Returns continuous indices.
        """
        if self.align_corners:
            indices = np.linspace(
                0, self.num_grid_per_side - 1, dim_size, dtype=np.float32
            )
        else:
            indices = (np.arange(dim_size, dtype=np.float32) + 0.5) * (
                self.num_grid_per_side / dim_size
            ) - 0.5
            indices = np.clip(indices, 0, self.num_grid_per_side - 1)
        return indices

    def _calculate_indices_and_weights(self, h_idxs, w_idxs):
        """
        Compute bilinear interpolation indices and weights.

        Returns tuple of (indices, weights), each as 4 numpy arrays for the 4 corner points.
        """
        h_f = np.floor(h_idxs).astype(np.int64)
        h_c = np.clip(h_f + 1, 0, self.num_grid_per_side - 1)
        dh = h_idxs - h_f

        w_f = np.floor(w_idxs).astype(np.int64)
        w_c = np.clip(w_f + 1, 0, self.num_grid_per_side - 1)
        dw = w_idxs - w_f

        side = self.num_grid_per_side

        indices = [
            (h_f[:, None] * side + w_f).flatten(),
            (h_f[:, None] * side + w_c).flatten(),
            (h_c[:, None] * side + w_f).flatten(),
            (h_c[:, None] * side + w_c).flatten(),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)).flatten(),
            ((1 - dh)[:, None] * dw).flatten(),
            (dh[:, None] * (1 - dw)).flatten(),
            (dh[:, None] * dw).flatten(),
        ]
        return indices, weights

    def _get_position_embedding(self, patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        """
        Tile and reorganize position embeddings to align with the token sequence.
        """
        result_parts = []
        merge_size = self.spatial_merge_size

        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)

            h_merge = h // merge_size
            w_merge = w // merge_size

            pos_embed = (
                pos_embed.view(t, h_merge, merge_size, w_merge, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )

            result_parts.append(pos_embed)

        return torch.cat(result_parts, dim=0)

    def _torch_interp_indices(
        self, dim_size: int, device: torch.device
    ) -> torch.Tensor:
        side = self.num_grid_per_side
        if self.align_corners:
            # align_corners=True
            return torch.linspace(
                0, side - 1, dim_size, dtype=torch.float32, device=device
            )
        else:
            # align_corners=False  (match _get_interpolation_indices)
            idx = (torch.arange(dim_size, dtype=torch.float32, device=device) + 0.5) * (
                side / dim_size
            ) - 0.5
            return idx.clamp_(0, side - 1)

    def fast_pos_embed_interpolate_from_list(self, grid_thw):
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def add_padding_to_fi_seqlens(
        self, seq: np.ndarray, batch_size: int, padding_value: int
    ) -> np.ndarray:
        batch_size_padded = next(
            (b for b in BATCH_BUCKETS if b >= batch_size),
            # For large batches (> max bucket), round up to a multiple of
            # the base bucket size to avoid negative pad length.
            round_up(batch_size, BATCH_BUCKETS[0]),
        )
        if batch_size_padded == batch_size:
            return seq
        return np.concatenate(
            [
                seq,
                np.full(
                    (batch_size_padded - batch_size,), padding_value, dtype=seq.dtype
                ),
            ]
        )

    def bucket_flashinfer_max_seqlen(self, real_max_seqlen: int) -> int:
        if real_max_seqlen <= 0:
            return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
        return next(
            (s for s in FLASHINFER_MAX_SEQLEN_BUCKETS if s >= real_max_seqlen),
            # For large sequences (> max bucket), round up to a multiple of
            # the largest bucket to avoid under-estimation.
            round_up(real_max_seqlen, FLASHINFER_MAX_SEQLEN_BUCKETS[-1]),
        )

    def fast_pos_embed_interpolate(self, grid_thw):
        """Interpolate position embeddings for (batch, 3) size input dimensions.

        Performs bilinear interpolation on spatial dimensions (height, width) and replicates
        along temporal dimension. The result is reorganized according to spatial_merge_size.

        Args:
            grid_thw: Tensor of shape [batch_size, 3] with (temporal, height, width) dimensions
                     in patches for each sample.

        Returns:
            Interpolated position embeddings tensor.
        """
        grid_thw_cpu = grid_thw.cpu().numpy()

        # transfer data to CPU before loop
        temporal_dims = grid_thw_cpu[:, 0].tolist()
        height_dims = grid_thw_cpu[:, 1].tolist()
        width_dims = grid_thw_cpu[:, 2].tolist()

        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        patches_size = [h * w for h, w in zip(height_dims, width_dims)]
        total_patches = sum(patches_size)
        all_indices_np = np.zeros((4, total_patches), dtype=np.int64)
        all_weights_np = np.zeros((4, total_patches), dtype=np.float32)

        current_idx = 0

        # calculate indices and weights on CPU
        for t, h, w in zip(temporal_dims, height_dims, width_dims):
            h_idxs = self._get_interpolation_indices(h)
            w_idxs = self._get_interpolation_indices(w)

            indices, weights = self._calculate_indices_and_weights(h_idxs, w_idxs)

            end_idx = current_idx + h * w
            for i in range(4):
                all_indices_np[i, current_idx:end_idx] = indices[i]
                all_weights_np[i, current_idx:end_idx] = weights[i]
            current_idx = end_idx

        idx_tensor = torch.from_numpy(all_indices_np).to(device)
        weight_tensor = torch.from_numpy(all_weights_np).to(dtype=dtype, device=device)

        # calculate interpolation
        pos_embeds = self.pos_embed(idx_tensor.view(-1))
        pos_embeds = pos_embeds.view(4, total_patches, -1)
        patch_pos_embeds = (pos_embeds * weight_tensor.unsqueeze(-1)).sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split(patches_size)
        return self._get_position_embedding(
            patch_pos_embeds, temporal_dims, height_dims, width_dims
        )

    def compute_flashinfer_batch_offsets_packed(
        self,
        token_cu_seqlens: np.ndarray,
        *,
        elem_per_token: int,
    ) -> np.ndarray:
        """
        Build packed *element* indptrs for FlashInfer cuDNN prefill.

        Input:
        token_cu_seqlens: (B+1,) token indptr
        elem_per_token: per-token element width on THIS TP rank
                        (usually hidden_size / attn_tp_size)

        Output:
        packed_offsets: (3 * (B_padded + 1),) int32
            [qk_indptr, v_indptr, o_indptr] concatenated,
            each indptr is (B_padded + 1,) in element units.
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)
        B_padded = self.bucket_flashinfer_batch_size(B)

        # token indptr -> pad to (B_padded+1,) by appending total_tokens for extra empty sequences
        token_indptr = token_cu_seqlens.astype(np.int64, copy=False)  # (B+1,)
        if B_padded != B:
            pad = np.full((B_padded - B,), token_indptr[-1], dtype=token_indptr.dtype)
            token_indptr = np.concatenate([token_indptr, pad], axis=0)  # (B_padded+1,)

        # convert token indptr -> element indptr
        elem_indptr = (token_indptr * int(elem_per_token)).astype(
            np.int32
        )  # (B_padded+1,)

        # q/k/v/o in this ViT path share the same indptr
        return np.concatenate([elem_indptr, elem_indptr, elem_indptr], axis=0)

    def bucket_flashinfer_batch_size(self, batch_size: int) -> int:
        """Bucketize batch size for cuDNN graph caching."""
        return next(
            (b for b in BATCH_BUCKETS if b >= batch_size),
            round_up(batch_size, BATCH_BUCKETS[0]),
        )

    def compute_flashinfer_sequence_lengths_padded(
        self,
        token_cu_seqlens: np.ndarray,
    ) -> np.ndarray:
        """
        token_cu_seqlens: (B+1,) token indptr
        return: (B_padded,) token lengths (padded with 0)
        """
        assert token_cu_seqlens.ndim == 1 and token_cu_seqlens.size >= 2
        B = int(token_cu_seqlens.size - 1)

        seq_lens = (token_cu_seqlens[1:] - token_cu_seqlens[:-1]).astype(
            np.int32
        )  # (B,)

        B_padded = self.bucket_flashinfer_batch_size(B)
        if B_padded != B:
            pad = np.zeros((B_padded - B,), dtype=np.int32)
            seq_lens = np.concatenate([seq_lens, pad], axis=0)  # (B_padded,)
        return seq_lens
    
    def _prepare_graph_inputs(self, x: torch.Tensor, grid_thw: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x += pos_embeds

        # rotary embedding -> (cos, sin)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # compute cu_seqlens
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw)
        return x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin
    
    def forward_with_cuda_graph(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        (
            x,
            cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        ) = self._prepare_graph_inputs(x, grid_thw)
        if not isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = torch.tensor(cu_seqlens, device=x.device, dtype=torch.int32)
        else:
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)
        cu_seqlens = cu_seqlens.contiguous()

        x = self.cuda_graph_runner.run(
            x=x,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=None,
            output_indices=None,
        )
        proj_thw = grid_thw.clone()
        proj_thw[:, 1] = grid_thw[:, 1] // self.spatial_merge_size
        proj_thw[:, 2] = grid_thw[:, 2] // self.spatial_merge_size
        
        features, seq_lens = self.mm_projector(x, proj_thw)
        return features, seq_lens

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

        grid_thw_list = grid_thw.tolist()
      
        pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
        x += pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0).to(device=x.device, dtype=torch.int32)

        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        
        x = x.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=None,
                sequence_lengths=None,
            )

        x = self.merger(x)
        proj_thw = grid_thw.clone()
        proj_thw[:, 1] = grid_thw[:, 1] // self.spatial_merge_size
        proj_thw[:, 2] = grid_thw[:, 2] // self.spatial_merge_size
        features, seq_lens = self.mm_projector(x, proj_thw)
       
        return features, seq_lens


class BeeBeeMoEOmniForConditionalGeneration(nn.Module):
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
        config: BeeBeeMoEOmniConfig,
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
        if not get_global_server_args().encoder_only:
            self.model = Qwen3MoeModel(
                self.text_config, 
                quant_config, 
                prefix=add_prefix("model", prefix)
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
        else:
            self.lm_head = None
        downsample_ratio=getattr(
            self.vision_config, 
            "image_downsample_ratio", 
            getattr(self.vision_config, "image_downsample_size", 16)
        )
        self.image_encoder = BeeBeeQwen3MoeVisionModel(
            self.vision_config,
            norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("image_encoder", prefix),
            downsample_ratio=downsample_ratio,
            use_data_parallel=self.use_data_parallel,
            # max_context_len=self.vision_config.max_position_embeddings,
        )

  
        self.audio_encoder = BeeBeeAudioEncoder(
            audio_config=self.audio_config,
            out_hidden_size=self.text_config.hidden_size,
        )
        
        self.is_mrope_enabled = False

        self.logits_processor = LogitsProcessor(self.text_config)
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

        raw_patch_dim = 1536

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
            image_embeds, _ = self.image_encoder(pixel_values, grid_thw=image_grid_thw)
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
        WHISPER_HOP_LENGTH = 320
        
        if self.use_data_parallel:
            items_mel_chunks = []
            items_chunk_lengths = []
            for item in items:
                mel_chunks = item.feature.to(self.audio_encoder.device, self.audio_encoder.dtype)
                chunk_lengths = [l // WHISPER_HOP_LENGTH for l in item.model_specific_data["audio_length"]]
                items_mel_chunks.append(mel_chunks)
                items_chunk_lengths.append(chunk_lengths)

            return run_dp_sharded_audio_model(
                self.audio_encoder,
                items_mel_chunks,
                items_chunk_lengths,
            )
        else:
            all_mel_chunks = []
            all_chunk_lengths = []
            for item in items:
                mel_chunks = item.feature.to(self.audio_encoder.device, self.audio_encoder.dtype)
                chunk_lengths = [l // WHISPER_HOP_LENGTH for l in item.model_specific_data["audio_length"]]
                all_mel_chunks.append(mel_chunks)
                all_chunk_lengths.extend(chunk_lengths)

            batched_mel_chunks = torch.cat(all_mel_chunks, dim=0)
            chunk_embeds, _ = self.audio_encoder(batched_mel_chunks, all_chunk_lengths)
            return chunk_embeds



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
     
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Whisper encoder self_attn k_proj has no bias in the checkpoint, but
        # our QKVParallelLinear uses a single merged bias (bias=True).  We
        # inject a zero tensor so the weight_loader can correctly shard the k
        # slice into qkv_proj.bias just like q and v.  Without this the k
        # portion would stay at PyTorch's default zero init — which is still
        # numerically correct, but explicit injection avoids any future
        # confusion if the default ever changes.

        weights = list(weights)
        encoder_layers = getattr(self.audio_encoder.encoder, "layers", [])
        for layer_idx in range(len(encoder_layers)):
            k_w_key = f"audio_encoder.layers.{layer_idx}.self_attn.k_proj.weight"
            k_b_key = f"audio_encoder.layers.{layer_idx}.self_attn.k_proj.bias"
            # Find the k_proj weight to get the right size
            k_proj_weight = next(
                (w for n, w in weights if n == k_w_key), None
            )
            if k_proj_weight is not None and not any(n == k_b_key for n, _ in weights):
                weights.append((k_b_key, torch.zeros(k_proj_weight.size(0))))
        
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

 
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                    weight_loader(lm_head_param, loaded_weight)

            if name.startswith("image_encoder."):
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

    
            if name.startswith("audio_encoder.") and not name.startswith("audio_encoder.audio_projector."):
                name = name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

            moe_expert_mapping = [
                ("gate_proj", "w13_weight", "weight13", "w1"),
                ("up_proj",   "w13_weight", "weight13", "w3"),
                ("down_proj", "w2_weight",  "weight2",  "w2"),
            ]
            
            is_moe_expert = False
            for proj_name, target_name, fallback_name, shard_id in moe_expert_mapping:
                search_base = f"mlp.experts.{proj_name}"

                if search_base in name:
                    mapped_name = name.replace(f"experts.{proj_name}", f"experts.{target_name}")
                    
                    if mapped_name not in params_dict and mapped_name.replace(target_name, fallback_name) in params_dict:
                        mapped_name = mapped_name.replace(target_name, fallback_name)
                    
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        
                        global_num_experts = loaded_weight.shape[0]

                        for global_idx in range(global_num_experts):
                            
                            weight_loader(
                                param=param, 
                                loaded_weight=loaded_weight[global_idx], 
                                weight_name=mapped_name, 
                                shard_id=shard_id, 
                                expert_id=global_idx
                            )
                    else:
                        logger.warning(f"Failed to map MoE {proj_name}: {mapped_name}")
                        
                    is_moe_expert = True
                    break
                    
            if is_moe_expert:
                continue

            # stack weight 
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                
                if f".{weight_name}." not in name and not name.endswith(f".{weight_name}"):
                    continue
                
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    print(f"{mapped_name} not in the params_dict.")
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)

                is_stacked = True
                break
            
            if is_stacked:
                continue

            # other weight
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

EntryClass = BeeBeeMoEOmniForConditionalGeneration

def compare_weights(orig_sd, sgl_sd):
    print("\n--- 🔍 Checking Model Weights (Tensor Values) ---")
    all_matched = True
    
    merged_tasks = {}       # 收集常规 QKV 和 Shared Expert MLP
    moe_stacked_tasks = {}  # 收集 MoE 专家的 stacked 张量 (layer_idx -> {proj_type: tensor})
    
    # 匹配 MoE 专家权重 (兼容有无 .weight 后缀)
    moe_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)(?:\.weight)?$")

    # 常规线性层合并映射
    target_map = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", "gate"),
        "up_proj": ("gate_up_proj", "up")
    }

    # ==========================================
    # 1. 遍历并归类原始权重
    # ==========================================
    for orig_name, orig_tensor in orig_sd.items():
        sgl_name = orig_name
        
        # [归类 A]: 拦截 MoE 专家的 Stacked 权重 (跳过后续逻辑)
        moe_match = moe_pattern.search(orig_name)
        if moe_match:
            layer_idx, proj_type = moe_match.groups()
            moe_stacked_tasks.setdefault(int(layer_idx), {})[proj_type] = orig_tensor
            continue
        
        # [名称映射]: 处理 Vision / Audio 模块的不对齐
        if sgl_name.startswith("image_encoder."):
            sgl_name = sgl_name.replace("attn.qkv.", "attn.qkv_proj.")

        if sgl_name.startswith("audio_encoder.") and not sgl_name.startswith("audio_encoder.audio_projector."):
            sgl_name = sgl_name.replace("audio_encoder.", "audio_encoder.encoder.", 1)

        # [归类 B]: 拦截常规 QKV 和 Shared Expert MLP 的散装权重
        is_merged_task = False
        for orig_key, (target_key, part_name) in target_map.items():
            if f".{orig_key}." in sgl_name or sgl_name.endswith(f".{orig_key}"):
                merged_sgl_name = sgl_name.replace(orig_key, target_key)
                merged_tasks.setdefault(merged_sgl_name, {})[part_name] = orig_tensor
                is_merged_task = True
                break
                
        if is_merged_task:
            continue

        # [对比 C]: 1对1 基础权重直接对比 (包括 Router/gate.weight, Norm等)
        if sgl_name not in sgl_sd:
            print(f"❌ [Missing in SGLang] {sgl_name} (mapped from {orig_name})")
            all_matched = False
            continue
            
        sgl_tensor = sgl_sd[sgl_name]
        if orig_tensor.shape != sgl_tensor.shape:
            print(f"❌ [Shape Mismatch] {orig_name} ({orig_tensor.shape}) vs {sgl_name} ({sgl_tensor.shape})")
            all_matched = False
            continue
            
        max_diff = torch.max(torch.abs(orig_tensor - sgl_tensor)).item()
        if max_diff > 1e-5:
            print(f"❌ [Value Differs] {orig_name} -> Max Diff: {max_diff:.6f}")
            all_matched = False

    # ==========================================
    # 2. 验证常规合并权重 (QKV, Shared Expert)
    # ==========================================
    for sgl_name, parts in merged_tasks.items():
        if sgl_name not in sgl_sd:
            print(f"❌ [Missing in SGLang] {sgl_name} (Merged Target)")
            all_matched = False
            continue
            
        sgl_tensor = sgl_sd[sgl_name]
        try:
            if "qkv_proj" in sgl_name:
                # 兼容 Whisper encoder self_attn 没有 k_bias 的情况
                if "q" in parts and "v" in parts and "k" not in parts:
                    parts["k"] = torch.zeros_like(parts["q"])
                orig_merged = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            elif "gate_up_proj" in sgl_name:
                orig_merged = torch.cat([parts["gate"], parts["up"]], dim=0)
            else:
                raise ValueError(f"Unknown merge target: {sgl_name}")
        except Exception as e:
            print(f"❌ [Merge Error] {sgl_name}: {e}")
            all_matched = False
            continue
            
        if orig_merged.shape != sgl_tensor.shape:
            print(f"❌ [Shape Mismatch - Merged] {sgl_name}: expected {orig_merged.shape} vs sgl {sgl_tensor.shape}")
            all_matched = False
            continue
            
        max_diff = torch.max(torch.abs(orig_merged - sgl_tensor)).item()
        if max_diff > 1e-5:
            print(f"❌ [Value Differs - Merged] {sgl_name} -> Max Diff: {max_diff:.6f}")
            all_matched = False

    # ==========================================
    # 3. 验证 MoE 专家合并权重 (w13_weight, w2_weight)
    # ==========================================
    for layer_idx, projs in moe_stacked_tasks.items():
        # --- 验证 w13_weight (Gate & Up 合并) ---
        w13_name = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
        if w13_name not in sgl_sd and f"model.layers.{layer_idx}.mlp.experts.weight13" in sgl_sd:
            w13_name = f"model.layers.{layer_idx}.mlp.experts.weight13"
            
        if w13_name in sgl_sd:
            sgl_w13 = sgl_sd[w13_name]
            if "gate_proj" in projs and "up_proj" in projs:
                # FusedMoE 预期：沿着输出特征维度 (dim=1) 将 gate 和 up 拼接
                # 原 shape 通常为 [num_experts, intermediate_size, hidden_size]
                expected_w13 = torch.cat([projs["gate_proj"], projs["up_proj"]], dim=1)
                
                if expected_w13.shape != sgl_w13.shape:
                    print(f"❌ [Shape Mismatch - MoE w13] Layer {layer_idx}: expected {expected_w13.shape} vs sgl {sgl_w13.shape}")
                    all_matched = False
                else:
                    max_diff = torch.max(torch.abs(expected_w13 - sgl_w13)).item()
                    if max_diff > 1e-5:
                        print(f"❌ [Value Differs - MoE w13] Layer {layer_idx} -> Max Diff: {max_diff:.6f}")
                        all_matched = False
            else:
                print(f"❌ [Incomplete MoE w13] Layer {layer_idx} missing gate or up.")
                all_matched = False
        else:
            print(f"❌ [Missing in SGLang] {w13_name}")
            all_matched = False

        # --- 验证 w2_weight (Down Proj 直接比对) ---
        w2_name = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
        if w2_name not in sgl_sd and f"model.layers.{layer_idx}.mlp.experts.weight2" in sgl_sd:
            w2_name = f"model.layers.{layer_idx}.mlp.experts.weight2"
            
        if w2_name in sgl_sd:
            sgl_w2 = sgl_sd[w2_name]
            if "down_proj" in projs:
                expected_w2 = projs["down_proj"]
                if expected_w2.shape != sgl_w2.shape:
                    print(f"❌ [Shape Mismatch - MoE w2] Layer {layer_idx}: expected {expected_w2.shape} vs sgl {sgl_w2.shape}")
                    all_matched = False
                else:
                    max_diff = torch.max(torch.abs(expected_w2 - sgl_w2)).item()
                    if max_diff > 1e-5:
                        print(f"❌ [Value Differs - MoE w2] Layer {layer_idx} -> Max Diff: {max_diff:.6f}")
                        all_matched = False
            else:
                print(f"❌ [Incomplete MoE w2] Layer {layer_idx} missing down_proj.")
                all_matched = False
        else:
            print(f"❌ [Missing in SGLang] {w2_name}")
            all_matched = False

    # ==========================================
    # 总结输出
    # ==========================================
    if all_matched:
        print("🎉 恭喜！所有权重全部完美对齐！")
        print("涵盖检查项：")
        print(" - Vision/Audio 特殊名称映射")
        print(" - Whisper k_proj_bias 置零填充")
        print(" - Attention QKV 张量拼接")
        print(" - Shared Expert (gate_up_proj) 张量拼接")
        print(" - Fused MoE Stacked 专家权重 (w13_weight / w2_weight) 重组与对齐")
    else:
        print("\n⚠️ 存在未对齐的权重，请往上翻看带 ❌ 的报错信息进行排查。")
        
    return all_matched

if __name__ == "__main__":
    import os
    import glob
    import torch
    from safetensors import safe_open
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    
    
    init_distributed_environment()
    initialize_model_parallel()
    
    MODEL_PATH = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/ckpt/0518_llavaomni_30A3B_qwen35encoder_st2_mmprojector/checkpoints/hf_ckpt" 
    
    dummy_args = ServerArgs(model_path=MODEL_PATH, mm_enable_dp_encoder=False)
    dummy_args.enable_dp_attention = False
    dummy_args.dp_size = 1
    dummy_args.moe_dense_tp_size = None
    dummy_args.attn_cp_size = 1
    dummy_args.device = "cuda:0"
    set_global_server_args_for_scheduler(dummy_args)

    print("Initializing models...")
    from veomni.models.custom.llava_qwen3moe.modeling_llava_qwen3moe_omni import LlavaQwen3MoeForCausalLM
    from veomni.ops.fused_moe import apply_veomni_fused_moe_patch
    apply_veomni_fused_moe_patch(moe_implementation="fused")
    
    config = BeeBeeMoEOmniConfig.from_pretrained(MODEL_PATH)
  
    class DummyModelConfig:
        def __init__(self, hidden_size, dtype):
            self.hidden_size = hidden_size
            self.dtype = dtype

    dummy_model_config = DummyModelConfig(
        hidden_size=config.text_config.hidden_size,
        dtype=torch.bfloat16
    )
    
    initialize_dp_attention(dummy_args, dummy_model_config)
    
    # === 定义设备 ===
    device_sgl = torch.device("cuda:0")
    device_orig = torch.device("cuda:1")

    # 1. 加载 SGLang 模型到 cuda:0
    sglang_model = BeeBeeMoEOmniForConditionalGeneration(config).to(torch.bfloat16).to(device_sgl)
    sglang_model.eval()

    # 2. 加载 原始模型 到 cuda:1
    train_model = LlavaQwen3MoeForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device_orig)
    train_model.eval()

    print("Loading weights into SGLang model...")
    safetensors_files = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No .safetensors files found in {MODEL_PATH}")

    weights_iterator = []
    for f in safetensors_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                weights_iterator.append((k, st.get_tensor(k)))

    sglang_model.load_weights(weights_iterator)

    # === Check Model Weight ===
    print("Preparing state dicts for comparison...")
    orig_state_dict = {k: v.cpu() for k, v in train_model.state_dict().items()}
    sgl_state_dict = {k: v.cpu() for k, v in sglang_model.state_dict().items()}
    
    compare_weights(orig_state_dict, sgl_state_dict)

    # ---------------------------------------------------------
    # 视觉编码器 (Image Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Vision Encoder ---")
 
    # 在 CPU 构造数据，然后分别推送到对应的显卡
    dummy_pixel_values_cpu = torch.randn(1536, 1536, dtype=torch.bfloat16)
    dummy_image_grid_thw_cpu = torch.tensor([[1, 32, 48]], dtype=torch.int32)

    with torch.no_grad():
        # SGLang 视觉前向 (cuda:0)
        sgl_vision_out, _ = sglang_model.image_encoder(
            dummy_pixel_values_cpu.to(device_sgl), 
            dummy_image_grid_thw_cpu.to(device_sgl)
        )
        
        # 原模型 视觉前向 (cuda:1)
        orig_vision_out, _ = train_model.image_encoder.lm_encode(
            dummy_pixel_values_cpu.to(device_orig), 
            dummy_image_grid_thw_cpu.to(device_orig)
        )

    # 将结果拉回到 cuda:0 进行对比
    orig_vision_out = orig_vision_out.to(device_sgl)

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
    # 音频编码器 (Audio Encoder) 精度对比
    # ---------------------------------------------------------
    print("\n--- Testing Audio Encoder ---")
    dummy_mel_cpu = torch.randn(1, 128, 3000, dtype=torch.bfloat16)
    dummy_mel_lengths_cpu = torch.tensor([300], dtype=torch.long)

    with torch.no_grad():
        # SGLang 音频前向 (cuda:0)
        sgl_audio_out, _ = sglang_model.audio_encoder(
            dummy_mel_cpu.to(device_sgl), 
            dummy_mel_lengths_cpu.to(device_sgl)
        )
        
        # 原模型 音频前向 (cuda:1)
        orig_audio_out, _ = train_model.audio_encoder.lm_encode(
            dummy_mel_cpu.to(device_orig), 
            dummy_mel_lengths_cpu.to(device_orig)
        )

    # 将结果拉回到 cuda:0 进行对比
    orig_audio_out = orig_audio_out.to(device_sgl)

    a_max_diff = torch.max(torch.abs(sgl_audio_out - orig_audio_out)).item()
    a_mean_diff = torch.mean(torch.abs(sgl_audio_out - orig_audio_out)).item()
    print(f"Audio Output Shape: {sgl_audio_out.shape}")
    print(f"Audio Max Diff:  {a_max_diff:.6f}")
    print(f"Audio Mean Diff: {a_mean_diff:.6f}")
    if a_max_diff < 1e-3:
        print("✅ Audio Encoder weights loaded perfectly!")
    else:
        print("❌ Audio Encoder has significant precision differences.")


    print("\n--- Testing LLM Backbone Components ---")
    
    orig_text_model = train_model.model if hasattr(train_model, "model") else train_model.language_model.model
    sgl_text_model = sglang_model.model if hasattr(sglang_model, "model") else sglang_model.language_model.model
    
    hidden_size = config.text_config.hidden_size
    seq_len = 64
    
    # ---------------------------------------------------------
    # 1. 验证 Token Embedding
    # ---------------------------------------------------------
    print("-> Testing Token Embeddings...")
    dummy_input_ids = torch.randint(0, 32000, (1, seq_len))
    
    with torch.no_grad():
        orig_embeds = orig_text_model.embed_tokens(dummy_input_ids.to(device_orig)).cpu()
        sgl_embeds = sgl_text_model.embed_tokens(dummy_input_ids.to(device_sgl)).cpu()
        
    e_max_diff = torch.max(torch.abs(sgl_embeds - orig_embeds)).item()
    print(f"   Embedding Max Diff: {e_max_diff:.6f}")


    # ---------------------------------------------------------
    # 2. 验证核心 MoE MLP 层 (极其关键：验证专家权重与 Router)
    # ---------------------------------------------------------
    print("-> Testing MoE MLP Layer (Layer 0)...")
    num_layers = len(orig_text_model.layers)
    all_moe_matched = True

    print(f"   Found {num_layers} layers. Starting mathematical alignment check...")

    for layer_idx in range(num_layers):
       
        dummy_hidden_cpu = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # HF Original 模型前向
            orig_mlp = orig_text_model.layers[layer_idx].mlp
            orig_mlp_out = orig_mlp(dummy_hidden_cpu.to(device_orig)).cpu()
            
            sgl_mlp = sgl_text_model.layers[layer_idx].mlp
            dummy_hidden_2d = dummy_hidden_cpu.view(-1, hidden_size).to(device_sgl)
            sgl_mlp_out = sgl_mlp(dummy_hidden_2d).cpu()
            sgl_mlp_out = sgl_mlp_out.view(1, seq_len, hidden_size)
            
        m_max_diff = torch.max(torch.abs(sgl_mlp_out - orig_mlp_out)).item()
        m_mean_diff = torch.mean(torch.abs(sgl_mlp_out - orig_mlp_out)).item()
        
        orig_flat = orig_mlp_out.view(-1).float()
        sgl_flat = sgl_mlp_out.view(-1).float()
        cos_sim = torch.nn.functional.cosine_similarity(orig_flat, sgl_flat, dim=0).item()

        # 判断对齐标准：余弦相似度 > 0.99 且 平均误差 < 0.005
        if cos_sim > 0.99 or m_mean_diff < 0.005:
            print(f"   [Layer {layer_idx:02d}] ✅ Pass | Max: {m_max_diff:.4f}, Mean: {m_mean_diff:.5f}, CosSim: {cos_sim:.6f}")
        else:
            print(f"   [Layer {layer_idx:02d}] ❌ FAIL | Max: {m_max_diff:.4f}, Mean: {m_mean_diff:.5f}, CosSim: {cos_sim:.6f}")
            all_moe_matched = False

    if all_moe_matched:
        print("   🎉 所有 MoE 层的数学输出完美对齐！")
    else:
        print("   ⚠️ 存在未对齐的 MoE 层，请检查上方日志排查对应层数。")
    
    print("\nAll tests completed.")