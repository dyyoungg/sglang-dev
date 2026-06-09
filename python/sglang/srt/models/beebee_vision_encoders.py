

import logging
from functools import partial, lru_cache
from typing import  List, Optional, Tuple, Type, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention, BATCH_BUCKETS, FLASHINFER_MAX_SEQLEN_BUCKETS, FLASHINFER_WORKSPACE_SIZE_BYTES
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)

from sglang.srt.configs.beebeeomni_config import BeeBeeVisionConfig
from sglang.srt.configs.beebeeomni_moe_config import BeeBeeMoEVisionConfig
from sglang.srt.models.utils import RotaryPosMixin, permute_inv, compute_cu_seqlens_from_grid_numpy
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu, round_up, is_cpu

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


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

class BeeBeeQwen25VisionModel(nn.Module, RotaryPosMixin):

    def __init__(
        self,
        vision_config: BeeBeeVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        downsample_ratio: int = 16,
        use_data_parallel: bool = False,
        max_context_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = vision_config
        self.downsample_ratio = downsample_ratio
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

        self.embed_dim = hidden_size * self.spatial_merge_unit
        self.mm_projector = DynamicAvgPoolProjector(
            encoder_hidden=self.embed_dim,
            out_hidden=vision_config.output_size,
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

       
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], 
            grid_thw[:, 0]
        ).cumsum(dim=0).to(device=x.device, dtype=torch.int32)
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
        return features, seq_lens

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
        
        return features, seq_lens
    

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
            if torch.cuda.is_available() and (not is_npu()):
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