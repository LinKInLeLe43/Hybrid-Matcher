from copy import deepcopy
from typing import Optional, Sequence, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .utils import window_partition, window_unpartition

try:
    # This will be deprecated after torch 2.3.0, see https://github.com/pytorch/pytorch/releases/tag/v2.3.0
    from torch.backends.cuda import sdp_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False


class Attention(Module):
    def __init__(
        self, enable_sdpa: bool = False, enable_flash: bool = False
    ) -> None:
        super().__init__()
        if enable_sdpa and not SDPA_AVAILABLE:
            warn(
                "`scaled_dot_product_attention` (SDPA) is not available. "
                "Consider installing PyTorch >= 2.0.",
                stacklevel=2,
            )
        self.enable_sdpa = enable_sdpa and SDPA_AVAILABLE
        if enable_flash and not self.enable_sdpa:
            warn("Flash attention requires SDPA to be enabled.", stacklevel=2)
        self.enable_flash = enable_flash and self.enable_sdpa

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if self.enable_sdpa:
            if self.enable_flash:
                assert mask is None
                q, k, v = [t.half().contiguous() for t in [q, k, v]]
                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    message = sdpa(q, k, v).to(q.dtype)
            else:
                q, k, v = [t.contiguous() for t in [q, k, v]]
                message = sdpa(q, k, v, attn_mask=mask)
        else:
            q = q * q.shape[-1] ** -0.5
            similarity = q @ k.transpose(-2, -1)
            if mask is not None:
                similarity.masked_fill_(~mask, -float("inf"))
            attention = similarity.softmax(dim=-1)
            message = attention @ v
        if mask is not None:
            message.nan_to_num_()
        return message


class TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        if stride > 1:
            self.down_q = nn.Conv2d(
                dim, dim, stride, stride=stride, groups=dim, bias=False
            )
            self.down_kv = nn.MaxPool2d(stride, stride=stride)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        rope: Optional[Module] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n, _, h, w = x0.shape

        x0_, x1_ = x0, x1
        if self.stride > 1:
            x0_, x1_ = self.down_q(x0), self.down_kv(x1)
            h, w = h // self.stride, w // self.stride
        x0_, x1_ = x0_.permute(0, 2, 3, 1), x1_.permute(0, 2, 3, 1)
        q, k, v = self.q_proj(x0_), self.k_proj(x1_), self.v_proj(x1_)
        if rope is not None:
            q, k = rope.rel_pe(q), rope.rel_pe(k)
        q, k, v = [
            t.view(n, -1, self.num_heads, self.head_dim).transpose(-3, -2)
            for t in [q, k, v]
        ]
        message = (
            self.attention(q, k, v, mask=mask)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        message = (
            self.norm1(self.merge(message))
            .transpose(-2, -1)
            .unflatten(-1, (h, w))
        )
        if self.stride > 1:
            message = message.contiguous()
            message = F.interpolate(
                message,
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            )
        message = torch.cat([x0, message], dim=1).permute(0, 2, 3, 1)
        x0 = x0 + self.norm2(self.mlp(message)).permute(0, 3, 1, 2)
        return x0


class RegionSelectiveTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x0: Tensor, x1: Tensor, size: Sequence[int]) -> Tensor:
        q, k, v = self.q_proj(x0), self.k_proj(x1), self.v_proj(x1)
        q, k, v = [
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
            for t in [q, k, v]
        ]
        message = (
            self.attention(q, k, v).transpose(-3, -2).flatten(start_dim=-2)
        )
        message = self.norm1(self.merge(message))
        message = torch.cat([x0, message], dim=-1)
        message = window_unpartition(message, size, self.stride)
        message = self.mlp(message)
        message = window_partition(message, self.stride)
        x0 = x0 + self.norm2(message)
        return x0


class RegionSelectiveTransformer(Module):
    def __init__(
        self,
        dim_list: Sequence[int],
        num_heads: int,
        stride: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert len(dim_list) == 2
        self.stride = stride
        delta_indices = create_meshgrid(
            self.stride,
            self.stride,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("delta_indices", delta_indices, persistent=False)

        self.x_up = nn.Conv2d(dim_list[0], dim_list[1], 1, bias=False)
        self.y_up = nn.Conv2d(dim_list[1], dim_list[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(dim_list[1], dim_list[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_list[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_list[1], dim_list[0], 3, padding=1, bias=False),
        )
        layer = RegionSelectiveTransformerBlock(dim_list[0], num_heads, stride)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fuse(self, x_list: Sequence[Tensor]) -> Tensor:
        assert len(x_list) == 2
        x, y = self.x_up(x_list[0]), self.y_up(x_list[1])
        x = x + F.interpolate(
            y, scale_factor=self.stride, mode="bilinear", align_corners=False
        )
        x = self.down(x)
        return x

    def _gather_attended(self, x: Tensor, indices: Tensor) -> Tensor:
        x = x[
            torch.arange(x.shape[0], device=x.device)[:, None, None], indices
        ].flatten(start_dim=2, end_dim=3)
        return x

    def _map_indices(self, x: Tensor, size: Sequence[int], w: int) -> Tensor:
        row = (x[..., None] // w) * self.stride + self.delta_indices[:, 1]
        col = (x[..., None] % w) * self.stride + self.delta_indices[:, 0]
        x = (
            (row * w * self.stride + col)
            .view(x.shape[0], *size, -1)
            .repeat_interleave(self.stride, dim=1)
            .repeat_interleave(self.stride, dim=2)
            .flatten(start_dim=1, end_dim=2)
        )
        return x

    def forward(
        self,
        x0_list: Sequence[Tensor],
        x1_list: Sequence[Tensor],
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        _, _, fh0, fw0 = x0_list[-1].shape
        _, _, fh1, fw1 = x1_list[-1].shape

        if (fh0, fw0) == (fh1, fw1):
            x0, x1 = self._fuse(
                [torch.cat(t) for t in zip(x0_list, x1_list)]
            ).chunk(2)
        else:
            x0, x1 = self._fuse(x0_list), self._fuse(x1_list)
        x0 = window_partition(x0, self.stride)
        x1 = window_partition(x1, self.stride)

        for layer in self.layers:
            attended0 = self._gather_attended(x1, indices0_to_1)
            x0 = layer(x0, attended0, (fh0, fw0))
            attended1 = self._gather_attended(x0, indices1_to_0)
            x1 = layer(x1, attended1, (fh1, fw1))
        attended0 = self._gather_attended(x1, indices0_to_1)
        indices0_to_1 = self._map_indices(indices0_to_1, (fh0, fw0), fw1)
        indices1_to_0 = self._map_indices(indices1_to_0, (fh1, fw1), fw0)
        return x0, x1, attended0, attended1, indices0_to_1, indices1_to_0
