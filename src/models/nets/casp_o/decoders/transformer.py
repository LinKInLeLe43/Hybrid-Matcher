from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..utils import gather_attended

try:
    # This will be deprecated after torch 2.3.0, see https://github.com/pytorch/pytorch/releases/tag/v2.3.0
    from torch.backends.cuda import sdp_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False


def apply_rotary_emb(x: Tensor, encoding: Tensor) -> Tensor:
    freqs_sin, freqs_cos = encoding.unflatten(-1, (-1, 2)).chunk(2, dim=-1)
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack([-x2, x1], dim=-1)
    x = (x * freqs_cos + rotated_x * freqs_sin).flatten(start_dim=-2)
    return x


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
        inf = 1e9 if self.training else float("inf")
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
                similarity.masked_fill_(~mask, -inf)
            attention = similarity.softmax(dim=-1)
            message = attention @ v
        if mask is not None:
            message = message.nan_to_num()
        return message


class SelfBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        if stride > 1:
            self.patch_embed = nn.Conv2d(
                dim, dim, stride, stride=stride, groups=dim, bias=bias
            )
            # self.patch_embed1 = nn.MaxPool2d(stride, stride=stride)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        encoding: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n, h, w, c = x.shape
        mask = mask[:, None] if mask is not None else None

        x_ = x
        if self.stride > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.patch_embed(x_).permute(0, 2, 3, 1)
            h, w = h // self.stride, w // self.stride

        q, k, v = (
            self.qkv_proj(x_)
            .reshape(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        if encoding is not None:
            encoding = (
                encoding[:h, :w, :c]
                .reshape(-1, self.num_heads, self.head_dim)
                .transpose(-3, -2)
            )
            q = apply_rotary_emb(q, encoding)
            k = apply_rotary_emb(k, encoding)
        message = (
            self.attention(q, k, v, mask=mask)
            .transpose(-3, -2)
            .reshape(n, h, w, c)
        )
        message = self.norm1(self.out_proj(message))
        if self.stride > 1:
            message = message.permute(0, 3, 1, 2).contiguous()
            message = F.interpolate(
                message,
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        x = x + self.norm2(self.ffn(torch.cat([x, message], dim=-1)))
        return x


class CrossBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        if stride > 1:
            self.patch_embed = nn.Conv2d(
                dim, dim, stride, stride=stride, groups=dim, bias=bias
            )
            # self.patch_embed1 = nn.MaxPool2d(stride, stride=stride)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        n, h0, w0, c = x0.shape
        _, h1, w1, _ = x1.shape
        mask01 = mask[:, None] if mask is not None else None
        mask10 = mask01.transpose(-2, -1) if mask is not None else None

        x0_, x1_ = x0, x1
        if self.stride > 1:
            x0_, x1_ = x0_.permute(0, 3, 1, 2), x1_.permute(0, 3, 1, 2)
            x0_ = self.patch_embed(x0_).permute(0, 2, 3, 1)
            x1_ = self.patch_embed(x1_).permute(0, 2, 3, 1)
            h0, w0, h1, w1 = [t // self.stride for t in [h0, w0, h1, w1]]

        q0, k0, v0 = (
            self.qkv_proj(x0_)
            .reshape(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1_)
            .reshape(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        message0 = (
            self.attention(q0, k1, v1, mask=mask01)
            .transpose(-3, -2)
            .reshape(n, h0, w0, c)
        )
        message1 = (
            self.attention(q1, k0, v0, mask=mask10)
            .transpose(-3, -2)
            .reshape(n, h1, w1, c)
        )
        message0 = self.norm1(self.out_proj(message0))
        message1 = self.norm1(self.out_proj(message1))
        if self.stride > 1:
            message0 = message0.permute(0, 3, 1, 2).contiguous()
            message1 = message1.permute(0, 3, 1, 2).contiguous()
            message0 = F.interpolate(
                message0,
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            message1 = F.interpolate(
                message1,
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        x0 = x0 + self.norm2(self.ffn(torch.cat([x0, message0], dim=-1)))
        x1 = x1 + self.norm2(self.ffn(torch.cat([x1, message1], dim=-1)))
        return x0, x1


class TransformerLayer(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.self_block = SelfBlock(*args, **kwargs)
        self.cross_block = CrossBlock(*args, **kwargs)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        encoding0: Optional[Tensor] = None,
        encoding1: Optional[Tensor] = None,
        mask00: Optional[Tensor] = None,
        mask11: Optional[Tensor] = None,
        mask01: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0 = self.self_block(x0, encoding=encoding0, mask=mask00)
        x1 = self.self_block(x1, encoding=encoding1, mask=mask11)
        x0, x1 = self.cross_block(x0, x1, mask=mask01)
        return x0, x1


class RegionSelectiveCrossBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        q0, k0, v0 = (
            self.qkv_proj(x0)
            .unflatten(-1, (self.num_heads * self.head_dim, 3))
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1)
            .unflatten(-1, (self.num_heads * self.head_dim, 3))
            .unbind(dim=-1)
        )
        k0 = gather_attended(k0, indices1_to_0)
        v0 = gather_attended(v0, indices1_to_0)
        k1 = gather_attended(k1, indices0_to_1)
        v1 = gather_attended(v1, indices0_to_1)
        q0, k0, v0, q1, k1, v1 = [
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
            for t in [q0, k0, v0, q1, k1, v1]
        ]
        message0 = (
            self.attention(q0, k1, v1).transpose(-3, -2).flatten(start_dim=-2)
        )
        message1 = (
            self.attention(q1, k0, v0).transpose(-3, -2).flatten(start_dim=-2)
        )
        message0 = self.norm1(self.out_proj(message0))
        message1 = self.norm1(self.out_proj(message1))
        x0 = x0 + self.norm2(self.ffn(torch.cat([x0, message0], dim=-1)))
        x1 = x1 + self.norm2(self.ffn(torch.cat([x1, message1], dim=-1)))
        return x0, x1


RegionSelectiveTransformerLayer = RegionSelectiveCrossBlock
