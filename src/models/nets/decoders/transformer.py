from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

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
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        if stride > 1:
            self.down_proj = nn.Conv2d(
                dim, dim, self.stride, stride=self.stride, groups=dim, bias=bias
            )
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
        rotated_x = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return rotated_x

    def _apply_rotary_encoding(self, x: Tensor, encoding: Tensor) -> Tensor:
        x = x * encoding[0] + self._rotate_half(x) * encoding[1]
        return x

    def forward(
        self,
        x: Tensor,
        encoding: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n, h, w, c = x.shape

        x_ = x
        if self.stride > 1:
            x_ = self.down_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h, w = h // self.stride, w // self.stride
        q, k, v = (
            self.qkv_proj(x_)
            .view(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        if encoding is not None:
            encoding = (
                encoding[:, :h, :w, :c]
                .contiguous()
                .view(2, -1, self.num_heads, self.head_dim)
                .transpose(-3, -2)
            )
            q = self._apply_rotary_encoding(q, encoding)
            k = self._apply_rotary_encoding(k, encoding)
        message = (
            self.attention(q, k, v, mask=mask)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        message = self.out_proj(message).unflatten(1, (h, w))
        if self.stride > 1:
            # [TODO]: test contiguous()
            message = F.interpolate(
                message.permute(0, 3, 1, 2),
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        x = x + self.ffn(torch.cat([x, message], dim=-1))
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
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        if stride > 1:
            self.down_proj = nn.Conv2d(
                dim, dim, self.stride, stride=self.stride, groups=dim, bias=bias
            )
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        n, h0, w0, _ = x0.shape
        _, h1, w1, _ = x1.shape

        x0_, x1_ = x0, x1
        if self.stride > 1:
            x0_ = self.down_proj(x0.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x1_ = self.down_proj(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h0, w0, h1, w1 = [t // self.stride for t in [h0, w0, h1, w1]]
        q0, k0, v0 = (
            self.qkv_proj(x0_)
            .view(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1_)
            .view(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        message0 = (
            self.attention(q0, k1, v1, mask=mask)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        message1 = (
            self.attention(
                q1,
                k0,
                v0,
                mask=mask.transpose(-2, -1) if mask is not None else None,
            )
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        message0 = self.out_proj(message0).unflatten(1, (h0, w0))
        message1 = self.out_proj(message1).unflatten(1, (h1, w1))
        if self.stride != 1:
            # [TODO]: test contiguous()
            message0 = F.interpolate(
                message0.permute(0, 3, 1, 2),
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            message1 = F.interpolate(
                message1.permute(0, 3, 1, 2),
                scale_factor=self.stride,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        x0 = x0 + self.ffn(torch.cat([x0, message0], dim=-1))
        x1 = x1 + self.ffn(torch.cat([x1, message1], dim=-1))
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
        encoding: Optional[Tensor] = None,
        mask00: Optional[Tensor] = None,
        mask11: Optional[Tensor] = None,
        mask01: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0 = self.self_block(x0, encoding=encoding, mask=mask00)
        x1 = self.self_block(x1, encoding=encoding, mask=mask11)
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
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        n_range = torch.arange(x0.shape[0], device=x0.device)[:, None, None]

        q0, k0, v0 = (
            self.qkv_proj(x0)
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1)
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .unbind(dim=-1)
        )
        k0 = k0[n_range, indices1_to_0].flatten(start_dim=2, end_dim=3)
        v0 = v0[n_range, indices1_to_0].flatten(start_dim=2, end_dim=3)
        k1 = k1[n_range, indices0_to_1].flatten(start_dim=2, end_dim=3)
        v1 = v1[n_range, indices0_to_1].flatten(start_dim=2, end_dim=3)
        q0, k0, v0, q1, k1, v1 = [
            t.transpose(-3, -2) for t in [q0, k0, v0, q1, k1, v1]
        ]
        message0 = (
            self.attention(q0, k1, v1).transpose(-3, -2).flatten(start_dim=-2)
        )
        message1 = (
            self.attention(q1, k0, v0).transpose(-3, -2).flatten(start_dim=-2)
        )
        message0, message1 = self.out_proj(message0), self.out_proj(message1)
        x0 = x0 + self.ffn(torch.cat([x0, message0], dim=-1))
        x1 = x1 + self.ffn(torch.cat([x1, message1], dim=-1))
        return x0, x1


RegionSelectiveTransformerLayer = RegionSelectiveCrossBlock
