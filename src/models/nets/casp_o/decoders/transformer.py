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


def apply_rotary_emb(x: Tensor, encoding: Tensor) -> Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    rotated_x = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
    x = x * encoding[0] + rotated_x * encoding[1]
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
        patch_size: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size

        if patch_size > 1:
            self.patch_embed = nn.Conv2d(
                dim, dim, patch_size, stride=patch_size, groups=dim, bias=bias
            )
            # self.patch_embed1 = nn.MaxPool2d(stride, stride=stride)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=bias),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
        )

    def forward(
        self,
        x: Tensor,
        encoding: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n, h, w, c = x.shape

        x_ = x
        if self.patch_size > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.patch_embed(x_).permute(0, 2, 3, 1)
            h, w = h // self.patch_size, w // self.patch_size

        q, k, v = (
            self.qkv_proj(x_)
            .reshape(n, -1, self.num_heads, self.head_dim, 3)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        if encoding is not None:
            encoding = (
                encoding[:, :h, :w, :c]
                .reshape(2, -1, self.num_heads, self.head_dim)
                .transpose(-3, -2)
            )
            q = apply_rotary_emb(q, encoding)
            k = apply_rotary_emb(k, encoding)
        message = (
            self.attention(q, k, v, mask=mask)
            .transpose(-3, -2)
            .reshape(n, h, w, c)
        )
        message = self.out_proj(message)
        if self.patch_size > 1:
            message = message.permute(0, 3, 1, 2).contiguous()
            message = F.interpolate(
                message,
                scale_factor=self.patch_size,
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
        patch_size: int = 1,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size

        if patch_size > 1:
            self.patch_embed = nn.Conv2d(
                dim, dim, patch_size, stride=patch_size, groups=dim, bias=bias
            )
            # self.patch_embed1 = nn.MaxPool2d(stride, stride=stride)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
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
        n, h0, w0, c = x0.shape
        _, h1, w1, _ = x1.shape

        x0_, x1_ = x0, x1
        if self.patch_size > 1:
            x0_, x1_ = x0_.permute(0, 3, 1, 2), x1_.permute(0, 3, 1, 2)
            x0_ = self.patch_embed(x0_).permute(0, 2, 3, 1)
            x1_ = self.patch_embed(x1_).permute(0, 2, 3, 1)
            h0, w0, h1, w1 = [t // self.patch_size for t in [h0, w0, h1, w1]]

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
            self.attention(q0, k1, v1, mask=mask)
            .transpose(-3, -2)
            .reshape(n, h0, w0, c)
        )
        message1 = (
            self.attention(
                q1,
                k0,
                v0,
                mask=mask.transpose(-1, -2) if mask is not None else None,
            )
            .transpose(-3, -2)
            .reshape(n, h1, w1, c)
        )
        message0 = self.out_proj(message0)
        message1 = self.out_proj(message1)
        if self.patch_size > 1:
            message0 = message0.permute(0, 3, 1, 2).contiguous()
            message1 = message1.permute(0, 3, 1, 2).contiguous()
            message0 = F.interpolate(
                message0,
                scale_factor=self.patch_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            message1 = F.interpolate(
                message1,
                scale_factor=self.patch_size,
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
        stride: int,
        dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stride = stride

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=bias)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        size: Tuple[int, int],
    ) -> Tensor:
        n, l, _, _ = x0.shape
        fh, fw = size

        q = (
            self.q_proj(x0)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(-3, -2)
        )
        k, v = (
            self.kv_proj(x1)[
                torch.arange(n, device=x0.device)[:, None, None], indices0_to_1
            ]
            .reshape(n, l, -1, self.num_heads, self.head_dim, 2)
            .transpose(-4, -3)
            .unbind(dim=-1)
        )
        message = (
            self.attention(q, k, v).transpose(-3, -2).flatten(start_dim=-2)
        )
        message = self.norm1(self.out_proj(message))
        message = (
            torch.cat([x0, message], dim=-1)
            .reshape(n, fh, fw, self.stride, self.stride, -1)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(n, -1, fh * self.stride, fw * self.stride)
        )
        message = (
            self.ffn(message)
            .reshape(n, -1, fh, self.stride, fw, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh * fw, self.stride * self.stride, -1)
        )
        x0 = x0 + self.norm2(message)
        return x0


class RegionSelectiveTransformerLayer(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.block = RegionSelectiveCrossBlock(*args, **kwargs)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        assert len(size0) == 2 and len(size1) == 2
        x0 = self.block(x0, x1, indices0_to_1, size0)
        x1 = self.block(x1, x0, indices1_to_0, size1)
        return x0, x1
