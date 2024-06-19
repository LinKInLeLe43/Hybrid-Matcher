from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

if hasattr(F, "scaled_dot_product_attention"):
    from torch.backends.cuda import sdp_kernel
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

from .mlp import Mlp


class Attention(nn.Module):
    def __init__(
        self,
        use_liner: bool = False,
        try_sdpa: bool = False,
        try_flash: bool = False
    ) -> None:
        super().__init__()
        self.use_linear = use_liner
        self.use_sdpa = not use_liner and try_sdpa and FLASH_AVAILABLE
        self.force_flash = try_flash and self.use_sdpa

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: detect NAN
        if self.use_linear:
            if attn_mask is not None:
                raise ValueError("")

            q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
            if q_mask is not None:
                q = q_mask[:, None, :, None] * q
            if kv_mask is not None:
                k = kv_mask[:, None, :, None] * k
                v = kv_mask[:, None, :, None] * v

            s = v.shape[2]
            kv = torch.einsum("...sd,...sc->...dc", k, v / s)
            div = 1 / (torch.einsum("...ld,...sd->...l", q, k) + 1e-6)
            out = s * torch.einsum("...ld,...dc,...l->...lc", q, kv, div)
        else:
            mask = None
            if q_mask is not None and kv_mask is not None:
                mask = q_mask[:, None, :, None] & kv_mask[:, None, None, :]
                mask = (torch.zeros_like(mask, dtype=q.dtype)
                        .masked_fill(~mask, float("-inf")))

            if attn_mask is not None:
                mask = attn_mask if mask is None else mask + attn_mask

            if not self.use_sdpa:
                d = q.shape[3]
                similarity = torch.einsum("...ld,...sd->...ls", q, k) / d ** 0.5
                if mask is not None:
                    similarity = similarity + mask
                attn = F.softmax(similarity, dim=3)
                out = torch.einsum("...ls,...sc->...lc", attn, v)
            else:
                if self.force_flash:
                    args = [x.half().contiguous() for x in [q, k, v]]
                    with sdp_kernel(
                        enable_math=False, enable_flash=True,
                        enable_mem_efficient=False):
                        out = F.scaled_dot_product_attention(
                            *args, attn_mask=mask).to(q.dtype)
                else:
                    args = [x.contiguous() for x in [q, k, v]]
                    out = F.scaled_dot_product_attention(
                        *args, attn_mask=mask)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        head_count: int,
        bias: bool = True,
        use_liner: bool = False,
        try_sdpa: bool = False,
        try_flash: bool = False,
        mlp_kernel_size0: int = 1,
        mlp_kernel_size1: int = 1
    ) -> None:
        super().__init__()
        self.head_count = head_count

        self.qkv_proj = nn.Linear(depth, 3 * depth, bias=bias)
        self.attention = Attention(
            use_liner=use_liner, try_sdpa=try_sdpa, try_flash=try_flash)
        self.merge = nn.Linear(depth, depth, bias=bias)
        self.norm0 = nn.LayerNorm(depth)

        self.mlp = Mlp(
            2 * depth, 2 * depth, depth, bias=bias,
            kernel_size0=mlp_kernel_size0, kernel_size1=mlp_kernel_size1)
        self.norm1 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        qkv = qkv.unflatten(2, (self.head_count, -1)).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=3)
        message = self.attention(q, k, v, q_mask=mask, kv_mask=mask)
        message = message.transpose(1, 2).flatten(start_dim=2)
        message = self.merge(message)
        message = self.norm0(message)

        message = torch.cat([x, message], dim=2)
        message = self.mlp(message)
        message = self.norm1(message)

        out = x + message
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        head_count: int,
        bias: bool = True,
        use_liner: bool = False,
        try_sdpa: bool = False,
        try_flash: bool = False,
        mlp_kernel_size0: int = 1,
        mlp_kernel_size1: int = 1
    ) -> None:
        super().__init__()
        self.head_count = head_count

        self.q_proj = nn.Linear(depth, depth, bias=bias)
        self.kv_proj = nn.Linear(depth, 2 * depth, bias=bias)
        self.attention = Attention(
            use_liner=use_liner, try_sdpa=try_sdpa, try_flash=try_flash)
        self.merge = nn.Linear(depth, depth, bias=bias)
        self.norm0 = nn.LayerNorm(depth)

        self.mlp = Mlp(
            2 * depth, 2 * depth, depth, bias=bias,
            kernel_size0=mlp_kernel_size0, kernel_size1=mlp_kernel_size1)
        self.norm1 = nn.LayerNorm(depth)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, kv = self.q_proj(x0), self.kv_proj(x1)
        q = q.unflatten(2, (self.head_count, -1)).transpose(1, 2)
        kv = kv.unflatten(2, (self.head_count, -1)).transpose(1, 2)
        k, v = kv.chunk(2, dim=3)
        message = self.attention(q, k, v, q_mask=mask0, kv_mask=mask1)
        message = message.transpose(1, 2).flatten(start_dim=2)
        message = self.merge(message)
        message = self.norm0(message)

        message = torch.cat([x0, message], dim=2)
        message = self.mlp(message)
        message = self.norm1(message)

        out = x0 + message
        return out
