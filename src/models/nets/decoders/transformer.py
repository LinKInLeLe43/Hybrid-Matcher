from typing import Optional, Sequence, Tuple
from warnings import warn

import torch
from torch import Tensor, nn
from torch.nn import Module

try:
    # deprecated after torch 2.3.0, see https://github.com/pytorch/pytorch/releases/tag/v2.3.0
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
            warn("", stacklevel=2)
        self.enable_sdpa = enable_sdpa and SDPA_AVAILABLE
        if enable_flash and not self.enable_sdpa:
            warn("", stacklevel=2)
        self.enable_flash = enable_flash and self.enable_sdpa

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if self.enable_sdpa:
            if self.enable_flash:
                assert mask is None
                q, k, v = [x.half().contiguous() for x in [q, k, v]]
                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    message = sdpa(q, k, v).to(q.dtype)
            else:
                q, k, v = [x.contiguous() for x in [q, k, v]]
                message = sdpa(q, k, v, attn_mask=mask)
        else:
            q = q * q.shape[-1] ** -0.5
            similarity = q @ k.transpose(-1, -2)
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
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        if stride > 1:
            self.down_proj = nn.Conv2d(
                dim,
                dim,
                self.stride,
                stride=self.stride,
                groups=dim,
                bias=bias,
            )
            self.up_proj = nn.ConvTranspose2d(
                dim,
                dim,
                self.stride,
                stride=self.stride,
                groups=dim,
                bias=bias,
            )
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=bias),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=bias),
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
        _, h, w, c = x.shape

        _x = x
        if self.stride != 1:
            _x = self.down_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h, w = h // self.stride, w // self.stride
        q, k, v = (
            self.qkv_proj(_x)
            .flatten(start_dim=1, end_dim=2)
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        if encoding is not None:
            encoding = (
                encoding[:, :h, :w, :c]
                .flatten(start_dim=1, end_dim=2)
                .unflatten(-1, (self.num_heads, self.head_dim))
                .transpose(1, 2)
            )
            q = self._apply_rotary_encoding(q, encoding)
            k = self._apply_rotary_encoding(k, encoding)
        message = self.attention(q, k, v, mask=mask)
        message = self.out_proj(message.transpose(1, 2).flatten(start_dim=-2))
        message = message.unflatten(1, (h, w))
        if self.stride != 1:
            message = self.up_proj(
                message.permute(0, 3, 1, 2).contiguous()
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
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.stride = stride

        if stride > 1:
            self.down_proj = nn.Conv2d(
                dim,
                dim,
                self.stride,
                stride=self.stride,
                groups=dim,
                bias=bias,
            )
            self.up_proj = nn.ConvTranspose2d(
                dim,
                dim,
                self.stride,
                stride=self.stride,
                groups=dim,
                bias=bias,
            )
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=bias),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=bias),
        )

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        _, h0, w0, _ = x0.shape
        _, h1, w1, _ = x1.shape

        _x0, _x1 = x0, x1
        if self.stride != 1:
            _x0 = self.down_proj(x0.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            _x1 = self.down_proj(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            h0, w0 = h0 // self.stride, w0 // self.stride
            h1, w1 = h1 // self.stride, w1 // self.stride
        q0, k0, v0 = (
            self.qkv_proj(_x0)
            .flatten(start_dim=1, end_dim=2)
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(_x1)
            .flatten(start_dim=1, end_dim=2)
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        message0 = self.attention(q0, k1, v1, mask=mask)
        message1 = self.attention(
            q1,
            k0,
            v0,
            mask=mask.transpose(-1, -2) if mask is not None else None,
        )
        message0 = self.out_proj(
            message0.transpose(1, 2).flatten(start_dim=-2)
        )
        message1 = self.out_proj(
            message1.transpose(1, 2).flatten(start_dim=-2)
        )
        message0 = message0.unflatten(1, (h0, w0))
        message1 = message1.unflatten(1, (h1, w1))
        if self.stride != 1:
            message0 = self.up_proj(
                message0.permute(0, 3, 1, 2).contiguous()
            ).permute(0, 2, 3, 1)
            message1 = self.up_proj(
                message1.permute(0, 3, 1, 2).contiguous()
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
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.stride = stride
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=bias),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        size: Tuple[int, int],
    ) -> Tensor:
        n = x0.shape[0]
        fh, fw = size

        q = (
            self.q_proj(x0)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(2, 3)
        )
        k, v = (
            self.kv_proj(x1)[
                torch.arange(n, device=x0.device)[:, None, None], indices0_to_1
            ]
            .flatten(start_dim=2, end_dim=3)
            .unflatten(-1, (self.num_heads, self.head_dim, 2))
            .transpose(2, 3)
            .unbind(dim=-1)
        )
        message = self.attention(q, k, v)
        message = self.norm1(
            self.out_proj(message.transpose(2, 3).flatten(start_dim=-2))
        )
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
        size0: Sequence[int],
        size1: Sequence[int],
    ) -> Tuple[Tensor, Tensor]:
        assert len(size0) == 2 and len(size1) == 2
        x0 = self.block(x0, x1, indices0_to_1, size0)
        x1 = self.block(x1, x0, indices1_to_0, size1)
        return x0, x1
