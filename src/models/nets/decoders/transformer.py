from copy import deepcopy
from typing import Optional, Sequence, Tuple
from warnings import warn

import torch
from kornia import create_meshgrid
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
        scale: int,
        dim: int,
        prompt_dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.scale = scale
        self.dim = dim
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )

        if scale > 1:
            self.down_proj = nn.Conv2d(
                dim, dim, self.scale, stride=self.scale, groups=dim, bias=bias
            )
            self.up_proj = nn.ConvTranspose2d(
                dim, dim, self.scale, stride=self.scale, groups=dim, bias=bias
            )
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.prompt_proj = nn.Linear(prompt_dim, prompt_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=bias),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=bias),
        )
        self.prompt_ffn = nn.Sequential(
            nn.Linear(dim + prompt_dim, dim + prompt_dim, bias=bias),
            nn.LayerNorm(dim + prompt_dim),
            nn.GELU(),
            nn.Linear(dim + prompt_dim, dim, bias=bias),
        )

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
        x_rotated = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return x_rotated

    def _apply_rotary_encoding(self, x: Tensor, encoding: Tensor) -> Tensor:
        x = x * encoding[0] + self._rotate_half(x) * encoding[1]
        return x

    def forward(
        self,
        x: Tensor,
        encoding: Tensor,
        prompt: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x_ = x
        # if self.scale != 1:
        #     x_ = self.down_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        q, k, v = (
            self.qkv_proj(x_.flatten(start_dim=1, end_dim=2))
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        n, h, w, c = x_.shape
        encoding = (
            encoding[:, :h, :w, :c]
            .flatten(start_dim=1, end_dim=2)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(1, 2)
        )
        q = self._apply_rotary_encoding(q, encoding)
        k = self._apply_rotary_encoding(k, encoding)
        if prompt is not None:
            prompt = (
                self.prompt_proj(prompt)
                .flatten(start_dim=1, end_dim=2)
                .unflatten(-1, (self.num_heads, -1))
                .transpose(1, 2)
            )
            v = torch.cat([v, prompt], dim=-1)
        message = self.attention(q, k, v, mask=mask)
        message = message.permute(0, 2, 3, 1).reshape(n, h, w, -1)
        # if self.scale != 1:
        #     message = self.up_proj(
        #         message.permute(0, 3, 1, 2).contiguous()
        #     ).permute(0, 2, 3, 1)
        if prompt is not None:
            message, prompt = message.split(
                (self.dim, self.prompt_dim), dim=-1
            )
        x = x + self.ffn(torch.cat([x, message], dim=-1))
        if prompt is not None:
            x = x + self.prompt_ffn(torch.cat([x, prompt], dim=-1))
        return x


class CrossBlock(Module):
    def __init__(
        self,
        scale: int,
        dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.scale = scale
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )

        if scale > 1:
            self.down_proj = nn.Conv2d(
                dim, dim, self.scale, stride=self.scale, groups=dim, bias=bias
            )
            self.up_proj = nn.ConvTranspose2d(
                dim, dim, self.scale, stride=self.scale, groups=dim, bias=bias
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
        x0_ = x0
        x1_ = x1
        if self.scale != 1:
            x0_ = self.down_proj(x0.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x1_ = self.down_proj(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        q0, k0, v0 = (
            self.qkv_proj(x0_.flatten(start_dim=1, end_dim=2))
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1_.flatten(start_dim=1, end_dim=2))
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        _, h0, w0, c = x0_.shape
        _, h1, w1, c = x1_.shape
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
        if self.scale != 1:
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
        encoding: Tensor,
        prompt0: Optional[Tensor] = None,
        prompt1: Optional[Tensor] = None,
        mask00: Optional[Tensor] = None,
        mask11: Optional[Tensor] = None,
        mask01: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0 = self.self_block(x0, encoding, prompt=prompt0, mask=mask00)
        x1 = self.self_block(x1, encoding, prompt=prompt1, mask=mask11)
        x0, x1 = self.cross_block(x0, x1, mask=mask01)
        return x0, x1


def _gather(x: Tensor, indices: Tensor) -> Tensor:
    out = x[
        torch.arange(x.shape[0], device=x.device)[:, None, None], indices
    ].flatten(start_dim=2, end_dim=3)
    return out


class RegionSelectiveCrossBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        scale: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale
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
        sh, sw = self.scale, self.scale
        fh, fw = size[0] // sh, size[1] // sw
        c = x0.shape[-1]

        q = (
            self.q_proj(x0)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(2, 3)
        )
        k, v = (
            _gather(self.kv_proj(x1), indices0_to_1)
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
            .reshape(-1, fh, fw, sh, sw, 2 * c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(-1, 2 * c, size[0], size[1])
        )
        message = (
            self.ffn(message)
            .reshape(-1, c, fh, sh, fw, sw)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, fh * fw, sh * sw, c)
        )
        x0 = x0 + self.norm2(message)
        return x0


class RegionSelectiveTransformerLayer(Module):
    def __init__(self, num_layers: int, **kwargs) -> None:
        super().__init__()
        layer = RegionSelectiveCrossBlock(**kwargs)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )
        self.scale = kwargs["scale"]
        delta_indices = create_meshgrid(
            self.scale,
            self.scale,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("delta_indices", delta_indices, persistent=False)

    def map_indices(
        self, x: Tensor, size: Sequence[int], fw: int
    ) -> torch.Tensor:
        row = (x[..., None] // fw) * self.scale + self.delta_indices[:, 1]
        col = (x[..., None] % fw) * self.scale + self.delta_indices[:, 0]
        out = row * fw * self.scale + col
        out = (
            out.unflatten(1, size)
            .repeat_interleave(self.scale, dim=1)
            .repeat_interleave(self.scale, dim=2)
            .flatten(start_dim=1, end_dim=2)
            .flatten(start_dim=-2)
        )
        return out

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        size0: Sequence[int],
        size1: Sequence[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n = indices0_to_1.shape[0]
        sh = sw = self.scale
        h0, w0 = size0
        h1, w1 = size1
        fh0, fw0, fh1, fw1 = h0 // sh, w0 // sw, h1 // sh, w1 // sw

        indices1_to_0 = indices1_to_0.transpose(1, 2)
        # range = torch.arange(n, device=x0.device)[:, None, None]
        # _indices0_to_1 = (indices0_to_1 + fh1 * fw1 * range).flatten(end_dim=1)
        # _indices1_to_0 = (indices1_to_0 + fh0 * fw0 * range).flatten(end_dim=1)

        for layer in self.layers:
            x0 = layer(x0, x1, indices0_to_1, (h0, w0))
            x1 = layer(x1, x0, indices1_to_0, (h1, w1))
        selective0 = _gather(x0, indices1_to_0)
        selective1 = _gather(x1, indices0_to_1)

        indices0_to_1 = self.map_indices(indices0_to_1, (fh0, fw0), fw1)
        indices1_to_0 = self.map_indices(indices1_to_0, (fh1, fw1), fw0)
        indices1_to_0 = indices1_to_0.transpose(1, 2)
        return x0, x1, selective0, selective1, indices0_to_1, indices1_to_0
