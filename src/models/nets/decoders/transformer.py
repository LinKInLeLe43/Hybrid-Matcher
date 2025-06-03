from copy import deepcopy
from typing import Optional, Tuple
from warnings import warn

import einops
import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F

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


class TransformerEncoder(Module):
    def __init__(self, depth: int, heads_count: int) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.attention = Attention()

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False),
        )
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self, x: Tensor, source: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        fc = self.heads_count

        q = einops.rearrange(self.q_proj(x), "n l (fc sc) -> n fc l sc", fc=fc)
        k = einops.rearrange(
            self.k_proj(source), "n s (fc sc) -> n fc s sc", fc=fc
        )
        v = einops.rearrange(
            self.v_proj(source), "n s (fc sc) -> n fc s sc", fc=fc
        )
        out = self.attention(q, k, v, mask=mask)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)

        out = torch.cat([x, out], dim=2)
        out = self.mlp(out)
        out = self.norm2(out)

        out += x
        return out


class ConvTransformerEncoder(Module):
    def __init__(self, scale: int, depth: int, heads_count: int) -> None:
        super().__init__()
        self.scale = scale
        self.heads_count = heads_count
        self.attention = Attention()

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Conv2d(2 * depth, 2 * depth, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * depth, depth, 3, padding=1, bias=False),
        )
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self, x: Tensor, source: Tensor, size: Tuple[int, int]
    ) -> Tensor:
        sh, sw, fc = self.scale, self.scale, self.heads_count
        fh, fw = size[0] // sh, size[1] // sw

        q = einops.rearrange(self.q_proj(x), "n l (fc sc) -> n fc l sc", fc=fc)
        k = einops.rearrange(
            self.k_proj(source), "n s (fc sc) -> n fc s sc", fc=fc
        )
        v = einops.rearrange(
            self.v_proj(source), "n s (fc sc) -> n fc s sc", fc=fc
        )
        out = self.attention(q, k, v)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)

        out = torch.cat([x, out], dim=2)
        out = einops.rearrange(
            out,
            "(n fh fw) (sh sw) c -> n c (fh sh) (fw sw)",
            fh=fh,
            sh=sh,
            fw=fw,
            sw=sw,
        )
        out = self.mlp(out)
        out = einops.rearrange(
            out, "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c", sh=sh, sw=sw
        )
        out = self.norm2(out)

        out += x
        return out


class AggregatedEncoder(Module):
    def __init__(self, depth: int, heads_count: int, scale: int) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.scale = scale
        self.attention = Attention()

        self.down_q = nn.Identity()
        self.down_kv = nn.Identity()

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False),
        )
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: Tensor,
        source: Tensor,
        rope: Optional[Module] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        s, fc = self.scale, self.heads_count
        q = self.down_q(x).permute(0, 2, 3, 1)
        kv = self.down_kv(source).permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)

        if rope is not None:
            q, k = rope.rel_pe(q), rope.rel_pe(k)

        q = einops.rearrange(q, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        k = einops.rearrange(k, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        v = einops.rearrange(v, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        out = self.attention(q, k, v, mask=mask)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)
        out = out.transpose(1, 2).unflatten(2, (x.shape[2], x.shape[3]))
        # out = F.interpolate(out, scale_factor=s, mode="bilinear")

        out = torch.cat([x, out], dim=1)
        out = out.permute(0, 2, 3, 1)
        out = self.mlp(out)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        out += x
        return out


class SelfBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
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
        x_rotated = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return x_rotated

    def _apply_rotary_encoding(self, x: Tensor, encoding: Tensor) -> Tensor:
        x = x * encoding[0] + self._rotate_half(x) * encoding[1]
        return x

    def forward(
        self, x: Tensor, encoding: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        q, k, v = (
            self.qkv_proj(x.flatten(start_dim=1, end_dim=2))
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        encoding = (
            encoding.flatten(start_dim=1, end_dim=2)
            .unflatten(-1, (self.num_heads, self.head_dim))
            .transpose(1, 2)
        )
        q = self._apply_rotary_encoding(q, encoding)
        k = self._apply_rotary_encoding(k, encoding)
        message = self.attention(q, k, v, mask=mask)
        message = self.out_proj(message.transpose(1, 2).flatten(start_dim=-2))
        message = message.unflatten(1, (x.shape[1], x.shape[2]))
        x = x + self.ffn(torch.cat([x, message], dim=-1))
        return x


class CrossBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
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
        q0, k0, v0 = (
            self.qkv_proj(x0.flatten(start_dim=1, end_dim=2))
            .unflatten(-1, (self.num_heads, self.head_dim, 3))
            .transpose(1, 2)
            .unbind(dim=-1)
        )
        q1, k1, v1 = (
            self.qkv_proj(x1.flatten(start_dim=1, end_dim=2))
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
        message0 = message0.unflatten(1, (x0.shape[1], x0.shape[2]))
        message1 = message1.unflatten(1, (x1.shape[1], x1.shape[2]))
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
        encoding0: Tensor,
        encoding1: Tensor,
        mask00: Optional[Tensor] = None,
        mask11: Optional[Tensor] = None,
        mask01: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0 = self.self_block(x0, encoding0, mask00)
        x1 = self.self_block(x1, encoding1, mask11)
        x0, x1 = self.cross_block(x0, x1, mask01)
        return x0, x1


class FusedSelectiveTransformer(Module):
    def __init__(
        self,
        scale: int,
        depths: Tuple[int, int],
        encoder: Module,
        layer_count: int,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.x_up = nn.Conv2d(depths[0], depths[1], 1, bias=False)
        self.y_up = nn.Conv2d(depths[1], depths[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(depths[1], depths[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(depths[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depths[1], depths[0], 3, padding=1, bias=False),
        )

        self.layers = nn.ModuleList(
            [deepcopy(encoder) for _ in range(layer_count)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Tensor,
        y1: Tensor,
        idxes0_to_1: Tensor,
        idxes1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s = sh = sw = self.scale
        n, _, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape
        fh0, fw0, fh1, fw1 = h0 // sh, w0 // sw, h1 // sh, w1 // sw

        x, y = self.x_up(torch.cat([x0, x1])), self.y_up(torch.cat([y0, y1]))
        x += F.interpolate(y, scale_factor=s, mode="bilinear")
        x0, x1 = einops.rearrange(
            self.down(x),
            "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c",
            sh=sh,
            sw=sw,
        ).chunk(2)

        idxes1_to_0 = idxes1_to_0.transpose(1, 2)
        range = torch.arange(n, device=x0.device)[:, None, None]
        _idxes0_to_1 = (idxes0_to_1 + fh1 * fw1 * range).flatten(end_dim=1)
        _idxes1_to_0 = (idxes1_to_0 + fh0 * fw0 * range).flatten(end_dim=1)

        for layer in self.layers:
            x0 = layer(
                x0, x1[_idxes0_to_1].flatten(start_dim=1, end_dim=2), (h0, w0)
            )
            x1 = layer(
                x1, x0[_idxes1_to_0].flatten(start_dim=1, end_dim=2), (h1, w1)
            )

        out0 = einops.rearrange(
            x0,
            "(n fh fw) (sh sw) c -> n (fh sh fw sw) c",
            fh=fh0,
            sh=sh,
            fw=fw0,
            sw=sw,
        )
        out1 = einops.rearrange(
            x1,
            "(n fh fw) (sh sw) c -> n (fh sh fw sw) c",
            fh=fh1,
            sh=sh,
            fw=fw1,
            sw=sw,
        )
        selective0 = einops.repeat(
            x0[_idxes1_to_0],
            "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            fh=fh1,
            sh=sh,
            fw=fw1,
            sw=sw,
        )
        selective1 = einops.repeat(
            x1[_idxes0_to_1],
            "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            fh=fh0,
            sh=sh,
            fw=fw0,
            sw=sw,
        )
        idxes0_to_1 = (
            w1 * s * (idxes0_to_1 // (w1 // s)) + s * (idxes0_to_1 % (w1 // s))
        )[..., None]
        idxes1_to_0 = (
            w0 * s * (idxes1_to_0 // (w0 // s)) + s * (idxes1_to_0 % (w0 // s))
        )[..., None]
        idxes0_to_1 = idxes0_to_1 + idxes0_to_1.new_tensor([0, 1, w1, w1 + 1])
        idxes1_to_0 = idxes1_to_0 + idxes1_to_0.new_tensor([0, 1, w0, w0 + 1])
        idxes0_to_1 = einops.repeat(
            idxes0_to_1,
            "n (fh fw) k ss -> n (fh sh fw sw) (k ss)",
            fh=fh0,
            sh=sh,
            fw=fw0,
            sw=sw,
        )
        idxes1_to_0 = einops.repeat(
            idxes1_to_0,
            "n (fh fw) k ss -> n (k ss) (fh sh fw sw)",
            fh=fh1,
            sh=sh,
            fw=fw1,
            sw=sw,
        )
        return out0, out1, selective0, selective1, idxes0_to_1, idxes1_to_0
