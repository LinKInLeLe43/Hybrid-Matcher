# TODO:
# - Detect NaN
# - Change weight init
# - Change variable name

from copy import deepcopy
from typing import Dict, Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils import create_meshgrid

try:
    # deprecated after torch 2.3.0, see https://github.com/pytorch/pytorch/releases/tag/v2.3.0
    from torch.backends.cuda import sdp_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False


class Attention(nn.Module):
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.enable_sdpa:
            if self.enable_flash:
                if mask is not None:
                    raise ValueError("")

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
            similarity = torch.einsum("...ld,...sd->...ls", q, k)
            if mask is not None:
                similarity.masked_fill_(~mask, -float("inf"))

            attention = similarity.softmax(dim=-1)
            message = torch.einsum("...ls,...sd->...ld", attention, v)

        if mask is not None:
            message.nan_to_num_()
        return message


class VanillaTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.merge = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim, dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        q, k, v = [
            rearrange(x, "n l (fc sc) -> n fc l sc", fc=self.num_heads)
            for x in [q, k, v]
        ]
        message = self.attention(
            q, k, v, mask=mask[:, None] if mask is not None else None
        )
        message = rearrange(message, "n fc l sc -> n l (fc sc)")
        message = self.norm1(self.merge(message))
        message = torch.cat([x, message], dim=-1)
        x = x + self.norm2(self.mlp(message))
        return x


class AggregatedTransformerLayer(nn.Module):
    def __init__(
        self,
        scale: int,
        dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads

        if scale > 1:
            self.down_q = nn.Conv2d(
                dim, dim, scale, stride=scale, groups=dim, bias=False
            )
            self.down_kv = nn.MaxPool2d(scale, stride=scale)
        else:
            self.down_q = self.down_kv = nn.Identity()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.merge = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim, dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rope: Optional[nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.down_q(x).permute(0, 2, 3, 1)
        kv = self.down_kv(y).permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
        if rope is not None:
            q, k = rope.rel_pe(q), rope.rel_pe(k)
        q, k, v = [
            rearrange(x, "n h w (fc sc) -> n fc (h w) sc", fc=self.num_heads)
            for x in [q, k, v]
        ]
        message = self.attention(
            q, k, v, mask=mask[:, None] if mask is not None else None
        )
        message = rearrange(message, "n fc l sc -> n l (fc sc)")
        message = self.norm1(self.merge(message))
        message = rearrange(
            message, "n (sh sw) c -> n c sh sw", sh=x.shape[2] // self.scale
        )
        message = F.interpolate(
            message,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )
        message = torch.cat([x, message], dim=1).permute(0, 2, 3, 1)
        x = x + self.norm2(self.mlp(message)).permute(0, 3, 1, 2)
        return x


class RegionBasedSelectiveTransformerLayer(nn.Module):
    def __init__(
        self,
        scale: int,
        dim: int,
        num_heads: int,
        enable_sdpa: bool = False,
        enable_flash: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attention = Attention(
            enable_sdpa=enable_sdpa, enable_flash=enable_flash
        )
        self.merge = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, axes_lengths: Dict[str, int]
    ) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        q, k, v = [  # l = ss/(k * ss) for q/kv
            rearrange(x, "n ff l (fc sc) -> n ff fc l sc", fc=self.num_heads)
            for x in [q, k, v]
        ]
        message = self.attention(q, k, v)
        message = rearrange(message, "n ff fc ss sc -> n ff ss (fc sc)")
        message = self.norm1(self.merge(message))
        message = torch.cat([x, message], dim=-1)
        message = rearrange(
            message,
            "n (fh fw) (sh sw) c -> n c (fh sh) (fw sw)",
            **axes_lengths,
        )
        message = self.mlp(message)
        message = rearrange(
            message,
            "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c",
            **axes_lengths,
        )
        message = self.norm2(message)
        x = x + message
        return x


class LocalFeatureTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        name: str = "vanilla",
        scale: Optional[int] = None,
    ) -> None:
        super().__init__()

        if name == "vanilla":
            layer = VanillaTransformerLayer(dim, num_heads)
        elif name == "aggregated":
            if scale is None:
                raise ValueError()
            layer = AggregatedTransformerLayer(scale, dim, num_heads)
        else:
            raise ValueError()
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(2 * num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask00 = mask11 = mask01 = mask10 = None
        if mask0 is not None and mask1 is not None:
            mask00 = mask0[..., :, None] & mask0[..., None, :]
            mask11 = mask1[..., :, None] & mask1[..., None, :]
            mask01 = mask0[..., :, None] & mask1[..., None, :]
            mask10 = mask01.transpose(-1, -2)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x0 = layer(x0, x0, mask=mask00)
                x1 = layer(x1, x1, mask=mask11)
            else:
                x0 = layer(x0, x1, mask=mask01)
                x1 = layer(x1, x0, mask=mask10)
        return x0, x1


class RegionBasedSelectiveTransformer(nn.Module):
    def __init__(
        self,
        scale: int,
        dims: Tuple[int, int],
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.x_up = nn.Conv2d(dims[0], dims[1], 1, bias=False)
        self.y_up = nn.Conv2d(dims[1], dims[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dims[1], dims[0], 3, padding=1, bias=False),
        )

        layer = RegionBasedSelectiveTransformerLayer(scale, dims[0], num_heads)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

        delta_indices = create_meshgrid(
            self.scale,
            self.scale,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("delta_indices", delta_indices, persistent=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def fpn_fuse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.x_up(x), self.y_up(y)
        x = x + F.interpolate(
            y, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        x = self.down(x)
        return x

    def reshape_to_region_based(
        self, x: torch.Tensor, axes_lengths: Dict[str, int]
    ) -> torch.Tensor:
        out = rearrange(
            x, "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c", **axes_lengths
        )
        return out

    def gather_attended(
        self, x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        out = x[
            torch.arange(x.shape[0], device=x.device)[:, None, None], indices
        ].flatten(start_dim=2, end_dim=3)
        return out

    def map_indices(
        self, x: torch.Tensor, axes_lengths: Dict[str, int], fw: int
    ) -> torch.Tensor:
        row = (x[..., None] // fw) * self.scale + self.delta_indices[:, 1]
        col = (x[..., None] % fw) * self.scale + self.delta_indices[:, 0]
        out = row * fw * self.scale + col
        out = repeat(
            out, "n (fh fw) k ss -> n (fh sh fw sw) (k ss)", **axes_lengths
        )
        return out

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: torch.Tensor,
        y1: torch.Tensor,
        indices0_to_1: torch.Tensor,
        indices1_to_0: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, _, fh0, fw0 = y0.shape
        _, _, fh1, fw1 = y1.shape
        axes_lengths0 = {
            "fh": fh0,
            "fw": fw0,
            "sh": self.scale,
            "sw": self.scale,
        }
        axes_lengths1 = {
            "fh": fh1,
            "fw": fw1,
            "sh": self.scale,
            "sw": self.scale,
        }

        if (fh0, fw0) == (fh1, fw1):
            x, y = torch.cat([x0, x1]), torch.cat([y0, y1])
            x = self.fpn_fuse(x, y)
            x0, x1 = self.reshape_to_region_based(x, axes_lengths0).chunk(2)
        else:
            x0, x1 = self.fpn_fuse(x0, y0), self.fpn_fuse(x1, y1)
            x0 = self.reshape_to_region_based(x0, axes_lengths0)
            x1 = self.reshape_to_region_based(x1, axes_lengths1)

        for layer in self.layers:
            attended0 = self.gather_attended(x1, indices0_to_1)
            x0 = layer(x0, attended0, axes_lengths0)
            attended1 = self.gather_attended(x0, indices1_to_0)
            x1 = layer(x1, attended1, axes_lengths1)
        attended0 = self.gather_attended(x1, indices0_to_1)
        attended1 = self.gather_attended(x0, indices1_to_0)

        indices0_to_1 = self.map_indices(indices0_to_1, axes_lengths0, fw1)
        indices1_to_0 = self.map_indices(indices1_to_0, axes_lengths1, fw0)
        return x0, x1, attended0, attended1, indices0_to_1, indices1_to_0
