# TODO:
# - Detect NaN
# - Change weight init
# - Change variable name
# - SelectiveTransformerLayer accepts cropped tensors


import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils.grid import create_meshgrid
from torch import Tensor

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
            warnings.warn("SDPA is not available.", stacklevel=2)
        self.enable_sdpa = enable_sdpa and SDPA_AVAILABLE
        if enable_flash and not self.enable_sdpa:
            warnings.warn("SDPA is not enabled.", stacklevel=2)
        self.enable_flash = enable_flash and self.enable_sdpa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = None
        if q_mask is not None and kv_mask is not None:
            mask = q_mask[..., :, None] & kv_mask[..., None, :]

        if self.enable_sdpa:
            if self.enable_flash:
                if mask is not None:
                    raise ValueError("")

                # sdpa forces flash kernel
                args = [x.half().contiguous() for x in [q, k, v]]
                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    message = sdpa(*args).to(q.dtype)
            else:
                # sdpa automatically selects kernel
                args = [x.contiguous() for x in [q, k, v]]
                message = sdpa(*args, attn_mask=mask)
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
        x: Tensor,
        y: Tensor,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        fc = self.num_heads

        if x_mask is not None and y_mask is not None:
            x_mask, y_mask = x_mask[:, None], y_mask[:, None]

        q, k, v = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        q, k, v = [
            rearrange(x, "n l (fc sc) -> n fc l sc", fc=fc) for x in [q, k, v]
        ]
        message = self.attention(q, k, v, q_mask=x_mask, kv_mask=y_mask)
        message = rearrange(message, "n fc l sc -> n l (fc sc)")
        message = self.merge(message)
        message = torch.cat([x, self.norm1(message)], dim=2)

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

        self.down_q = nn.Conv2d(
            dim, dim, scale, stride=scale, groups=dim, bias=False
        )
        self.down_kv = nn.MaxPool2d(scale, stride=scale)
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
        x: Tensor,
        y: Tensor,
        rope: Optional[nn.Module] = None,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        fc = self.num_heads
        sh = x.shape[2] // self.scale

        if x_mask is not None and y_mask is not None:
            x_mask, y_mask = x_mask[:, None], y_mask[:, None]

        q = self.down_q(x).permute(0, 2, 3, 1)
        kv = self.down_kv(y).permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
        if rope is not None:
            q, k = rope.rel_pe(q), rope.rel_pe(k)
        q, k, v = [
            rearrange(x, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
            for x in [q, k, v]
        ]
        message = self.attention(q, k, v, q_mask=x_mask, kv_mask=y_mask)
        message = rearrange(message, "n fc l sc -> n l (fc sc)")
        message = self.merge(message)

        message = self.norm1(message)
        message = rearrange(message, "n (sh sw) c -> n c sh sw", sh=sh)
        message = F.interpolate(
            message, scale_factor=self.scale, mode="bilinear"
        )
        message = torch.cat([x, message], dim=1).permute(0, 2, 3, 1)

        x = x + self.norm2(self.mlp(message)).permute(0, 3, 1, 2)
        return x


class SelectiveTransformerLayer(nn.Module):
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

    def forward(self, x: Tensor, y: Tensor, size: Tuple[int, int]) -> Tensor:
        fc = self.num_heads
        axes_lengths = {
            "fh": size[0],
            "fw": size[1],
            "sh": self.scale,
            "sw": self.scale,
        }

        q, k, v = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        q, k, v = [  # l = ss/kss for q/kv
            rearrange(x, "n ff l (fc sc) -> n ff fc l sc", fc=fc)
            for x in [q, k, v]
        ]
        message = self.attention(q, k, v)
        message = rearrange(message, "n ff fc ss sc -> n ff ss (fc sc)")
        message = self.merge(message)
        message = torch.cat([x, self.norm1(message)], dim=3)

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
        type: str,
        dim: int,
        num_heads: int,
        num_layers: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        if type == "vanilla":
            layer = VanillaTransformerLayer(dim, num_heads)
        elif type == "aggregated":
            if "scale" not in kwargs:
                raise ValueError()
            layer = AggregatedTransformerLayer(kwargs["scale"], dim, num_heads)
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
        x0: Tensor,
        x1: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x0 = layer(x0, x0, x_mask=mask0, y_mask=mask0)
                x1 = layer(x1, x1, x_mask=mask1, y_mask=mask1)
            else:
                x0 = layer(x0, x1, x_mask=mask0, y_mask=mask1)
                x1 = layer(x1, x0, x_mask=mask1, y_mask=mask0)
        return x0, x1


class FusedSelectiveTransformer(nn.Module):
    def __init__(
        self,
        scale: int,
        dims: Tuple[int, int],
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.scale = scale
        indices = create_meshgrid(
            self.scale,
            self.scale,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=2)
        self.register_buffer("indices", indices, persistent=False)

        self.x_up = nn.Conv2d(dims[0], dims[1], 1, bias=False)
        self.y_up = nn.Conv2d(dims[1], dims[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dims[1], dims[0], 3, padding=1, bias=False),
        )

        layer = SelectiveTransformerLayer(scale, dims[0], num_heads)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fpn_fuse(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self.x_up(x), self.y_up(y)
        x = x + F.interpolate(y, scale_factor=self.scale, mode="bilinear")
        x = self.down(x)
        return x

    def _map_prior(self, x: Tensor, fw: int) -> Tensor:
        row = (x[..., None] // fw) * self.scale + self.indices[:, 1]
        col = (x[..., None] % fw) * self.scale + self.indices[:, 0]
        x = (row * fw * self.scale + col).flatten(start_dim=-2)
        return x

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Tensor,
        y1: Tensor,
        prior0_to_1: Tensor,
        prior1_to_0: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        (n, _, fh0, fw0), (_, _, fh1, fw1) = y0.shape, y1.shape
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
        device = x0.device

        if x0.shape == x1.shape:
            x, y = torch.cat([x0, x1]), torch.cat([y0, y1])
            x0, x1 = self._fpn_fuse(x, y).chunk(2)
        else:
            x0, x1 = self._fpn_fuse(x0, y0), self._fpn_fuse(x1, y1)

        x0 = rearrange(
            x0, "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c", **axes_lengths0
        )
        x1 = rearrange(
            x1, "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c", **axes_lengths1
        )
        range = torch.arange(n, device=device)[:, None, None]
        for layer in self.layers:
            x0_to_1 = x1[range, prior0_to_1].flatten(start_dim=2, end_dim=3)
            x0 = layer(x0, x0_to_1, (fh0, fw0))
            x1_to_0 = x0[range, prior1_to_0].flatten(start_dim=2, end_dim=3)
            x1 = layer(x1, x1_to_0, (fh1, fw1))
        x0_to_1 = x1[range, prior0_to_1].flatten(start_dim=2, end_dim=3)
        x1_to_0 = x0[range, prior1_to_0].flatten(start_dim=2, end_dim=3)

        prior0_to_1 = self._map_prior(prior0_to_1, fw1)
        prior1_to_0 = self._map_prior(prior1_to_0, fw0)
        prior0_to_1 = repeat(
            prior0_to_1,
            "n (fh fw) kss -> n (fh sh fw sw) kss",
            **axes_lengths0,
        )
        prior1_to_0 = repeat(
            prior1_to_0,
            "n (fh fw) kss -> n (fh sh fw sw) kss",
            **axes_lengths1,
        )
        return x0, x1, x0_to_1, x1_to_0, prior0_to_1, prior1_to_0
