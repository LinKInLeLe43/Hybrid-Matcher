# TODO:
# - Remove .contiguous() after test training and inferencing
# - Change weight init
# - Change variable name

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self, allow_sdp: bool = False, force_flash: bool = False
    ) -> None:
        super().__init__()
        self.enable_sdp = allow_sdp and FLASH_AVAILABLE
        self.force_flash = force_flash

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = None
        if q_mask is not None and kv_mask is not None:
            mask = q_mask[..., :, None] & kv_mask[..., None, :]

        if self.enable_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            if self.force_flash:
                # Flash kernel does not support mask and FP32 precision
                if mask is not None:
                    raise ValueError()

                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    message = F.scaled_dot_product_attention(*args)
            else:
                # Automatically selects kernel
                message = F.scaled_dot_product_attention(*args, attn_mask=mask)
        else:
            scale = q.shape[-1] ** -0.5
            sim = torch.einsum("...ld,...sd->...ls", q, k) * scale
            if mask is not None:
                sim.masked_fill_(~mask, -float("inf"))

            attn = F.softmax(sim, dim=-1)
            message = torch.einsum("...ls,...sc->...lc", attn, v)
            if mask is not None:
                message.nan_to_num_()
        return message


class TransformerLayer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        allow_sdp: bool = False,
        force_flash: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

        self.merge = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * feat_dim, feat_dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask0 is not None and mask1 is not None:
            mask0, mask1 = mask0[:, None], mask1[:, None]

        q, k, v = self.q_proj(feat0), self.k_proj(feat1), self.v_proj(feat1)
        q, k, v = (
            x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
            for x in (q, k, v)
        )
        message = self.attention(q, k, v, q_mask=mask0, kv_mask=mask1)

        message = message.transpose(1, 2).flatten(start_dim=-2)
        message = self.norm1(self.merge(message))
        message = torch.cat([feat0, message], dim=-1)
        feat0 = feat0 + self.norm2(self.mlp(message))
        return feat0


class AggregatedTransformerLayer(nn.Module):
    def __init__(
        self,
        scale: int,
        feat_dim: int,
        num_heads: int,
        allow_sdp: bool = False,
        force_flash: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads

        self.down_q = nn.Conv2d(
            feat_dim,
            feat_dim,
            scale,
            stride=scale,
            groups=feat_dim,
            bias=False,
        )
        self.down_kv = nn.MaxPool2d(scale, stride=scale)
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

        self.merge = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, 2 * feat_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * feat_dim, feat_dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        rope: Optional[nn.Module] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask0 is not None and mask1 is not None:
            mask0, mask1 = mask0[:, None], mask1[:, None]

        q, kv = self.down_q(feat0), self.down_kv(feat1)
        q, kv = q.permute(0, 2, 3, 1), kv.permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
        if rope is not None:
            q, k = rope(q, "rel"), rope(k, "rel")
        q, k, v = (
            rearrange(
                x, "... h w (fc sc) -> ... fc (h w) sc", fc=self.num_heads
            )
            for x in (q, k, v)
        )
        message = self.attention(q, k, v, q_mask=mask0, kv_mask=mask1)

        message = message.transpose(1, 2).flatten(start_dim=-2)
        message = self.norm1(self.merge(message))
        message = message.transpose(1, 2).unflatten(
            -1, (feat0.shape[2] // self.scale, feat0.shape[3] // self.scale)
        )
        message = F.interpolate(
            message, scale_factor=self.scale, mode="bilinear"
        )
        message = torch.cat([feat0, message], dim=1).permute(0, 2, 3, 1)
        message = self.norm2(self.mlp(message)).permute(0, 3, 1, 2)
        feat0 = feat0 + message.contiguous()
        return feat0


class SelectiveTransformerLayer(nn.Module):
    def __init__(
        self,
        scale: int,
        feat_dim: int,
        num_heads: int,
        allow_sdp: bool = False,
        force_flash: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

        self.merge = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * feat_dim, 2 * feat_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * feat_dim, feat_dim, 3, padding=1, bias=False),
        )
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        size0: Tuple[int, int],
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask0 is not None and mask1 is not None:
            mask0, mask1 = mask0[:, None], mask1[:, None]

        axes_lengths = {
            "fh": size0[0],
            "fw": size0[1],
            "sh": self.scale,
            "sw": self.scale,
        }

        q, k, v = self.q_proj(feat0), self.k_proj(feat1), self.v_proj(feat1)
        q, k, v = (
            x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
            for x in (q, k, v)
        )
        message = self.attention(q, k, v, q_mask=mask0, kv_mask=mask1)

        message = message.transpose(1, 2).flatten(start_dim=-2)
        message = self.norm1(self.merge(message))
        message = torch.cat([feat0, message], dim=-1)
        message = rearrange(
            message,
            "... (fh fw) (sh sw) c -> ... c (fh sh) (fw sw)",
            **axes_lengths,
        )
        message = self.mlp(message)
        message = rearrange(
            message,
            "... c (fh sh) (fw sw) -> ... (fh fw) (sh sw) c",
            **axes_lengths,
        )
        feat0 = feat0 + self.norm2(message)
        return feat0


class LocalFeatureTransformer(nn.Module):
    def __init__(self, layer: nn.Module, types: List[str]) -> None:
        super().__init__()
        self.types = types

        self.layers = nn.ModuleList([deepcopy(layer) for _ in types])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, type in zip(self.layers, self.types):
            if type == "self":
                feat0 = layer(feat0, feat0, mask0=mask0, mask1=mask0)
                feat1 = layer(feat1, feat1, mask0=mask1, mask1=mask1)
            elif type == "cross":
                feat0 = layer(feat0, feat1, mask0=mask0, mask1=mask1)
                feat1 = layer(feat1, feat0, mask0=mask1, mask1=mask0)
            else:
                raise ValueError()
        return feat0, feat1


class FusedSelectiveTransformer(nn.Module):
    def __init__(
        self,
        scale: int,
        feat_dims: Tuple[int, int],
        num_heads: int,
        num_layers: int,
        allow_sdp: bool = False,
        force_flash: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale

        btm_dim, top_dim = feat_dims
        self.x_up = nn.Conv2d(btm_dim, top_dim, 1, bias=False)
        self.y_up = nn.Conv2d(top_dim, top_dim, 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(top_dim, top_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(top_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(top_dim, btm_dim, 3, padding=1, bias=False),
        )

        layer = SelectiveTransformerLayer(
            scale,
            btm_dim,
            num_heads,
            allow_sdp=allow_sdp,
            force_flash=force_flash,
        )
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fuse_feats(
        self, btm_feat: torch.Tensor, top_feat: torch.Tensor, **axes_lengths
    ) -> torch.Tensor:
        btm_feat, top_feat = self.x_up(btm_feat), self.y_up(top_feat)
        btm_feat = btm_feat + F.interpolate(
            top_feat, scale_factor=self.scale, mode="bilinear"
        )
        btm_feat = self.down(btm_feat)
        btm_feat = rearrange(
            btm_feat,
            "... c (fh sh) (fw sw) -> ... (fh fw) (sh sw) c",
            **axes_lengths,
        )
        return btm_feat

    def _upsample_indices(
        self, indices: torch.Tensor, window_size: int, **axes_lengths
    ) -> torch.Tensor:
        rows, cols = indices // window_size, indices % window_size
        w = self.scale * window_size
        indices = (self.scale * w * rows + self.scale * cols)[..., None]
        indices = indices + indices.new_tensor([0, 1, w, w + 1])
        indices = repeat(
            indices, "n (fh fw) k ss -> n (fh sh fw sw) (k ss)", **axes_lengths
        )
        return indices

    def forward(
        self,
        btm_feat0: torch.Tensor,
        btm_feat1: torch.Tensor,
        top_feat0: torch.Tensor,
        top_feat1: torch.Tensor,
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
        fh0 = btm_feat0.shape[2] // self.scale
        fw0 = btm_feat0.shape[3] // self.scale
        fh1 = btm_feat1.shape[2] // self.scale
        fw1 = btm_feat1.shape[3] // self.scale
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

        if btm_feat0.shape == btm_feat1.shape:
            btm_feat = torch.cat([btm_feat0, btm_feat1])
            top_feat = torch.cat([top_feat0, top_feat1])
            feat = self._fuse_feats(btm_feat, top_feat, **axes_lengths0)
            feat0, feat1 = feat.chunk(2)
        else:
            feat0 = self._fuse_feats(btm_feat0, top_feat0, **axes_lengths0)
            feat1 = self._fuse_feats(btm_feat1, top_feat1, **axes_lengths1)

        indices1_to_0 = indices1_to_0.transpose(1, 2)
        range = torch.arange(btm_feat0.shape[0], device=btm_feat0.device)
        range = range[:, None, None]
        for layer in self.layers:
            feat0_to_1 = feat1[range, indices0_to_1]
            feat0_to_1 = feat0_to_1.flatten(start_dim=2, end_dim=3)
            feat0 = layer(feat0, feat0_to_1, (fh0, fw0))
            feat1_to_0 = feat0[range, indices1_to_0]
            feat1_to_0 = feat1_to_0.flatten(start_dim=2, end_dim=3)
            feat1 = layer(feat1, feat1_to_0, (fh1, fw1))
        feat0_to_1 = feat1[range, indices0_to_1]
        feat0_to_1 = feat0_to_1.flatten(start_dim=2, end_dim=3)

        feat0 = rearrange(
            feat0,
            "... (fh fw) (sh sw) c -> ... (fh sh fw sw) c",
            **axes_lengths0,
        )
        feat1 = rearrange(
            feat1,
            "... (fh fw) (sh sw) c -> ... (fh sh fw sw) c",
            **axes_lengths1,
        )
        feat0_to_1 = repeat(
            feat0_to_1,
            "... (fh fw) k ss c -> ... (fh sh fw sw) (k ss) c",
            **axes_lengths0,
        )
        feat1_to_0 = repeat(
            feat1_to_0,
            "... (fh fw) k ss c -> ... (fh sh fw sw) (k ss) c",
            **axes_lengths1,
        )
        indices0_to_1 = self._upsample_indices(
            indices0_to_1, fw1, **axes_lengths0
        )
        indices1_to_0 = self._upsample_indices(
            indices1_to_0, fw0, **axes_lengths1
        )
        indices1_to_0 = indices1_to_0.transpose(1, 2)
        return (
            feat0,
            feat1,
            feat0_to_1,
            feat1_to_0,
            indices0_to_1,
            indices1_to_0,
        )
