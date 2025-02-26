import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .attention import Attention


class VanillaTransformerLayer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        allow_sdp: bool = False,
        force_flash: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)

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
        out = feat0 + self.norm2(self.mlp(torch.cat([feat0, message], dim=-1)))
        return out


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

        self.head_dim = feat_dim // num_heads
        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

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

        n, _, h, w = feat0.shape
        q = self.down_q(feat0).permute(0, 2, 3, 1)
        kv = self.down_kv(feat1).permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)

        if rope is not None:
            q, k = rope(q, "rel"), rope(k, "rel")

        q, k, v = (
            x.reshape(n, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            for x in (q, k, v)
        )
        message = self.attention(q, k, v, q_mask=mask0, kv_mask=mask1)
        message = message.transpose(1, 2).flatten(start_dim=-2)
        message = self.norm1(self.merge(message))
        message = message.transpose(1, 2).unflatten(
            -1, (h // self.scale, w // self.scale)
        )
        message = F.interpolate(
            message, scale_factor=self.scale, mode="bilinear"
        )
        message = torch.cat([feat0, message], dim=1).permute(0, 2, 3, 1)
        message = self.norm2(self.mlp(message)).permute(0, 3, 1, 2)
        out = feat0 + message.contiguous()
        return out


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

        self.attention = Attention(
            allow_sdp=allow_sdp, force_flash=force_flash
        )

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)

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

        kwargs = {
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
            message, "(n fh fw) (sh sw) c -> n c (fh sh) (fw sw)", **kwargs
        )
        message = self.mlp(message)
        message = rearrange(
            message, "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c", **kwargs
        )
        out = feat0 + self.norm2(message)
        return out


class LocalFeatureTransformer(nn.Module):
    def __init__(self, layer: nn.Module, types: List[str]) -> None:
        super().__init__()
        self.types = types

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in types])

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
        layer: nn.Module,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.x_up = nn.Conv2d(feat_dims[0], feat_dims[1], 1, bias=False)
        self.y_up = nn.Conv2d(feat_dims[1], feat_dims[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(feat_dims[1], feat_dims[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dims[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dims[1], feat_dims[0], 3, padding=1, bias=False),
        )

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fuse_feats(
        self, btm_feat: torch.Tensor, top_feat: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        btm_feat, top_feat = self.x_up(btm_feat), self.y_up(top_feat)
        out = btm_feat + F.interpolate(
            top_feat, scale_factor=self.scale, mode="bilinear"
        )
        out = self.down(out)
        out = rearrange(
            out, "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c", **kwargs
        )
        return out

    def _scale_indices(
        self, indices: torch.Tensor, tgt_fw: int, **kwargs
    ) -> torch.Tensor:
        tgt_w = self.scale * tgt_fw
        rows, cols = indices // tgt_fw, indices % tgt_fw
        out = tgt_w * self.scale * rows + self.scale * cols
        out = indices.new_tensor([0, 1, tgt_w, tgt_w + 1]) + out[..., None]
        out = repeat(out, "n (fh fw) k ss -> n (fh sh fw sw) (k ss)", **kwargs)
        return out

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
        kwargs0 = {"fh": fh0, "fw": fw0, "sh": self.scale, "sw": self.scale}
        kwargs1 = {"fh": fh1, "fw": fw1, "sh": self.scale, "sw": self.scale}

        if btm_feat0.shape == btm_feat1.shape:
            feat0, feat1 = self._fuse_feats(
                torch.cat([btm_feat0, btm_feat1]),
                torch.cat([top_feat0, top_feat1]),
                **kwargs0,
            ).chunk(2)
        else:
            feat0 = self._fuse_feats(btm_feat0, top_feat0, **kwargs0)
            feat1 = self._fuse_feats(btm_feat1, top_feat1, **kwargs1)

        indices1_to_0 = indices1_to_0.transpose(1, 2)
        range = torch.arange(
            btm_feat0.shape[0],
            device=btm_feat0.device,
        )[:, None, None]
        _idxes0_to_1 = (indices0_to_1 + fh1 * fw1 * range).flatten(end_dim=1)
        _idxes1_to_0 = (indices1_to_0 + fh0 * fw0 * range).flatten(end_dim=1)

        for layer in self.layers:
            feat0 = layer(
                feat0,
                feat1[_idxes0_to_1].flatten(start_dim=1, end_dim=2),
                (fh0, fw0),
            )
            feat1 = layer(
                feat1,
                feat0[_idxes1_to_0].flatten(start_dim=1, end_dim=2),
                (fh1, fw1),
            )

        out0 = rearrange(
            feat0, "(n fh fw) (sh sw) c -> n (fh sh fw sw) c", **kwargs0
        )
        out1 = rearrange(
            feat1, "(n fh fw) (sh sw) c -> n (fh sh fw sw) c", **kwargs1
        )
        selective0 = repeat(
            feat0[_idxes1_to_0],
            "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            **kwargs1,
        )
        selective1 = repeat(
            feat1[_idxes0_to_1],
            "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            **kwargs0,
        )
        indices0_to_1 = self._scale_indices(indices0_to_1, fw1, **kwargs0)
        indices1_to_0 = self._scale_indices(indices1_to_0, fw0, **kwargs1)
        return out0, out1, selective0, selective1, indices0_to_1, indices1_to_0
