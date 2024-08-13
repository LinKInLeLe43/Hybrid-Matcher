import copy
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.attention = attention
        self.nchw = False

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False))
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.q_proj(x).unflatten(2, (self.heads_count, -1))
        k = self.k_proj(source).unflatten(2, (self.heads_count, -1))
        v = self.v_proj(source).unflatten(2, (self.heads_count, -1))
        out = self.attention(
            q, k, v, q_mask=x_mask, kv_mask=source_mask).flatten(start_dim=2)

        out = self.merge(out)
        out = self.norm1(out)

        out = torch.cat([x, out], dim=2)
        out = self.mlp(out)
        out = self.norm2(out)

        out += x
        return out


class AggregatedEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        scale: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.scale = scale
        self.attention = attention
        self.nchw = True

        self.down_q = nn.Conv2d(
            depth, depth, scale, stride=scale, groups=depth, bias=False)
        self.down_kv = nn.MaxPool2d(scale, stride=scale)

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False))
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        s = self.scale

        q = self.down_q(x).flatten(start_dim=2).transpose(1, 2)
        kv = self.down_kv(source).flatten(start_dim=2).transpose(1, 2)

        q = self.q_proj(q).unflatten(2, (self.heads_count, -1))
        k = self.k_proj(kv).unflatten(2, (self.heads_count, -1))
        v = self.v_proj(kv).unflatten(2, (self.heads_count, -1))
        out = self.attention(
            q, k, v, q_mask=x_mask, kv_mask=source_mask).flatten(start_dim=2)

        out = self.merge(out)
        out = self.norm1(out)
        out = out.transpose(1, 2).unflatten(2, (x.shape[2] // s, x.shape[3] // s))
        out = F.interpolate(out, scale_factor=s, mode="bilinear")

        out = torch.cat([x, out], dim=1)
        out = out.permute(0, 2, 3, 1)
        out = self.mlp(out)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        out += x
        return out


class LoFTR(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        types: List[str]
    ) -> None:
        super().__init__()
        self.types = types
        self.nchw = encoder.nchw

        self.layers = nn.ModuleList([copy.deepcopy(encoder) for _ in types])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        size0: Optional[Tuple[int, int]] = None,
        size1: Optional[Tuple[int, int]] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.nchw:
            if size0 is None or size1 is None:
                raise ValueError("")

            feature0 = feature0.transpose(1, 2).unflatten(2, size0).contiguous()
            feature1 = feature1.transpose(1, 2).unflatten(2, size1).contiguous()

        for layer, type in zip(self.layers, self.types):
            if type == "self":
                feature0 = layer(
                    feature0, feature0, x_mask=mask0, source_mask=mask0)
                feature1 = layer(
                    feature1, feature1, x_mask=mask1, source_mask=mask1)
            elif type == "cross":
                feature0 = layer(
                    feature0, feature1, x_mask=mask0, source_mask=mask1)
                feature1 = layer(
                    feature1, feature0, x_mask=mask1, source_mask=mask0)
            else:
                raise ValueError("")

        if self.nchw:
            feature0 = feature0.flatten(start_dim=2).transpose(1, 2)
            feature1 = feature1.flatten(start_dim=2).transpose(1, 2)
        return feature0, feature1
