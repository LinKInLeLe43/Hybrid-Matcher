import copy
from typing import List, Optional, Tuple

import torch
from torch import nn


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


class LoFTR(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        attention: nn.Module,
        types: List[str]
    ) -> None:
        super().__init__()
        self.types = types

        encoder = TransformerEncoder(depth, heads_count, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder) for _ in types])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return feature0, feature1
