import copy
from typing import Tuple

import torch
from torch import nn


class MlpMixerEncoder(nn.Module):
    def __init__(self, length: int, depth: int) -> None:
        super().__init__()

        self.token_mlp = nn.Sequential(
            nn.Linear(length, length),
            nn.GELU(),
            nn.Linear(length, length))
        self.channel_mlp = nn.Sequential(
            nn.Linear(depth, depth),
            nn.GELU(),
            nn.Linear(depth, depth))
        self.norm0 = nn.LayerNorm(depth)
        self.norm1 = nn.LayerNorm(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm0(self.token_mlp(x.transpose(1, 2)).transpose(1, 2))
        x = x + self.norm1(self.channel_mlp(x))
        return x


class MlpMixer(nn.Module):
    def __init__(self, length: int, depth: int, layer_count: int) -> None:
        super().__init__()

        encoder = MlpMixerEncoder(2 * length, depth)
        self.layers = nn.ModuleList([copy.deepcopy(encoder)
                                     for _ in range(layer_count)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x0, x1], dim=1)
        for layer in self.layers:
            x = layer(x)

        x0, x1 = x.chunk(2, dim=1)
        return x0, x1
