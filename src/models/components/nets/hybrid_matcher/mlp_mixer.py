import copy
from typing import List, Optional, Tuple

import torch
from torch import nn


class MLPMixerEncoder(nn.Module):
    def __init__(self, token_depth: int, channel_depth: int) -> None:
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.Linear(token_depth, token_depth),
            nn.GELU(),
            nn.Linear(token_depth, token_depth))
        self.channel_mixer = nn.Sequential(
            nn.Linear(channel_depth, channel_depth),
            nn.GELU(),
            nn.Linear(channel_depth, channel_depth))
        self.norm0 = nn.LayerNorm(channel_depth)
        self.norm1 = nn.LayerNorm(channel_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        message = x.transpose(1, 2)
        message = self.token_mixer(message)
        message = message.transpose(1, 2)
        message = self.norm0(message)
        x = x + message

        message = self.channel_mixer(x)
        message = self.norm1(message)
        out = x + message
        return out


class MLPMixer(nn.Module):
    def __init__(
        self,
        left_token_depth: int,
        right_token_depth: int,
        channel_depth: int,
        types: List[str]
    ) -> None:
        super().__init__()
        self.left_token_depth = left_token_depth
        self.right_token_depth = right_token_depth
        self.types = types

        encoder = MLPMixerEncoder(
            left_token_depth + right_token_depth, channel_depth)
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
        feature = torch.cat([feature0, feature1], dim=1)
        for layer, type in zip(self.layers, self.types):
            if type == "mixer":
                feature = layer(feature)
            else:
                raise ValueError("")
        feature0, feature1 = feature.split(
            [self.left_token_depth, self.right_token_depth], dim=1)
        return feature0, feature1
