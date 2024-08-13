from typing import List, Tuple

import torch
from torch import nn

from .repvgg import RepVgg82
from .resnet import ResNetFpn82


class SeperatedBackbone(nn.Module):
    def __init__(self, layer_depths: Tuple[int, int, int]) -> None:
        super().__init__()
        self.scales = 8,2

        self.resnet = ResNetFpn82()
        self.repvgg = RepVgg82()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        x0_8x, x0_32x = self.resnet(x)
        x1_2x, x1_4x = self.repvgg(x)

        x_8x = torch.cat([x0_8x, self.maxpool(x1_4x)], dim=1)
        return [x1_2x, x1_4x], x_8x, x0_32x
