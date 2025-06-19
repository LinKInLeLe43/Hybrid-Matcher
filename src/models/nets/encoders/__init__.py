from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .convnextv2 import (
    convnextv2_pico,
    convnextv2_pico_modified,
    convnextv2_tiny,
)
from .repvgg import create_RepVGG_A1, create_RepVGG_A2


class Encoder(Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "repvgg_a1":
            self.backbone = create_RepVGG_A1()
        elif name == "repvgg_a2":
            self.backbone = create_RepVGG_A2()
        elif name == "convnextv2_pico":
            self.backbone = convnextv2_pico()
        elif name == "convnextv2_pico_modified":
            self.backbone = convnextv2_pico_modified()
        elif name == "convnextv2_tiny":
            self.backbone = convnextv2_tiny()
        else:
            raise ValueError("")

        self.stem = nn.Sequential(
            nn.Conv2d(1, 1024, 9, stride=8, padding=4, bias=False),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(1024 // 16),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x0: Tensor, x1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if x0.shape == x1.shape:
            x = torch.cat([x0, x1])
            x_list = self.backbone(x)
            x0_list, x1_list = map(list, zip(*[t.chunk(2) for t in x_list]))
            x0_2x, x1_2x = self.stem(x).chunk(2)
        else:
            x0_list, x1_list = self.backbone(x0), self.backbone(x1)
            x0_2x, x1_2x = self.stem(x0), self.stem(x1)
        x0_list = [x0_2x, *x0_list]
        x1_list = [x1_2x, *x1_list]
        return x0_list, x1_list
