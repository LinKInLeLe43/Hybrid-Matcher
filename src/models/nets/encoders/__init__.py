from typing import List, Tuple

import torch
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

    def forward(
        self, x0: Tensor, x1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if x0.shape == x1.shape:
            x_list = self.backbone(torch.cat([x0, x1]))
            x0_list, x1_list = map(list, zip(*[x.chunk(2) for x in x_list]))
        else:
            x0_list, x1_list = self.backbone(x0), self.backbone(x1)
        return x0_list, x1_list
