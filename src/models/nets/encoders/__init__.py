from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .repvgg_adapter import RepVGGAdapter


class Encoder(Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "repvgg_a1":
            self.backbone = RepVGGAdapter([64, 128, 256], [2, 4, 14])
        else:
            raise ValueError("")

    def forward(
        self, x0: Tensor, x1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if x0.shape == x1.shape:
            x = torch.cat([x0, x1])
            x_list = [t.chunk(2) for t in self.backbone(x)]
            x0_list, x1_list = map(list, zip(*x_list))
        else:
            x0_list, x1_list = self.backbone(x0), self.backbone(x1)
        return x0_list, x1_list
