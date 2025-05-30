from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .repvgg import create_RepVGG_A1, create_RepVGG_A2


class Encoder(Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "repvgg_a1":
            self.backbone = create_RepVGG_A1()
        elif name == "repvgg_a2":
            self.backbone = create_RepVGG_A2()
        else:
            raise ValueError("")

    def forward(
        self, image0: Tensor, image1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if image0.shape == image1.shape:
            xs = self.backbone(torch.cat([image0, image1]))
            x0s, x1s = zip(*(x.chunk(2, dim=0) for x in xs))
            x0s, x1s = list(x0s), list(x1s)
        else:
            x0s, x1s = self.backbone(image0), self.backbone(image1)
        return x0s, x1s
