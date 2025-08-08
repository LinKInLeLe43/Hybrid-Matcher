from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .repvgg_adapter import RepVGGAdapter


class Encoder(Module):
    def __init__(
        self, name: str, dim_list: List[int], num_blocks_list: List[int]
    ) -> None:
        super().__init__()
        if name == "repvgg":
            self.backbone = RepVGGAdapter(dim_list, num_blocks_list)
        else:
            raise ValueError("")

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, image0: Tensor, image1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if image0.shape == image1.shape:
            x_list = self.backbone(torch.cat([image0, image1]))
            x0_list, x1_list = map(list, zip(*[x.chunk(2) for x in x_list]))
        else:
            x0_list, x1_list = self.backbone(image0), self.backbone(image1)
        return x0_list, x1_list
