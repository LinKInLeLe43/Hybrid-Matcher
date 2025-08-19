from typing import List, Tuple

import torch
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

    def forward(
        self, image0: Tensor, image1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if image0.shape == image1.shape:
            x_list = self.backbone(torch.cat([image0, image1]))
            x0_list, x1_list = map(list, zip(*[x.chunk(2) for x in x_list]))
        else:
            x0_list, x1_list = self.backbone(image0), self.backbone(image1)
        return x0_list, x1_list
