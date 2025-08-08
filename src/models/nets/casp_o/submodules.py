from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class PyramidFuser(Module):
    def __init__(
        self,
        dim_list: List[int],
        bias: bool = False,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.lateral_convs.append(
            nn.Conv2d(dim_list[0], dim_list[0], 1, bias=bias)
        )
        for i in range(len(dim_list) - 1):
            dim0, dim1 = dim_list[i : i + 2]
            self.lateral_convs.append(nn.Conv2d(dim1, dim0, 1, bias=bias))
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(dim0, dim0, 3, padding=1, bias=bias),
                    nn.BatchNorm2d(dim0),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(dim0, dim1, 3, padding=1, bias=bias),
                )
            )

    def fuse(self, x_list: List[Tensor]) -> Tensor:
        x = self.lateral_convs[0](x_list[0])
        for i in range(len(x_list) - 1):
            size = tuple(x_list[i + 1].shape[-2:])
            x = F.interpolate(
                x, size=size, mode="bilinear", align_corners=self.align_corners
            )
            x = x + self.lateral_convs[i + 1](x_list[i + 1])
            x = self.fusion_blocks[i](x)
        return x

    def forward(
        self, x0_list: List[Tensor], x1_list: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if x0_list[0].shape == x1_list[0].shape:
            x_list = [torch.cat(x) for x in zip(x0_list, x1_list)]
            x0, x1 = self.fuse(x_list).chunk(2)
        else:
            x0, x1 = self.fuse(x0_list), self.fuse(x1_list)
        return x0, x1
