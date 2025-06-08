from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class PyramidFuser(Module):
    def __init__(
        self,
        dims: Sequence[int],
        bias: bool = False,
        align_corners: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            dim0, dim1 = dims[i : i + 2]
            self.lateral_convs.append(nn.Conv2d(dim0, dim1, 1, bias=bias))
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(dim1, dim1, 3, padding=1, bias=bias),
                    nn.BatchNorm2d(dim1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(dim1, dim0, 3, padding=1, bias=bias),
                )
            )
        self.lateral_convs.append(nn.Conv2d(dim1, dim1, 1, bias=bias))
        self.align_corners = align_corners

    def _fuse(self, x_seq: Sequence[Tensor]) -> Tensor:
        x = self.lateral_convs[-1](x_seq[-1])
        for i in reversed(range(len(x_seq) - 1)):
            x = F.interpolate(
                x,
                size=x_seq[i].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            x = x + self.lateral_convs[i](x_seq[i])
            x = self.fusion_blocks[i](x)
        return x

    def forward(
        self, x0_seq: Sequence[Tensor], x1_seq: Sequence[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if x0_seq[0].shape == x1_seq[0].shape:
            x_seq = [torch.cat(t) for t in zip(x0_seq, x1_seq)]
            x0, x1 = self._fuse(x_seq).chunk(2)
        else:
            x0, x1 = self._fuse(x0_seq), self._fuse(x1_seq)
        return x0, x1
