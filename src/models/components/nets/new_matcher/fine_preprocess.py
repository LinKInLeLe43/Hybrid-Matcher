from typing import Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F


class FinePreprocess(nn.Module):
    def __init__(
        self,
        window_size: int,
        stride: int,
        padding: int = 0,
        right_extra: int = 0,
        scale: int = 1
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.right_extra = right_extra
        self.scale = scale

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w0, w1, s, e, p = (
            self.window_size, self.window_size + 2 * self.right_extra,
            self.stride, self.right_extra, self.padding)
        w0w0, w1w1 = w0 ** 2, w1 ** 2
        b_idxes, i_idxes, j_idxes = idxes
        if len(b_idxes) == 0:
            c = x0.shape[1]
            out0, out1 = x0.new_empty((0, w0w0, c)), x1.new_empty((0, w1w1, c))
            return out0, out1

        if self.scale > 1:
            x0 = F.interpolate(x0, scale_factor=self.scale, mode="bilinear")
            x1 = F.interpolate(x1, scale_factor=self.scale, mode="bilinear")

        out0 = F.unfold(x0, w0, padding=p, stride=s)
        out0 = einops.rearrange(
            out0, "n (c ww) l -> n l ww c", ww=w0w0)[b_idxes, i_idxes]
        out1 = F.unfold(x1, w1, padding=p + e, stride=s)
        out1 = einops.rearrange(
            out1, "n (c ww) l -> n l ww c", ww=w1w1)[b_idxes, j_idxes]
        return out0, out1
