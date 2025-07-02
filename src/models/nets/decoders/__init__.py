from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from ..submodules import PyramidFuser
from .transformer import RegionSelectiveTransformerLayer, TransformerLayer


class Decoder(Module):
    def __init__(
        self,
        stride: int,
        topk: int,
        dims: Sequence[int],
        num_x_layers: int,
        num_y_layers: int,
        enable_crop: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(dims) == 2
        self.stride = stride
        self.topk = topk
        self.dims = dims
        self.enable_crop = enable_crop
        self.scale = dims[0] ** -0.5

        x_layer = TransformerLayer(dims[0], **kwargs)
        self.x_layers = nn.ModuleList(
            [deepcopy(x_layer) for _ in range(num_x_layers)]
        )
        self.fuser = PyramidFuser(dims)
        y_layer = RegionSelectiveTransformerLayer(stride, dims[1], **kwargs)
        self.y_layers = nn.ModuleList(
            [deepcopy(y_layer) for _ in range(num_y_layers)]
        )

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Tensor,
        y1: Tensor,
        encoding: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor, Tensor]:
        n, _, fh0, fw0 = x0.shape
        _, _, fh1, fw1 = x1.shape

        x0, x1 = x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1)
        if mask0 is None and mask1 is None:
            for layer in self.x_layers:
                x0, x1 = layer(x0, x1, encoding)
        elif not self.enable_crop:
            mask0 = mask0.flatten(start_dim=1)
            mask1 = mask1.flatten(start_dim=1)
            mask00 = mask0[:, None, :, None] & mask0[:, None, None, :]
            mask11 = mask1[:, None, :, None] & mask1[:, None, None, :]
            mask01 = mask0[:, None, :, None] & mask1[:, None, None, :]
            for layer in self.x_layers:
                x0, x1 = layer(x0, x1, encoding, mask00, mask11, mask01)
        else:
            _x0, _x1 = torch.zeros_like(x0), torch.zeros_like(x1)
            for b in range(n):
                h0, w0 = [mask0[b].sum(dim=t).amax() for t in [0, 1]]
                h1, w1 = [mask1[b].sum(dim=t).amax() for t in [0, 1]]
                b_x0, b_x1 = x0[[b], :h0, :w0], x1[[b], :h1, :w1]
                for layer in self.x_layers:
                    b_x0, b_x1 = layer(b_x0, b_x1, encoding)
                _x0[[b], :h0, :w0], _x1[[b], :h1, :w1] = b_x0, b_x1
            x0, x1 = _x0, _x1

        _x0 = x0.flatten(start_dim=1, end_dim=2) * self.scale
        _x1 = x1.flatten(start_dim=1, end_dim=2)
        similarity = _x0 @ _x1.transpose(-1, -2)
        if mask0 is not None and mask1 is not None:
            mask = mask0.view(n, -1, 1) & mask1.view(n, 1, -1)
            similarity.masked_fill_(~mask, -float("inf"))
        x0, x1 = x0.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2)

        y0, y1 = self.fuser([x0, y0], [x1, y1])
        y0 = (
            y0.reshape(n, -1, fh0, self.stride, fw0, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh0 * fw0, self.stride * self.stride, -1)
        )
        y1 = (
            y1.reshape(n, -1, fh1, self.stride, fw1, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh1 * fw1, self.stride * self.stride, -1)
        )
        _, indices0_to_1 = similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = similarity.transpose(-1, -2).topk(self.topk, dim=-1)
        for layer in self.y_layers:
            y0, y1 = layer(
                y0, y1, indices0_to_1, indices1_to_0, (fh0, fw0), (fh1, fw1)
            )
        return [y0, x0], [y1, x1], indices0_to_1, indices1_to_0, similarity
