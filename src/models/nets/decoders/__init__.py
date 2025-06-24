from copy import deepcopy
from typing import Optional, Sequence, Tuple

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
        num_z_layers: int,
        enable_crop: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(dims) == 3
        self.stride = stride
        self.topk = topk
        self.dims = dims
        self.enable_crop = enable_crop
        self.x_scale = dims[0] ** -0.5
        self.y_scale = dims[1] ** -0.5

        x_layer = TransformerLayer(dims[0], **kwargs)
        self.x_layers = nn.ModuleList(
            [deepcopy(x_layer) for _ in range(num_x_layers)]
        )
        self.x_fuser = PyramidFuser(dims[0:2])
        y_layer = RegionSelectiveTransformerLayer(stride, dims[1], **kwargs)
        self.y_layers = nn.ModuleList(
            [deepcopy(y_layer) for _ in range(num_y_layers)]
        )
        self.y_fuser = PyramidFuser(dims[1:3])
        z_layer = RegionSelectiveTransformerLayer(stride, dims[2], **kwargs)
        self.z_layers = nn.ModuleList(
            [deepcopy(z_layer) for _ in range(num_z_layers)]
        )

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Tensor,
        y1: Tensor,
        z0: Tensor,
        z1: Tensor,
        encoding: Tensor,
        x0_mask: Optional[Tensor] = None,
        x1_mask: Optional[Tensor] = None,
        y0_mask: Optional[Tensor] = None,
        y1_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        n, _, fh0, fw0 = x0.shape
        _, _, fh1, fw1 = x1.shape

        x0, x1 = x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1)
        if x0_mask is None and x1_mask is None:
            for layer in self.x_layers:
                x0, x1 = layer(x0, x1, encoding)
        elif not self.enable_crop:
            x0_mask = x0_mask.flatten(start_dim=1)
            x1_mask = x1_mask.flatten(start_dim=1)
            mask00 = x0_mask[:, None, :, None] & x0_mask[:, None, None, :]
            mask11 = x1_mask[:, None, :, None] & x1_mask[:, None, None, :]
            mask01 = x0_mask[:, None, :, None] & x1_mask[:, None, None, :]
            for layer in self.x_layers:
                x0, x1 = layer(x0, x1, encoding, mask00, mask11, mask01)
        else:
            _x0, _x1 = torch.zeros_like(x0), torch.zeros_like(x1)
            for b in range(n):
                h0, w0 = [x0_mask[b].sum(dim=t).amax() for t in [0, 1]]
                h1, w1 = [x1_mask[b].sum(dim=t).amax() for t in [0, 1]]
                b_x0, b_x1 = x0[[b], :h0, :w0], x1[[b], :h1, :w1]
                for layer in self.x_layers:
                    b_x0, b_x1 = layer(b_x0, b_x1, encoding)
                _x0[[b], :h0, :w0], _x1[[b], :h1, :w1] = b_x0, b_x1
            x0, x1 = _x0, _x1

        _x0 = x0.flatten(start_dim=1, end_dim=2) * self.x_scale
        _x1 = x1.flatten(start_dim=1, end_dim=2)
        x_similarity = _x0 @ _x1.transpose(-1, -2)
        if x0_mask is not None and x1_mask is not None:
            mask = x0_mask.view(n, -1, 1) & x1_mask.view(n, 1, -1)
            x_similarity.masked_fill_(~mask, -float("inf"))
        x0, x1 = x0.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2)

        y0, y1 = self.x_fuser([x0, y0], [x1, y1])
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
        _, indices0_to_1 = x_similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = x_similarity.transpose(-1, -2).topk(
            self.topk, dim=-1
        )
        for layer in self.y_layers:
            y0, y1 = layer(
                y0, y1, indices0_to_1, indices1_to_0, (fh0, fw0), (fh1, fw1)
            )
        y0 = (
            y0.reshape(n, fh0, fw0, self.stride, self.stride, -1)
            .transpose(2, 3)
            .reshape(n, fh0 * self.stride, fw0 * self.stride, -1)
        )
        y1 = (
            y1.reshape(n, fh1, fw1, self.stride, self.stride, -1)
            .transpose(2, 3)
            .reshape(n, fh1 * self.stride, fw1 * self.stride, -1)
        )
        _y0 = y0.flatten(start_dim=1, end_dim=2) * self.y_scale
        _y1 = y1.flatten(start_dim=1, end_dim=2)
        y_similarity = _y0 @ _y1.transpose(-1, -2)
        if y0_mask is not None and y1_mask is not None:
            mask = y0_mask.view(n, -1, 1) & y1_mask.view(n, 1, -1)
            y_similarity.masked_fill_(~mask, -float("inf"))
        y0 = y0.permute(0, 3, 1, 2).contiguous()
        y1 = y1.permute(0, 3, 1, 2).contiguous()

        fh0, fw0, fh1, fw1 = [t * self.stride for t in [fh0, fw0, fh1, fw1]]
        z0, z1 = self.y_fuser([y0, z0], [y1, z1])
        z0 = (
            z0.reshape(n, -1, fh0, self.stride, fw0, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh0 * fw0, self.stride * self.stride, -1)
        )
        z1 = (
            z1.reshape(n, -1, fh1, self.stride, fw1, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh1 * fw1, self.stride * self.stride, -1)
        )
        _, indices0_to_1 = y_similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = y_similarity.transpose(-1, -2).topk(
            self.topk, dim=-1
        )
        for layer in self.z_layers:
            z0, z1 = layer(
                z0, z1, indices0_to_1, indices1_to_0, (fh0, fw0), (fh1, fw1)
            )
        return z0, z1, indices0_to_1, indices1_to_0, x_similarity, y_similarity
