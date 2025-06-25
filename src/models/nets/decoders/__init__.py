from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from ..submodules import PyramidFuser
from .transformer import RegionSelectiveTransformerLayer, TransformerLayer


class GlobalDecoder(Module):
    def __init__(
        self, dim: int, num_layers: int, enable_crop: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.enable_crop = enable_crop

        layer = TransformerLayer(dim, **kwargs)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        encoding: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0, x1 = x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1)
        if mask0 is None and mask1 is None:
            for layer in self.layers:
                x0, x1 = layer(x0, x1, encoding=encoding)
        elif not self.enable_crop:
            mask0 = mask0.flatten(start_dim=1)
            mask1 = mask1.flatten(start_dim=1)
            mask00 = mask0[:, None, :, None] & mask0[:, None, None, :]
            mask11 = mask1[:, None, :, None] & mask1[:, None, None, :]
            mask01 = mask0[:, None, :, None] & mask1[:, None, None, :]
            for layer in self.layers:
                x0, x1 = layer(
                    x0,
                    x1,
                    encoding=encoding,
                    mask00=mask00,
                    mask11=mask11,
                    mask01=mask01,
                )
        else:
            _x0, _x1 = torch.zeros_like(x0), torch.zeros_like(x1)
            for b in range(x0.shape[0]):
                h0, w0 = mask0[b].sum(dim=0).amax(), mask0[b].sum(dim=1).amax()
                h1, w1 = mask1[b].sum(dim=0).amax(), mask1[b].sum(dim=1).amax()
                b_x0, b_x1 = x0[[b], :h0, :w0], x1[[b], :h1, :w1]
                for layer in self.layers:
                    b_x0, b_x1 = layer(b_x0, b_x1, encoding=encoding)
                _x0[[b], :h0, :w0], _x1[[b], :h1, :w1] = b_x0, b_x1
            x0, x1 = _x0, _x1
        x0, x1 = x0.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2)
        return x0, x1


class SelectiveDecoder(Module):
    def __init__(
        self,
        dim_list: Sequence[int],
        stride: int,
        topk: int,
        num_layers: int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(dim_list) == 2
        self.stride = stride
        self.topk = topk
        self.scale = dim_list[1] ** -0.5

        self.fuser = PyramidFuser(dim_list)
        layer = RegionSelectiveTransformerLayer(stride, dim_list[0], **kwargs)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        prior0: Tensor,
        prior1: Tensor,
        do_reshape: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        n, _, fh0, fw0 = prior0.shape
        _, _, fh1, fw1 = prior1.shape

        _prior0 = prior0.flatten(start_dim=2) * self.scale
        _prior1 = prior1.flatten(start_dim=2)
        similarity = _prior0.transpose(-1, -2) @ _prior1
        _, indices0_to_1 = similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = similarity.transpose(-1, -2).topk(self.topk, dim=-1)
        if not self.training:
            similarity = None

        x0, x1 = self.fuser([x0, prior0], [x1, prior1])
        x0 = (
            x0.reshape(n, -1, fh0, self.stride, fw0, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh0 * fw0, self.stride * self.stride, -1)
        )
        x1 = (
            x1.reshape(n, -1, fh1, self.stride, fw1, self.stride)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh1 * fw1, self.stride * self.stride, -1)
        )
        for layer in self.layers:
            x0, x1 = layer(x0, x1, indices0_to_1, indices1_to_0)
        if do_reshape:
            x0 = (
                x0.reshape(n, fh0, fw0, self.stride, self.stride, -1)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(n, -1, fh0 * self.stride, fw0 * self.stride)
            )
            x1 = (
                x1.reshape(n, fh1, fw1, self.stride, self.stride, -1)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(n, -1, fh1 * self.stride, fw1 * self.stride)
            )
        return x0, x1, indices0_to_1, indices1_to_0, similarity


class Decoder(Module):
    def __init__(
        self,
        dim_list: Sequence[int],
        stride_list: Sequence[int],
        topk_list: Sequence[int],
        num_layers_list: Sequence[int],
        enable_crop: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim_list = dim_list  # used in coarse matching
        self.stride_list = stride_list  # used in coarse matching

        self.global_decoder = GlobalDecoder(
            dim_list[-1],
            num_layers_list[-1],
            enable_crop=enable_crop,
            **kwargs,
        )
        self.selective_decoders = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.selective_decoders.append(
                SelectiveDecoder(
                    dim_list[i : i + 2],
                    stride_list[i],
                    topk_list[i],
                    num_layers_list[i],
                    **kwargs,
                )
            )

    def forward(
        self,
        x0_list: Sequence[Tensor],
        x1_list: Sequence[Tensor],
        encoding: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Optional[Tensor]]]:
        x0, x1 = self.global_decoder(
            x0_list[-1], x1_list[-1], encoding, mask0=mask0, mask1=mask1
        )
        similarity_list = []
        for i in reversed(range(len(self.selective_decoders))):
            x0, x1, indices0_to_1, indices1_to_0, similarity = (
                self.selective_decoders[i](
                    x0_list[i], x1_list[i], x0, x1, do_reshape=i != 0
                )
            )
            similarity_list.insert(0, similarity)
        return x0, x1, indices0_to_1, indices1_to_0, similarity_list
