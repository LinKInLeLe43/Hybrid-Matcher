from copy import deepcopy
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .transformer import (
    RegionSelectiveCrossBlock,
    RegionSelectiveTransformerLayer,
    TransformerLayer,
)


class Decoder(Module):
    def __init__(
        self, num_layers: int, enable_crop: bool = True, **kwargs
    ) -> None:
        super().__init__()
        layer = TransformerLayer(**kwargs)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )
        self.enable_crop = enable_crop

    def _forward(
        self,
        x0: Tensor,
        x1: Tensor,
        encoding: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        mask00 = mask11 = mask01 = None
        if mask0 is not None and mask1 is not None:
            mask0 = mask0.flatten(start_dim=1)
            mask1 = mask1.flatten(start_dim=1)
            mask00 = mask0[:, None, :, None] & mask0[:, None, None, :]
            mask11 = mask1[:, None, :, None] & mask1[:, None, None, :]
            mask01 = mask0[:, None, :, None] & mask1[:, None, None, :]
        for layer in self.layers:
            x0, x1 = layer(
                x0, x1, encoding, mask00=mask00, mask11=mask11, mask01=mask01
            )
        return x0, x1

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        encoding: Tensor,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x0 = x0.permute(0, 2, 3, 1)
        x1 = x1.permute(0, 2, 3, 1)
        if mask0 is not None and mask1 is not None and self.enable_crop:
            x0_t, x1_t = torch.zeros_like(x0), torch.zeros_like(x1)
            for b in range(x0.shape[0]):
                h0 = mask0[b].sum(dim=0).amax()
                w0 = mask0[b].sum(dim=1).amax()
                h1 = mask1[b].sum(dim=0).amax()
                w1 = mask1[b].sum(dim=1).amax()
                x0_t[[b], :h0, :w0], x1_t[[b], :h1, :w1] = self._forward(
                    x0[[b], :h0, :w0], x1[[b], :h1, :w1], encoding
                )
        else:
            x0_t, x1_t = self._forward(
                x0, x1, encoding, mask0=mask0, mask1=mask1
            )
        x0_t = x0_t.permute(0, 3, 1, 2)
        x1_t = x1_t.permute(0, 3, 1, 2)
        return x0_t, x1_t
