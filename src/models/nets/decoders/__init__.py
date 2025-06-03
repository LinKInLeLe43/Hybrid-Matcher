from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .transformer import (
    AggregatedEncoder,
    ConvTransformerEncoder,
    FusedSelectiveTransformer,
    TransformerEncoder,
    TransformerLayer,
)


class Decoder(Module):
    def __init__(
        self, layer: Module, num_layers: int, enable_crop: bool = True
    ) -> None:
        super().__init__()
        self.enable_crop = enable_crop

        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

    def _forward(
        self,
        x0: Tensor,
        x1: Tensor,
        rope,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if mask0 is not None and mask1 is not None:
            mask0 = mask0.flatten(start_dim=1)
            mask1 = mask1.flatten(start_dim=1)
        x0, x1 = x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1)
        for layer in self.layers:
            x0, x1 = layer(x0, x1, rope, mask0=mask0, mask1=mask1)
        x0, x1 = x0.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2)
        return x0, x1

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        rope,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if mask0 is None and mask1 is None:
            x0_t, x1_t = self._forward(x0, x1, rope)
        elif not self.enable_crop:
            x0_t, x1_t = self._forward(x0, x1, rope, mask0, mask1)
        else:
            x0_t, x1_t = torch.zeros_like(x0), torch.zeros_like(x1)
            for b in range(x0.shape[0]):
                h0 = mask0[b].sum(dim=0).amax()
                w0 = mask0[b].sum(dim=1).amax()
                h1 = mask1[b].sum(dim=0).amax()
                w1 = mask1[b].sum(dim=1).amax()
                b_x0_t, b_x1_t = self._forward(
                    x0[[b], :, :h0, :w0], x1[[b], :, :h1, :w1], rope
                )
                x0_t[[b], :, :h0, :w0] = b_x0_t
                x1_t[[b], :, :h1, :w1] = b_x1_t
        return x0_t, x1_t
