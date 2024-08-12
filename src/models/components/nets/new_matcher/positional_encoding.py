from typing import Optional, Tuple
import math

import torch
from torch import nn


class SinePositionalEncoding(nn.Module):
    def __init__(
        self,
        depth: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__()
        max_shape = 256, 256

        factor = torch.arange(depth // 4)[:, None, None]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = factor * torch.ones(max_shape).cumsum(1)
        y = factor * torch.ones(max_shape).cumsum(0)

        if test_size is not None and test_size != train_size:
            x *= train_size[1] / test_size[1]
            y *= train_size[0] / test_size[0]

        positional_encoding = torch.zeros((depth, *max_shape))
        positional_encoding[0::4, ...] = x.sin()
        positional_encoding[1::4, ...] = x.cos()
        positional_encoding[2::4, ...] = y.sin()
        positional_encoding[3::4, ...] = y.cos()
        self.register_buffer(
            "positional_encoding", positional_encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pe = self.positional_encoding[None, :, :h, :w]
        out = x + pe
        return out
