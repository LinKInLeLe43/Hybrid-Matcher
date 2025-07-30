import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class SinePositionalEncoding(Module):
    def __init__(
        self,
        depth: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
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
            "positional_encoding", positional_encoding, persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pe = self.positional_encoding[None, :, :h, :w]
        out = x + pe
        return out


class RoPESinePositionalEncoding(Module):
    def __init__(
        self,
        depth: int,
        num_heads: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        max_shape = 256, 256

        factor = torch.arange(depth // 4)[None, None, :]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = torch.ones(max_shape).cumsum(1)[:, :, None]
        y = torch.ones(max_shape).cumsum(0)[:, :, None]

        if test_size is not None and test_size != train_size:
            x *= train_size[1] / test_size[1]
            y *= train_size[0] / test_size[0]

        freqs_x = []
        freqs_y = []
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi
            fx = torch.cat([factor * torch.cos(angles), factor * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([factor * torch.sin(angles), factor * torch.sin(torch.pi/2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
        self.freqs = nn.Parameter(freqs, requires_grad=True)

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("y", y, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.x * self.freqs[0] + self.y * self.freqs[1]
