import math
from typing import Optional, Tuple

import torch
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
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()
        max_shape = 256, 256

        factor = torch.arange(depth // 4)[None, None, :]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = factor * torch.ones(max_shape).cumsum(1)[:, :, None]
        y = factor * torch.ones(max_shape).cumsum(0)[:, :, None]

        if test_size is not None and test_size != train_size:
            x *= train_size[1] / test_size[1]
            y *= train_size[0] / test_size[0]

        sin = torch.zeros((*max_shape, depth // 2))
        cos = torch.zeros((*max_shape, depth // 2))
        sin[..., 0::2] = y.sin()
        sin[..., 1::2] = x.sin()
        cos[..., 0::2] = y.cos()
        cos[..., 1::2] = x.cos()

        pe = (
            torch.stack([sin, cos], dim=-1)
            .flatten(start_dim=-2)
            .permute(2, 0, 1)
        )
        sin = sin.repeat_interleave(2, dim=2)
        cos = cos.repeat_interleave(2, dim=2)

        if fp16:
            pe, sin, cos = pe.half(), sin.half(), cos.half()

        self.register_buffer("pe", pe, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
        out = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return out

    def abs_pe(self, x: Tensor) -> Tensor:
        _, c, h, w = x.shape
        out = x + self.pe[:c, :h, :w]
        return out

    def rel_pe(self, x: Tensor) -> Tensor:
        _, h, w, c = x.shape
        out = self.cos[:h, :w, :c] * x + self.sin[
            :h, :w, :c
        ] * self._rotate_half(x)
        return out

    def get_encoding(self) -> Tensor:
        encoding = torch.stack([self.cos, self.sin])
        return encoding
