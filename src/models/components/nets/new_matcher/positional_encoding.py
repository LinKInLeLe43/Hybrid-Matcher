from math import log
from typing import Optional, Tuple

import torch
from torch.nn import Module


class RoPESinePositionalEncoding(Module):
    def __init__(
        self,
        dim: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()
        dim = dim // 4
        max_shape = 128, 128
        factor = (torch.arange(dim) * -log(10000.0) / dim).exp()
        y = torch.ones(*max_shape, 1).cumsum(0) * factor
        x = torch.ones(*max_shape, 1).cumsum(1) * factor

        if test_size is not None:
            y = y * min(train_size[0] / test_size[0], 1.0)
            x = x * min(train_size[1] / test_size[1], 1.0)

        sin = torch.stack([y.sin(), x.sin()], dim=-1).flatten(start_dim=-2)
        cos = torch.stack([y.cos(), x.cos()], dim=-1).flatten(start_dim=-2)
        pe = (
            torch.stack([sin, cos], dim=-1)
            .flatten(start_dim=-2)
            .permute(2, 0, 1)
        )
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        if fp16:
            pe, sin, cos = pe.half(), sin.half(), cos.half()

        self.register_buffer("pe", pe, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
        out = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return out

    def abs_pe(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        out = x + self.pe[:c, :h, :w]
        return out

    def rel_pe(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w, c = x.shape
        out = self.cos[:h, :w, :c] * x + self.sin[
            :h, :w, :c
        ] * self._rotate_half(x)
        return out
