import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

MAX_SHAPE = 256, 256


class SinePositionalEncoding(nn.Module):
    # TODO:
    # Change usage of fp16
    def __init__(
        self,
        feat_dim: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()

        factor = (
            -math.log(10000.0) / (feat_dim // 4) * torch.arange(feat_dim // 4)
        ).exp()
        y = factor * torch.ones(*MAX_SHAPE, 1).cumsum(0)
        x = factor * torch.ones(*MAX_SHAPE, 1).cumsum(1)

        if test_size is not None:
            y *= min(train_size[0] / test_size[0], 1.0)
            x *= min(train_size[1] / test_size[1], 1.0)

        sin = torch.stack([y.sin(), x.sin()], dim=-1).flatten(start_dim=-2)
        cos = torch.stack([y.cos(), x.cos()], dim=-1).flatten(start_dim=-2)
        pos_enc = (
            torch.stack([sin, cos], dim=-1)
            .flatten(start_dim=-2)
            .permute(2, 0, 1)
        )
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        if fp16:
            pos_enc, sin, cos = pos_enc.half(), sin.half(), cos.half()

        self.register_buffer("pos_enc", pos_enc, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
        out = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return out

    def forward(self, x: torch.Tensor, type: str) -> torch.Tensor:
        if type == "abs":
            _, c, h, w = x.shape
            out = x + self.pos_enc[:c, :h, :w]
        elif type == "rel":
            _, h, w, c = x.shape
            out = (  # fmt: skip
                (self.cos[:h, :w, :c] * x)
                + (self.sin[:h, :w, :c] * self._rotate_half(x))
            )
        else:
            raise ValueError()
        return out
