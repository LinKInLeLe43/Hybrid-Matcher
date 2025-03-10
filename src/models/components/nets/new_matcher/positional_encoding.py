# TODO:
# - Change usage of fp16

from math import log
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinePositionalEncoding(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        train_size: Tuple[int, int],
        test_size: Optional[Tuple[int, int]] = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()

        max_shape = 256, 256
        range = torch.arange(feat_dim // 4)
        factors = (-log(10000.0) / (feat_dim // 4) * range).exp()
        y = factors * torch.ones(*max_shape, 1).cumsum(0)
        x = factors * torch.ones(*max_shape, 1).cumsum(1)
        if test_size is not None:
            y = y * min(train_size[0] / test_size[0], 1.0)
            x = x * min(train_size[1] / test_size[1], 1.0)

        _sin = torch.stack([y.sin(), x.sin()], dim=-1).flatten(start_dim=-2)
        _cos = torch.stack([y.cos(), x.cos()], dim=-1).flatten(start_dim=-2)
        sin = _sin.repeat_interleave(2, dim=-1)
        cos = _cos.repeat_interleave(2, dim=-1)
        pos_enc = torch.stack([_sin, _cos], dim=-1)
        pos_enc = pos_enc.flatten(start_dim=-2).permute(2, 0, 1)
        if fp16:
            sin, cos, pos_enc = sin.half(), cos.half(), pos_enc.half()

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("pos_enc", pos_enc, persistent=False)

    def _rotate_half(self, feat: torch.Tensor) -> torch.Tensor:
        feat0, feat1 = feat.unflatten(-1, (-1, 2)).unbind(dim=-1)
        feat = torch.stack([-feat1, feat0], dim=-1).flatten(start_dim=-2)
        return feat

    def forward(self, feat: torch.Tensor, type: str) -> torch.Tensor:
        if type == "abs":
            _, c, h, w = feat.shape
            feat = feat + self.pos_enc[:c, :h, :w]
        elif type == "rel":
            _, h, w, c = feat.shape
            sin_term = self.sin[:h, :w, :c] * self._rotate_half(feat)
            cos_term = self.cos[:h, :w, :c] * feat
            feat = sin_term + cos_term
        else:
            raise ValueError()
        return feat
