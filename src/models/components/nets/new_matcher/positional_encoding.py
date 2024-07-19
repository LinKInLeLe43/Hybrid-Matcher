from typing import Optional, Tuple
import math

import kornia as K
import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth)
        self.linear1 = nn.Linear(hidden_depth, out_depth)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout(x)
        return x


class SinePositionalEncoding(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        max_shape = 256, 256

        factor = torch.arange(depth // 4)[:, None, None]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = factor * torch.ones(max_shape).cumsum(1)
        y = factor * torch.ones(max_shape).cumsum(0)

        positional_encoding = torch.zeros((depth, *max_shape))
        positional_encoding[0::4, ...] = x.sin()
        positional_encoding[1::4, ...] = x.cos()
        positional_encoding[2::4, ...] = y.sin()
        positional_encoding[3::4, ...] = y.cos()
        self.register_buffer(
            "positional_encoding", positional_encoding, persistent=False)

    def get(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        positional_encoding = self.positional_encoding[None, :, :h, :w]
        return positional_encoding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positional_encoding = self.get(x)
        out = x + positional_encoding
        return out, positional_encoding


class RoPESinePositionalEncoding(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        max_shape = 256, 256

        factor = torch.arange(depth // 4)[None, None, :]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = factor * torch.ones(max_shape).cumsum(1)[:, :, None]
        y = factor * torch.ones(max_shape).cumsum(0)[:, :, None]

        sin = torch.zeros((*max_shape, depth // 2))
        cos = torch.zeros((*max_shape, depth // 2))
        sin[..., 0::2] = y.sin()
        sin[..., 1::2] = x.sin()
        cos[..., 0::2] = y.cos()
        cos[..., 1::2] = x.cos()

        sin = sin.repeat_interleave(2, dim=2)
        cos = cos.repeat_interleave(2, dim=2)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def rotate_half(self, x:torch.Tensor):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = x.shape
        out = self.cos[:h, :w] * x + self.sin[:h, :w] * self.rotate_half(x)
        return out


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        in_depth: int,
        out_depth: Optional[int] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_depth = in_depth

        if out_depth is None:
            out_depth = in_depth

        self.proj = nn.Linear(2, in_depth // 2)
        self.mlp = Mlp(in_depth, in_depth, out_depth, dropout=dropout)

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        device = x.device

        coors = K.create_meshgrid(h, w, device=device) / 2
        out = self.proj(coors)
        out = torch.cat([out.cos(), out.sin()], dim=3)
        out /= self.in_depth ** 0.5
        out = self.mlp(out)
        out = out.permute(0, 3, 1, 2)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positional_encoding = self.get(x)
        out = x + positional_encoding
        return out, positional_encoding
