from typing import Optional, Tuple
import math

import torch
from torch import device
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


class RoPESinePositionalEncoding(nn.Module):
    def __init__(
        self,
        depth: int,
        train_size: Tuple[int, int],
        test_size: Tuple[int, int],
        fp16: bool = False
    ) -> None:
        super().__init__()
        self.depth = depth
        self.train_size = tuple(train_size)
        self.test_size = tuple(test_size)
        self.fp16 = fp16

        self.pe = torch.empty(0)
        self.sin = torch.empty(0)
        self.cos = torch.empty(0)

    def _build_tables(self, device: torch.device, dtype: torch.dtype) -> None:
        max_shape = 256, 256
        depth = self.depth
        train_size = self.train_size
        test_size = self.test_size

        factor = torch.arange(depth // 4, device=device, dtype=dtype)[None, None, :]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        x = factor * torch.ones(max_shape, device=device, dtype=dtype).cumsum(1)[:, :, None]
        y = factor * torch.ones(max_shape, device=device, dtype=dtype).cumsum(0)[:, :, None]

        if test_size != train_size:
            x *= train_size[1] / test_size[1]
            y *= train_size[0] / test_size[0]

        sin = torch.zeros((*max_shape, depth // 2), device=device, dtype=dtype)
        cos = torch.zeros((*max_shape, depth // 2), device=device, dtype=dtype)
        sin[..., 0::2] = y.sin()
        sin[..., 1::2] = x.sin()
        cos[..., 0::2] = y.cos()
        cos[..., 1::2] = x.cos()

        pe = (torch.stack([sin, cos], dim=-1).flatten(start_dim=-2)
              .permute(2, 0, 1))
        sin = sin.repeat_interleave(2, dim=2)
        cos = cos.repeat_interleave(2, dim=2)

        self.pe = pe
        self.sin = sin
        self.cos = cos

    def _ensure_tables(self, x: torch.Tensor) -> None:
        target_dtype = torch.float16 if (self.fp16 or x.dtype == torch.float16) else torch.float32
        if self.pe.numel() == 0 or  str(self.pe.device) != str(x.device) or self.pe.dtype != target_dtype:
            self._build_tables(x.device, target_dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.unflatten(-1, (x.shape[-1] // 2, 2)).unbind(dim=-1)
        out = torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)
        return out

    def abs_pe(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_tables(x)
        _, c, h, w = x.shape
        out = x + self.pe[:c, :h, :w]
        return out

    def rel_pe(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_tables(x)
        _, h, w, c = x.shape
        out = (self.cos[:h, :w, :c] * x +
               self.sin[:h, :w, :c] * self._rotate_half(x))
        return out
