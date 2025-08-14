from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class SinusoidalPositionalEncoding(Module):
    def __init__(
        self,
        dim: int,
        factor: int,
        patch_size: int,
        train_size: int,
        test_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.factor = factor // 8
        self.patch_size = patch_size
        self.train_size = train_size // 8
        self.test_size = test_size // 8 if test_size is not None else None
        stride = self.factor * patch_size

        theta = 10000
        freqs = (
            -torch.tensor(theta).log() * torch.arange(0, dim, 4) / dim
        ).exp()
        h, w = 256 // stride, 256 // stride
        freqs_y = ((torch.arange(h) + 0.5) * stride)[:, None, None] * freqs
        freqs_x = ((torch.arange(w) + 0.5) * stride)[None, :, None] * freqs
        freqs_y, freqs_x = freqs_y.expand(-1, w, -1), freqs_x.expand(h, -1, -1)
        if test_size is not None:
            if test_size > train_size:
                freqs_y = freqs_y * (train_size - 1) / (test_size - 1)
                freqs_x = freqs_x * (train_size - 1) / (test_size - 1)
        else:
            self.register_buffer("freqs_y", freqs_y, persistent=False)
            self.register_buffer("freqs_x", freqs_x, persistent=False)

        freqs_sin = torch.stack(
            [t.sin() for t in [freqs_y, freqs_x]], dim=-1
        ).flatten(start_dim=-2)
        freqs_cos = torch.stack(
            [t.cos() for t in [freqs_y, freqs_x]], dim=-1
        ).flatten(start_dim=-2)
        encoding = torch.stack([freqs_sin, freqs_cos], dim=-1).flatten(
            start_dim=-2
        )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        h, w = size[0] // self.patch_size, size[1] // self.patch_size
        test_size = max(size[0] * self.factor, size[1] * self.factor)
        if self.test_size is None and test_size > self.train_size:
            freqs_y = self.freqs_y * (self.train_size - 1) / (test_size - 1)
            freqs_x = self.freqs_x * (self.train_size - 1) / (test_size - 1)
            freqs_sin = torch.stack(
                [t.sin() for t in [freqs_y, freqs_x]], dim=-1
            ).flatten(start_dim=-2)
            freqs_cos = torch.stack(
                [t.cos() for t in [freqs_y, freqs_x]], dim=-1
            ).flatten(start_dim=-2)
            encoding = torch.stack([freqs_sin, freqs_cos], dim=-1).flatten(
                start_dim=-2
            )[:h, :w]
        else:
            encoding = self.encoding[:h, :w]
        return encoding
