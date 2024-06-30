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

        factor = torch.arange(depth // 4)[:, None, None]
        factor = (-math.log(10000.0) / (depth // 4) * factor).exp()

        max_shape = 256, 256
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


# class LearnableFourierPositionalEncoding(nn.Module):
#     def __init__(
#         self,
#         in_depth: int,
#         out_depth: Optional[int] = None,
#         dropout: float = 0.0
#     ) -> None:
#         super().__init__()
#         self.in_depth = in_depth
#
#         if out_depth is None:
#             out_depth = in_depth
#
#         self.proj = nn.Linear(2, in_depth // 2)
#         self.mlp = Mlp(in_depth, in_depth, out_depth, dropout=dropout)
#
#         # TODO: check weight init
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#
#     def get(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, h, w = x.shape
#         device = x.device
#
#         coors = K.create_meshgrid(h, w, device=device) / 2
#         out = self.proj(coors)
#         out = torch.cat([out.cos(), out.sin()], dim=3)
#         out /= self.in_depth ** 0.5
#         out = self.mlp(out)
#         out = out.permute(0, 3, 1, 2)
#         return out
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         positional_encoding = self.get(x)
#         out = x + positional_encoding
#         return out, positional_encoding


# class ContinuousPositionBias(nn.Module):
#     def __init__(self, head_count: int, window_size: Tuple[int, int]) -> None:
#         super().__init__()
#         self.window_size = window_size
#
#         self.mlp = nn.Sequential(
#             nn.Linear(2, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, head_count, bias=False))
#
#         h, w = window_size
#         relative_coords = K.create_meshgrid(2 * h - 1, 2 * w - 1)[0]
#         relative_coords = (relative_coords.sign() *
#                            (1.0 + (8 * relative_coords).abs()).log2() /
#                            torch.tensor(8).log2())
#         self.register_buffer("relative_coords", relative_coords)
#
#         relative_idxes = K.create_meshgrid(
#             h, w, normalized_coordinates=False, dtype=torch.long)[0]
#         relative_idxes = relative_idxes.flatten(end_dim=-2)
#         relative_idxes = (relative_idxes[:, None] - relative_idxes[None, :] +
#                           torch.tensor([w - 1, h - 1]))
#         relative_idxes = ((2 * w - 1) * relative_idxes[:, :, 1] +
#                           relative_idxes[:, :, 0])
#         relative_idxes = relative_idxes.flatten()
#         self.register_buffer("relative_idxes", relative_idxes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, l, _ = x.shape
#         h, w = self.window_size
#
#         out = self.mlp(self.relative_coords)
#         out = out.flatten(end_dim=-2)
#         out = out.index_select(0, self.relative_idxes)
#         out = out.unflatten(0, (h * w, h * w)).permute(2, 0, 1)
#         out = 16 * out.sigmoid()
#         out = F.pad(out, (l - h * w, 0, l - h * w, 0))
#         return out


# class PosEmbMLPSwinv1D(nn.Module):
#     def __init__(self, depth: int, seq_length=4) -> None:
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(2, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, depth, bias=False))
#
#         self.grid_exists = False
#         self.pos_emb = None
#         self.deploy = False
#         relative_bias = torch.zeros(1, seq_length, depth)
#         self.register_buffer("relative_bias", relative_bias)
#
#     def switch_to_deploy(self):
#         self.deploy = True
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         seq_length = x.shape[1]
#         if self.deploy:
#             return x + self.relative_bias
#         else:
#             self.grid_exists = False
#         if not self.grid_exists:
#             self.grid_exists = True
#
#             seq_length = int(seq_length**0.5)
#
#             relative_coords_table = K.create_meshgrid(seq_length, seq_length, device=x.device)
#
#             self.pos_emb = self.mlp(relative_coords_table.flatten(2).transpose(1,2))
#
#             self.relative_bias = self.pos_emb
#         x = x + self.pos_emb
#         return x
