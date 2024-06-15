from typing import Tuple

import kornia as K
import torch
from torch import nn


# class ContinuousPositionBias(nn.Module):
#     def __init__(self, head_count: int, test_window_size: Tuple[int, int], train_window_size: Tuple[int, int]):
#         super().__init__()
#
#         self.mlp = nn.Sequential(
#             nn.Linear(2, 512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, head_count, bias=False))
#
#         relative_coords_table = K.create_meshgrid(
#             2 * test_window_size[0] - 1, 2 * test_window_size[1] - 1)
#         if test_window_size != train_window_size:
#             relative_coords_table[..., 0] *= ((train_window_size[1] - 1) / )
#         torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)


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