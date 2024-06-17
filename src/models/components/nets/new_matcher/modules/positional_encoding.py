from typing import Tuple

import kornia as K
import torch
from torch import nn
from torch.nn import functional as F


class ContinuousPositionBias(nn.Module):
    def __init__(self, head_count: int, window_size: Tuple[int, int]) -> None:
        super().__init__()
        self.window_size = window_size

        self.mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, head_count, bias=False))

        h, w = window_size
        relative_coords = K.create_meshgrid(2 * h - 1, 2 * w - 1)[0]
        relative_coords = (relative_coords.sign() *
                           (1.0 + (8 * relative_coords).abs()).log2() /
                           torch.tensor(8).log2())
        self.register_buffer("relative_coords", relative_coords)

        relative_idxes = K.create_meshgrid(
            h, w, normalized_coordinates=False, dtype=torch.long)[0]
        relative_idxes = relative_idxes.flatten(end_dim=-2)
        relative_idxes = (relative_idxes[:, None] - relative_idxes[None, :] +
                          torch.tensor([w - 1, h - 1]))
        relative_idxes = ((2 * w - 1) * relative_idxes[:, :, 1] +
                          relative_idxes[:, :, 0])
        relative_idxes = relative_idxes.flatten()
        self.register_buffer("relative_idxes", relative_idxes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, l, _ = x.shape
        h, w = self.window_size

        out = self.mlp(self.relative_coords)
        out = out.flatten(end_dim=-2)
        out = out.index_select(0, self.relative_idxes)
        out = out.unflatten(0, (h * w, h * w)).permute(2, 0, 1)
        out = 16 * out.sigmoid()
        out = F.pad(out, (l - h * w, 0, l - h * w, 0))
        return out


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