from typing import Tuple

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
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)

        x = self.linear0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout(x)

        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FlowDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        radius: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.radius = radius

        self.mlp = Mlp(depth, depth // 2, 4, dropout=dropout)

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        flow0: torch.Tensor,
        size1: torch.Size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = flow0.device

        flow0_to_1 = self.mlp(flow0)
        scale = torch.tensor([size1[1], size1[0]], device=device)
        flow0_to_1[:, :, :2] = scale * flow0_to_1[:, :, :2].sigmoid()

        coor, log_std_mul_2 = flow0_to_1.detach()[:, :, None].chunk(2, dim=3)
        grid = K.create_meshgrid(
            size1[0], size1[1], normalized_coordinates=False, device=device)
        grid = grid.reshape(1, 1, -1, 2)
        span = self.radius * (0.5 * log_std_mul_2).exp()
        flow0_to_1_mask = ((grid[..., 0] > coor[..., 0] - span[..., 0]) &
                           (grid[..., 0] < coor[..., 0] + span[..., 0]) &
                           (grid[..., 1] > coor[..., 1] - span[..., 1]) &
                           (grid[..., 1] < coor[..., 1] + span[..., 1]))
        return flow0_to_1, flow0_to_1_mask
