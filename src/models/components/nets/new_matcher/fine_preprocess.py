from typing import Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F


class FinePreprocess(nn.Module):
    def __init__(
        self,
        coarse_depth: int,
        fine_depth: int,
        window_size: int = 5,
        concat_coarse: bool = True
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.concat_coarse = concat_coarse
        if concat_coarse:
            self.proj = nn.Linear(coarse_depth, fine_depth)
            self.merge = nn.Linear(2 * fine_depth, fine_depth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        coarse_feature0: torch.Tensor,
        coarse_feature1: torch.Tensor,
        fine_feature0: torch.Tensor,
        fine_feature1: torch.Tensor,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stride: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idxes, i_idxes, j_idxes = idxes
        w, ww = self.window_size, self.window_size ** 2
        if len(b_idxes) == 0:
            fine_feature0 = fine_feature1 = fine_feature0.new_empty(
                (0, ww, fine_feature0.shape[1]))
            return fine_feature0, fine_feature1

        fine_feature0 = einops.rearrange(
            F.unfold(fine_feature0, w, padding=w // 2, stride=stride),
            "n (c ww) l -> n l ww c", ww=ww)[b_idxes, i_idxes]
        fine_feature1 = einops.rearrange(
            F.unfold(fine_feature1, w, padding=w // 2, stride=stride),
            "n (c ww) l -> n l ww c", ww=ww)[b_idxes, j_idxes]

        if self.concat_coarse:
            coarse_feature = torch.cat([coarse_feature0[b_idxes, i_idxes],
                                        coarse_feature1[b_idxes, j_idxes]])
            coarse_feature = einops.repeat(
                self.proj(coarse_feature), "n c -> n ww c", ww=ww)
            fine_feature = torch.cat([fine_feature0, fine_feature1])
            feature = torch.cat([fine_feature, coarse_feature], dim=2)
            fine_feature0, fine_feature1 = self.merge(feature).chunk(2)
        return fine_feature0, fine_feature1
