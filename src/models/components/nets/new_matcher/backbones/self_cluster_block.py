from typing import Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.nets.new_matcher.modules.mlp import Mlp


class SelfCluster(nn.Module):
    def __init__(
        self,
        depth: int,
        head_count: int,
        fold_size: Tuple[int, int],
        anchor_size: Tuple[int, int],
        bias: bool = True,
        enable_efficient: bool = True
    ) -> None:
        super().__init__()
        self.head_count = head_count
        self.fold_size = fold_size
        self.anchor_size = anchor_size
        self.enable_efficient = enable_efficient

        self.proj = nn.Conv2d(depth, 2 * depth, 1, bias=bias)
        self.anchor_proposal = nn.AdaptiveAvgPool2d(anchor_size)
        self.merge = nn.Conv2d(depth, depth, 1, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc, fh, fw = self.head_count, self.fold_size[0], self.fold_size[1]
        n, c, h, w = x.shape
        m = n * fc * fh * fw
        sh, sw = h // fh, w // fw
        l, s = sh * sw, self.anchor_size[0] * self.anchor_size[1]

        x = self.proj(x)
        x = einops.rearrange(
            x, "n (fc sc) (fh sh) (fw sw) -> (n fc fh fw) sc sh sw", fc=fc,
            fh=fh, fw=fw)
        anchor = self.anchor_proposal(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        anchor = anchor.flatten(start_dim=2).transpose(1, 2)
        x_point, x_value = x.chunk(2, dim=2)
        anchor_point, anchor_value = anchor.chunk(2, dim=2)

        norm_x_point = F.normalize(x_point, dim=2)
        norm_anchor_point = F.normalize(anchor_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x_point, norm_anchor_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.rearrange(
                mask, "n (fh sh) (fw sw) -> (n fc fh fw) (sh sw)", fc=fc, fh=fh,
                fw=fw)
            similarities.masked_fill_(~mask[:, :, None], float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.enable_efficient:
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=x.device)[:, None])
            max_sim_values, max_sim_idxes, x_value, anchor_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, x_value, anchor_value))

            cat_x_value = torch.cat(
                [x_value, torch.ones_like(x_value[:, [0]])], dim=1)
            cat_anchor_value = torch.cat(
                [anchor_value, torch.ones_like(anchor_value[:, [0]])], dim=1)
            aggregated = cat_anchor_value.index_add_(
                0, max_sim_idxes, max_sim_values[:, None] * cat_x_value)
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = (max_sim_values[:, None] *
                          aggregated.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        else:
            mask = torch.zeros_like(similarities)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarities = (mask * similarities)[..., None]

            aggregated = (anchor_value +
                          (similarities * x_value[:, :, None, :]).sum(dim=1))
            aggregated /= 1 + similarities.sum(dim=1)
            dispatched = (similarities * aggregated[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw) (sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        dispatched = self.merge(dispatched)
        return dispatched


class SelfClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        out_depth: int,
        head_count: int,
        kernel_size: int,
        fold_size: Tuple[int, int],
        anchor_size: Tuple[int, int],
        stride: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.point_reducer = nn.Identity()
        if out_depth != in_depth or stride != 1:
            self.point_reducer = nn.Conv2d(
                in_depth, out_depth, kernel_size, stride=stride,
                padding=kernel_size // 2, bias=bias)

        self.cluster = SelfCluster(
            out_depth, head_count, fold_size, anchor_size, bias=bias)
        self.norm0 = nn.GroupNorm(1, out_depth)

        self.mlp = Mlp(2 * out_depth, 2 * out_depth, out_depth, bias=bias)
        self.norm1 = nn.GroupNorm(1, out_depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.point_reducer(x)

        message = self.cluster(x, mask=mask)
        message = self.norm0(message)

        message = torch.cat([x, message], dim=1)
        message = self.mlp(message)
        message = self.norm1(message)

        out = x + message
        return out
