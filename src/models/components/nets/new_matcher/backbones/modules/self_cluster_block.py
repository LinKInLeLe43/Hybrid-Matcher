from typing import Optional

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
        center_size: int,
        fold_size: int,
        bias: bool = True,
        enable_efficient: bool = True
    ) -> None:
        super().__init__()
        self.head_count = head_count
        self.center_size = center_size
        self.fold_size = fold_size
        self.enable_efficient = enable_efficient

        self.proj = nn.Conv2d(depth, 2 * depth, 1, bias=bias)
        self.center_proposal = nn.AdaptiveAvgPool2d(center_size)
        self.merge = nn.Conv2d(depth, depth, 1, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc, fh, fw = self.head_count, self.fold_size, self.fold_size
        n, c, h, w = x.shape
        sh, sw = h // fh, w // fw
        m, l, s = n * fc * fh * fw, sh * sw, self.center_size ** 2
        device = x.device

        x = self.proj(x)
        x = einops.rearrange(
            x, "n (fc sc) (fh sh) (fw sw) -> (n fc fh fw) sc sh sw", fc=fc,
            fh=fh, fw=fw)
        center = self.center_proposal(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        center = center.flatten(start_dim=2).transpose(1, 2)
        x_point, x_value = x.chunk(2, dim=2)
        center_point, center_value = center.chunk(2, dim=2)

        norm_x_point = F.normalize(x_point, dim=2)
        norm_center_point = F.normalize(center_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x_point, norm_center_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.repeat(
                mask, "n (fh sh) (fw sw) -> (n fc fh fw) (sh sw) s", fc=fc,
                fh=fh, fw=fw, s=s)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.enable_efficient:
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, x_value, center_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, x_value, center_value))

            cat_ones = torch.ones_like(x_value[:, [0]])
            cat_x_value = torch.cat([x_value, cat_ones], dim=1)
            cat_ones = torch.ones_like(center_value[:, [0]])
            cat_center_value = torch.cat([center_value, cat_ones], dim=1)
            aggregated = cat_center_value.index_add_(
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

            aggregated = (center_value +
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
        stride: int,
        head_count: int,
        center_size: int,
        fold_size: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.point_reducer = nn.Identity()
        if out_depth != in_depth or stride != 1:
            assert stride == 2
            kernel_size = 3
            self.point_reducer = nn.Conv2d(
                in_depth, out_depth, kernel_size, stride=stride,
                padding=kernel_size // 2)

        self.cluster = SelfCluster(
            out_depth, head_count, center_size, fold_size, bias=bias)
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
