import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class Mlp(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear0 = nn.Linear(in_dim, in_dim, bias=bias)
        self.act = nn.GELU()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(self.act(self.linear0(x)))
        return x


class Mlp3x3(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=bias)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.linear(x)).permute(0, 3, 1, 2)
        x = self.conv(x).permute(0, 2, 3, 1)
        return x


class SelfContextCluster(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_anchors: int,
        num_folds: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_folds = num_folds

        self.proj = nn.Linear(dim, dim * 2, bias=bias)
        self.anchor_proposal = nn.AdaptiveMaxPool2d(num_anchors)
        self.merge = nn.Linear(dim, dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        n, h, w, c = x.shape
        fh = fw = self.num_folds
        sh, sw = h // fh, w // fw
        m = n * self.num_heads * fh * fw

        x0 = (
            self.proj(x)
            .view(n, fh, sh, fw, sw, self.num_heads, self.head_dim * 2)
            .permute(0, 5, 1, 3, 6, 2, 4)
            .flatten(end_dim=3)
        )
        x1 = self.anchor_proposal(x0)
        x0_point, x0_value = (
            x0.flatten(start_dim=-2).transpose(-2, -1).chunk(2, dim=-1)
        )
        x1_point, x1_value = (
            x1.flatten(start_dim=-2).transpose(-2, -1).chunk(2, dim=-1)
        )
        x0_point = F.normalize(x0_point, dim=-1)
        x1_point = F.normalize(x1_point, dim=-1)
        similarity = x0_point @ x1_point.transpose(-2, -1)
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = (
                mask.view(n, 1, fh, sh, fw, sw)
                .expand(-1, self.num_heads, -1, -1, -1, -1)
                .transpose(-3, -2)
                .contiguous()
                .view(m, h * w, 1)
            )
            similarity.masked_fill_(~mask, float("-inf"))
        similarity = similarity.sigmoid()

        max_sim_idxes = similarity.argmax(dim=2, keepdim=True)
        mask = torch.zeros_like(similarity).scatter_(2, max_sim_idxes, 1.0)
        similarity = (mask * similarity)[..., None]
        aggregated = x1_value[:, None, :, :] + (
            x0_value[:, :, None, :] * similarity
        ).sum(dim=1, keepdim=True)
        aggregated = aggregated / (1 + similarity.sum(dim=1, keepdim=True))
        dispatched = (similarity * aggregated).sum(dim=2)
        dispatched = (
            dispatched.view(n, self.num_heads, fh, fw, sh, sw, self.head_dim)
            .permute(0, 2, 4, 3, 5, 1, 6)
            .contiguous()
            .view(n, h, w, c)
        )
        dispatched = self.merge(dispatched)
        return dispatched


class CrossContextCluster(Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, "`dim` should be divisible by `num_heads`."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj0 = nn.Linear(dim, dim, bias=bias)
        self.proj1 = nn.Linear(dim, dim * 2, bias=bias)
        self.merge = nn.Linear(dim, dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        n, h, w, c = x0.shape
        m = n * self.num_heads

        x0_point = (
            self.proj0(x0)
            .view(n, h * w, self.num_heads, self.head_dim)
            .transpose(-3, -2)
            .flatten(end_dim=1)
        )
        x1_point, x1_value = (
            self.proj1(x1)
            .view(n, -1, self.num_heads, self.head_dim * 2)
            .transpose(-3, -2)
            .flatten(end_dim=1)
            .chunk(2, dim=-1)
        )
        x0_point = F.normalize(x0_point, dim=-1)
        x1_point = F.normalize(x1_point, dim=-1)
        similarity = x0_point @ x1_point.transpose(-2, -1)
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = (
                mask.view(n, 1, -1)
                .expand(-1, self.num_heads, -1)
                .view(m, 1, -1)
            )
            similarity.masked_fill_(~mask, float("-inf"))
        similarity = similarity.sigmoid()

        m_range = torch.arange(m, device=x0.device)[:, None]
        max_sim, indices = similarity.max(dim=-1, keepdim=True)
        dispatched = max_sim * x1_value[m_range, indices[..., 0]]
        dispatched = (
            dispatched.view(n, self.num_heads, -1, self.head_dim)
            .transpose(-3, -2)
            .contiguous()
            .view(n, h, w, c)
        )
        dispatched = self.merge(dispatched)
        return dispatched


class SelfCoCBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_anchors: int,
        num_folds: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.cluster = SelfContextCluster(
            dim, num_heads, num_anchors, num_folds, bias=bias
        )
        self.norm0 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim * 2, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        message = self.norm0(self.cluster(x, mask=mask))
        message = torch.cat([x, message], dim=-1)
        x = x + self.norm1(self.mlp(message))
        return x


class CrossCoCBlock(Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.cluster = CrossContextCluster(dim, num_heads, bias=bias)
        self.norm0 = nn.LayerNorm(dim)
        self.mlp3x3 = Mlp3x3(dim * 2, dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)

    def forward(
        self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor]:
        message = self.norm0(self.cluster(x0, x1, mask=mask))
        message = torch.cat([x0, message], dim=-1)
        x0 = x0 + self.norm1(self.mlp3x3(message))
        return x0


class MergeBlock(Module):
    def __init__(self, stride: int, dim: int, bias: bool = True) -> None:
        super().__init__()
        self.stride = stride

        self.mlp = Mlp(dim * 2, dim, bias=bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        message = x1.permute(0, 3, 1, 2).contiguous()
        message = F.interpolate(
            message,
            scale_factor=self.stride,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        message = torch.cat([x0, message], dim=-1)
        message = self.norm(self.mlp(message))
        x0 = x0 + message
        message = message.permute(0, 3, 1, 2)
        x1 = x1 + F.max_pool2d(message, self.stride).permute(0, 2, 3, 1)
        return x0, x1


class LocalCoC(Module):
    def __init__(
        self,
        initial_depth: int,
        scales: List[int],
        blocks_counts: List[int],
        layer_depths: List[int],
        num_heads_list: List[int],
        center_sizes: List[int],
        fold_sizes: List[int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.scales = scales

        self.point_reducers, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(scales)):
            if scales[i] > 1:
                point_reducer = nn.Conv2d(
                    initial_depth,
                    layer_depths[i],
                    scales[i] + 1,
                    stride=scales[i],
                    padding=1,
                )
            else:
                point_reducer = nn.Identity()
            self.point_reducers.append(point_reducer)

            layer = nn.ModuleList()
            for _ in range(blocks_counts[i]):
                block = SelfCoCBlock(
                    layer_depths[i],
                    num_heads_list[i],
                    center_sizes[i],
                    fold_sizes[i],
                    bias=bias,
                )
                layer.append(block)
            self.layers.append(layer)

            initial_depth = layer_depths[i]

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        scale = 1
        outs = []
        for i in range(len(self.scales)):
            mask_ = None
            if mask is not None:
                scale = scale * self.scales[i]
                mask_ = F.max_pool2d(mask.float(), scale).bool()
            x = self.point_reducers[i](x).permute(0, 2, 3, 1)
            for block in self.layers[i]:
                x = block(x, mask_)
            x = x.permute(0, 3, 1, 2)
            outs.append(x)
        return outs


class GlobalCoC(Module):
    def __init__(
        self,
        stride: int,
        in_depth: int,
        num_heads: int,
        layer_count: int,
        attention_block: Module,
        bias: bool = True,
    ) -> None:
        super().__init__()

        merge_block = MergeBlock(stride, in_depth, bias=bias)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in range(layer_count)]
        )
        global_block = CrossCoCBlock(in_depth, num_heads, bias=bias)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in range(layer_count)]
        )
        self.self_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(layer_count)]
        )
        self.cross_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(layer_count)]
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x0_list: Sequence[Tensor],
        x1_list: Sequence[Tensor],
        rope: Optional[Tensor] = None,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        mask00 = mask11 = mask01 = mask10 = None
        if mask0 is not None and mask1 is not None:
            n = mask0.shape[0]
            mask0 = F.max_pool2d(mask0.float(), 4).bool()
            mask1 = F.max_pool2d(mask1.float(), 4).bool()

            mask00 = mask0.view(n, 1, -1, 1) & mask0.view(n, 1, 1, -1)
            mask11 = mask1.view(n, 1, -1, 1) & mask1.view(n, 1, 1, -1)
            mask01 = mask0.view(n, 1, -1, 1) & mask1.view(n, 1, 1, -1)
            mask10 = mask01.transpose(-1, -2)

        x0_16x, x0_32x, x1_16x, x1_32x = [
            t.permute(0, 2, 3, 1) for t in [*x0_list, *x1_list]
        ]
        for merge_block, global_block, self_block, cross_block in zip(
            self.merge_blocks,
            self.global_blocks,
            self.self_blocks,
            self.cross_blocks,
        ):
            x0_16x, x0_32x = merge_block(x0_16x, x0_32x)
            x1_16x, x1_32x = merge_block(x1_16x, x1_32x)
            x0_16x = global_block(x0_16x, x1_32x, mask=mask1)
            x1_16x = global_block(x1_16x, x0_32x, mask=mask0)
            x0_16x = self_block(x0_16x, x0_16x, rope=rope, mask=mask00)
            x1_16x = self_block(x1_16x, x1_16x, rope=rope, mask=mask11)
            x0_16x = cross_block(x0_16x, x1_16x, mask=mask01)
            x1_16x = cross_block(x1_16x, x0_16x, mask=mask10)
        x0_16x, x1_16x = x0_16x.permute(0, 3, 1, 2), x1_16x.permute(0, 3, 1, 2)
        return x0_16x, x1_16x
