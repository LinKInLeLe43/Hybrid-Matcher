import copy
from typing import List, Optional, Sequence, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class Mlp(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_dim, in_dim, bias=bias)
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.linear1(x)
        return x


class Mlp3x3(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()

        self.linear = nn.Linear(in_dim, in_dim, bias=bias)
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.gelu(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class LocalCluster(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        center_size: int,
        fold_size: int,
        bias: bool = True,
        type: str = "original",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.center_size = center_size
        self.fold_size = fold_size
        self.type = type

        self.proj = nn.Linear(dim, dim * 2, bias=bias)
        self.center_proposal = nn.AdaptiveMaxPool2d(center_size)
        self.merge = nn.Linear(dim, dim, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        fc, fh, fw = self.num_heads, self.fold_size, self.fold_size
        n, c, h, w = x.shape
        sh, sw = h // fh, w // fw
        m, s = n * fc * fh * fw, self.center_size**2
        device = x.device

        x = self.proj(x.permute(0, 2, 3, 1))
        x = einops.rearrange(
            x,
            "n (fh sh) (fw sw) (fc sc) -> (n fc fh fw) sc sh sw",
            fc=fc,
            fh=fh,
            fw=fw,
        )
        center = self.center_proposal(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        center = center.flatten(start_dim=2).transpose(1, 2)
        x_point, x_value = x.chunk(2, dim=2)
        center_point, center_value = center.chunk(2, dim=2)

        x_point = F.normalize(x_point, dim=2)
        center_point = F.normalize(center_point, dim=2)
        similarity = x_point @ center_point.transpose(-2, -1)
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = einops.repeat(
                mask,
                "n (fh sh) (fw sw) -> (n fc fh fw) (sh sw) s",
                fc=fc,
                fh=fh,
                fw=fw,
                s=s,
            )
            similarity.masked_fill_(~mask, float("-inf"))
        similarity.sigmoid_()
        max_sim_values, max_sim_idxes = similarity.max(dim=2)

        if self.type == "flattened_index":
            max_sim_idxes = (
                max_sim_idxes + s * torch.arange(m, device=device)[:, None]
            )
            max_sim_values, max_sim_idxes, x_value, center_value = [
                x.flatten(end_dim=1)
                for x in [max_sim_values, max_sim_idxes, x_value, center_value]
            ]

            cat_ones = torch.ones_like(x_value[:, [0]])
            cat_x_value = torch.cat([x_value, cat_ones], dim=1)
            cat_ones = torch.ones_like(center_value[:, [0]])
            cat_center_value = torch.cat([center_value, cat_ones], dim=1)
            aggregated = cat_center_value.index_add_(
                0, max_sim_idxes, max_sim_values[:, None] * cat_x_value
            )
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = max_sim_values[:, None] * aggregated.index_select(
                0, max_sim_idxes
            )
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fh sh) (fw sw) (fc sc)",
                fc=fc,
                fh=fh,
                fw=fw,
                sh=sh,
                sw=sw,
            )
        elif self.type == "original":
            mask = torch.zeros_like(similarity)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarity = (mask * similarity)[..., None]

            aggregated = center_value + (
                similarity * x_value[:, :, None, :]
            ).sum(dim=1)
            aggregated /= 1 + similarity.sum(dim=1)
            dispatched = (similarity * aggregated[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw) (sh sw) sc -> n (fh sh) (fw sw) (fc sc)",
                fc=fc,
                fh=fh,
                fw=fw,
                sh=sh,
                sw=sw,
            )
        else:
            raise NotImplementedError("")
        dispatched = self.merge(dispatched)
        return dispatched


class GlobalCluster(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        type: str = "flattened_index",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.type = type

        self.proj0 = nn.Linear(dim, dim, bias=bias)
        self.proj1 = nn.Linear(dim, dim * 2, bias=bias)
        self.merge = nn.Linear(dim, dim, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self, x0: Tensor, center1: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        fc = self.num_heads
        n, c, h0, w0 = x0.shape
        _, _, h1, w1 = center1.shape
        m, l, s = n * fc, h0 * w0, h1 * w1
        device = x0.device

        x0_point = self.proj0(x0.permute(0, 2, 3, 1))
        center1 = self.proj1(center1.permute(0, 2, 3, 1))
        x0_point = (
            x0_point.view(n, -1, self.num_heads, self.head_dim)
            .transpose(-3, -2)
            .flatten(end_dim=1)
        )
        center1 = (
            center1.view(n, -1, self.num_heads, self.head_dim)
            .transpose(-3, -2)
            .flatten(end_dim=1)
        )
        center1_point, center1_value = center1.chunk(2, dim=2)

        x0_point = F.normalize(x0_point, dim=2)
        center1_point = F.normalize(center1_point, dim=2)
        similarity = x0_point @ center1_point.transpose(-2, -1)
        similarity = self.alpha * similarity + self.beta
        if mask is not None:
            mask = einops.repeat(mask, "n l s -> (n fc) l s", fc=fc)
            similarity.masked_fill_(~mask, float("-inf"))
        similarity.sigmoid_()

        if self.type == "flattened_index":
            max_sim_values, max_sim_idxes = similarity.max(dim=2)
            max_sim_idxes = (
                max_sim_idxes + s * torch.arange(m, device=device)[:, None]
            )
            max_sim_values, max_sim_idxes, center1_value = [
                x.flatten(end_dim=1)
                for x in [max_sim_values, max_sim_idxes, center1_value]
            ]

            dispatched = max_sim_values[:, None] * center1_value.index_select(
                0, max_sim_idxes
            )
            dispatched = einops.rearrange(
                dispatched, "(n fc h w) sc -> n h w (fc sc)", fc=fc, h=h0, w=w0
            )
        elif self.type == "original":
            max_sim_idxes = similarity.argmax(dim=2)
            mask = torch.zeros_like(similarity)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarity = (mask * similarity)[..., None]

            dispatched = (similarity * center1_value[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched,
                "(n fc) (h w) sc -> n h w (fc sc)",
                fc=fc,
                h=h0,
                w=w0,
            )
        else:
            raise NotImplementedError("")
        dispatched = self.merge(dispatched)
        return dispatched


class LocalClusterBlock(Module):
    def __init__(
        self,
        in_depth: int,
        num_heads: int,
        center_size: int,
        fold_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.cluster = LocalCluster(
            in_depth, num_heads, center_size, fold_size, bias=bias
        )
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = Mlp(2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        new_x = self.cluster(x, mask=mask)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x.permute(0, 2, 3, 1), new_x], dim=3)
        new_x = self.mlp(new_x)
        new_x = self.norm1(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()

        new_x += x
        return new_x


class GlobalClusterBlock(Module):
    def __init__(
        self, in_depth: int, num_heads: int, bias: bool = True
    ) -> None:
        super().__init__()

        self.cluster = GlobalCluster(in_depth, num_heads, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = Mlp3x3(2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self, x0: Tensor, center1: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor]:
        new_x0 = self.cluster(x0, center1, mask=mask)
        new_x0 = self.norm0(new_x0)

        new_x0 = torch.cat([x0.permute(0, 2, 3, 1), new_x0], dim=3)
        new_x0 = self.mlp3x3(new_x0)
        new_x0 = self.norm1(new_x0)
        new_x0 = new_x0.permute(0, 3, 1, 2).contiguous()

        new_x0 += x0
        return new_x0


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

            layer = nn.Sequential()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i],
                    num_heads_list[i],
                    center_sizes[i],
                    fold_sizes[i],
                    bias=bias,
                )
                layer.append(block)
            self.layers.append(layer)

            initial_depth = layer_depths[i]

        # TODO: check FPN design
        # self.layer1_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[0]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False))
        # self.layer0_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[0]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False))
        # self.layer2_up = nn.Conv2d(
        #     layer_depths[2], layer_depths[2], 1, bias=False)
        # self.layer1_up = nn.Conv2d(
        #     layer_depths[1], layer_depths[2], 1, bias=False)
        # self.layer1_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[2], layer_depths[2], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[2]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[2], layer_depths[1], 3, padding=1, bias=False))
        # self.layer0_up = nn.Conv2d(
        #     layer_depths[0], layer_depths[1], 1, bias=False)
        # self.layer0_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[1], layer_depths[1], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[1]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[1], layer_depths[0], 3, padding=1, bias=False))

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        outs = []
        for point_reducer, layer in zip(self.point_reducers, self.layers):
            x = point_reducer(x)
            x = layer(x)
            outs.append(x)
        return outs[0], outs[-1]

        # x1 = x1 + F.interpolate(
        #     x2, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x1 = self.layer1_out(x1)
        # x0 = x0 + F.interpolate(
        #     x1, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x0 = self.layer0_out(x0)
        # return x2, x0
        # new_x2 = self.layer2_up(x2)
        # new_x1 = self.layer1_up(x1)
        # new_x1 += F.interpolate(
        #     new_x2, scale_factor=2.0, mode="bilinear", align_corners=True)
        # new_x1 = self.layer1_out(new_x1)
        # new_x0 = self.layer0_up(x0)
        # new_x0 += F.interpolate(
        #     new_x1, scale_factor=2.0, mode="bilinear", align_corners=True)
        # new_x0 = self.layer0_out(new_x0)
        # return new_x2, new_x0


class MergeBlock(Module):
    def __init__(self, scale: int, depth: int, bias: bool = True) -> None:
        super().__init__()
        self.scale = scale

        self.mlp = Mlp(2 * depth, depth, bias=bias)
        self.norm = nn.LayerNorm(depth)
        self.pooling = nn.MaxPool2d(scale, stride=scale)

    def forward(self, x: Tensor, center: Tensor) -> Tuple[Tensor, Tensor]:
        up_center = F.interpolate(
            center, scale_factor=self.scale, mode="bilinear"
        )
        new_x = torch.cat([x, up_center], dim=1)
        new_x = new_x.permute(0, 2, 3, 1)
        new_x = self.mlp(new_x)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()
        new_center = self.pooling(new_x)
        new_x = x + new_x
        new_center = center + new_center
        return new_x, new_center


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

        global_block = GlobalClusterBlock(in_depth, num_heads, bias=bias)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in range(layer_count)]
        )

        self.self_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(layer_count)]
        )
        self.cross_blocks = nn.ModuleList(
            [copy.deepcopy(attention_block) for _ in range(layer_count)]
        )

        # TODO: check weight init
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
        xy_mask01 = xy_mask10 = yy_mask00 = yy_mask11 = yy_mask01 = (
            yy_mask10
        ) = None
        if mask0 is not None and mask1 is not None:
            x0_mask = F.max_pool2d(mask0.float(), 2).bool()
            x1_mask = F.max_pool2d(mask1.float(), 2).bool()
            y0_mask = F.max_pool2d(mask0.float(), 4).bool()
            y1_mask = F.max_pool2d(mask1.float(), 4).bool()

            n = mask0.shape[0]
            xy_mask01 = x0_mask.view(n, -1, 1) & y1_mask.view(n, 1, -1)
            xy_mask10 = x1_mask.view(n, -1, 1) & y0_mask.view(n, 1, -1)
            yy_mask00 = y0_mask.view(n, 1, -1, 1) & y0_mask.view(n, 1, 1, -1)
            yy_mask11 = y1_mask.view(n, 1, -1, 1) & y1_mask.view(n, 1, 1, -1)
            yy_mask01 = y0_mask.view(n, 1, -1, 1) & y1_mask.view(n, 1, 1, -1)
            yy_mask10 = yy_mask01.transpose(-1, -2)

        x0_16x, x0_32x = x0_list
        x1_16x, x1_32x = x1_list
        for merge_block, global_block, self_block, cross_block in zip(
            self.merge_blocks,
            self.global_blocks,
            self.self_blocks,
            self.cross_blocks,
        ):
            x0_16x, x0_32x = merge_block(x0_16x, x0_32x)
            x1_16x, x1_32x = merge_block(x1_16x, x1_32x)
            x0_16x = global_block(x0_16x, x1_32x, mask=xy_mask01)
            x1_16x = global_block(x1_16x, x0_32x, mask=xy_mask10)
            x0_16x = self_block(x0_16x, x0_16x, rope=rope, mask=yy_mask00)
            x1_16x = self_block(x1_16x, x1_16x, rope=rope, mask=yy_mask11)
            x0_16x = cross_block(x0_16x, x1_16x, mask=yy_mask01)
            x1_16x = cross_block(x1_16x, x0_16x, mask=yy_mask10)
        return x0_16x, x1_16x
