from typing import List

import kornia as K
import torch
from torch import nn
from torch.nn import functional as F


class Backbone(nn.Module):
    def __init__(
        self,
        block_builders: List,
        block_counts: List[int],
        layer_depths: List[int],
        strides: List[int],
        use_fpn: bool = True,
        fpn_type: str = "loftr",
        coarse_scale: int = 16
    ) -> None:
        super().__init__()
        assert len(block_counts) == len(layer_depths) == len(strides)
        self.strides = strides

        if fpn_type not in ["loftr"]:
            raise ValueError("")
        self.fpn_type = fpn_type

        self.layers = nn.ModuleList()
        self.in_depth = 1
        idx = 0
        if isinstance(block_builders[0], nn.Module):
            assert block_counts[0] == 1
            self.layers.append(block_builders[0])
            self.in_depth = layer_depths[0]
            idx = 1
        for i in range(idx, len(layer_depths)):
            self.layers.append(self._make_layer(
                block_builders[i], layer_depths[i], block_counts[i],
                stride=strides[i]))

        if use_fpn:
            last_scale = cur_scale = 1
            fpn_layer_depths = []
            self.fpn_scale = []
            for i in range(len(strides)):
                cur_scale *= strides[i]
                if cur_scale > coarse_scale:
                    break
                if i == len(strides) - 1 or strides[i + 1] != 1:
                    fpn_layer_depths.append(layer_depths[i])
                    if cur_scale != 1:
                        self.fpn_scale.append(cur_scale // last_scale)
                    last_scale = cur_scale

            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            for i in range(len(fpn_layer_depths[:-1])):
                self.ups.append(self._create_up_branch(
                    fpn_layer_depths[i], fpn_layer_depths[i + 1]))
                self.downs.append(self._create_down_branch(
                    fpn_layer_depths[i + 1], fpn_layer_depths[i]))
            self.ups.append(self._create_up_branch(
                fpn_layer_depths[-1], fpn_layer_depths[-1]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(
        self,
        block_builder,
        out_depth: int,
        block_count: int,
        stride: int = 1
    ) -> nn.Module:
        strides = [stride] + (block_count - 1) * [1]
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block_builder(self.in_depth, out_depth, stride=stride))
            self.in_depth = out_depth
        return layer

    def _create_up_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        if self.fpn_type == "loftr":
            branch = nn.Conv2d(in_depth, out_depth, 1, bias=False)
        else:
            assert False
        return branch

    def _create_down_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        if self.fpn_type == "loftr":
            branch = nn.Sequential(
                nn.Conv2d(in_depth, in_depth, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_depth),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_depth, out_depth, 3, padding=1, bias=False))
        else:
            assert False
        return branch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        cur_scale = 1
        for i in range(len(self.strides)):
            x = self.layers[i](x)
            cur_scale *= self.strides[i]
            if i == len(self.strides) - 1 or self.strides[i + 1] != 1:
                out.append(x)
        return out

    def fuse(
        self,
        xs: List[torch.Tensor],
        align_corners: List[bool]
    ) -> List[torch.Tensor]:
        n = len(xs)
        out = (n - 1) * [None] + [self.ups[-1](xs[-1])]
        for i in reversed(range(n - 1)):
            out[i] = self.ups[i](xs[i])
            s = self.fpn_scale[i]
            if align_corners[i]:
                sh, sw = s * out[i + 1].shape[2], s * out[i + 1].shape[3]
                scale = xs[i].new_tensor([sw - s, sh - s])
                grid = K.create_meshgrid(
                    sh, sw, normalized_coordinates=False, device=xs[i].device)
                grid = (2 * grid / scale - 1).repeat(len(xs[i]), 1, 1, 1)
                out[i] += F.grid_sample(out[i + 1], grid, align_corners=True)
            else:
                out[i] += F.interpolate(
                    out[i + 1], scale_factor=s, mode="bilinear",
                    align_corners=False)
            out[i] = self.downs[i](out[i])
        return out
