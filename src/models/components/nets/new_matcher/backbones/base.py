from typing import List, Optional

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
        strides: List[int]
    ) -> None:
        super().__init__()
        assert (len(block_builders) == len(block_counts) == len(layer_depths) ==
                len(strides))
        self.strides = strides

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        cur_scale = 1
        for i in range(len(self.strides)):
            x = self.layers[i](x)
            cur_scale *= self.strides[i]
            if i == len(self.strides) - 1 or self.strides[i + 1] != 1:
                out.append(x)
        return out


class Fusion(nn.Module):
    def __init__(
        self,
        layer_depths: List[Optional[int]],
        type: str = "loftr"
    ) -> None:
        super().__init__()
        self.type = type

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        if type == "loftr":
            for i in range(len(layer_depths[:-1])):
                self.ups.append(self._create_up_branch(
                    layer_depths[i], layer_depths[i + 1]))
                self.downs.append(self._create_down_branch(
                    layer_depths[i + 1], layer_depths[i]))
            self.ups.append(self._create_up_branch(
                layer_depths[-1], layer_depths[-1]))
        else:
            raise ValueError("")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _create_up_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        if self.type == "loftr":
            branch = nn.Conv2d(in_depth, out_depth, 1, bias=False)
        else:
            assert False
        return branch

    def _create_down_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        if self.type == "loftr":
            branch = nn.Sequential(
                nn.Conv2d(in_depth, in_depth, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_depth),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_depth, out_depth, 3, padding=1, bias=False))
        else:
            assert False
        return branch

    def forward(
        self,
        xs: List[torch.Tensor],
        aligns: List[bool]
    ) -> List[torch.Tensor]:
        l = len(xs)
        out = (l - 1) * [None] + [self.ups[-1](xs[-1])]
        for i in reversed(range(l - 1)):
            out[i] = self.ups[i](xs[i])

            if aligns[i]:
                n, _, h, w = out[i + 1].shape
                grid = K.create_meshgrid(
                    h, w, normalized_coordinates=False,
                    device=out[i + 1].device)
                scale = out[i + 1].new_tensor([w - 2, h - 2])
                grid = (2 * grid / scale - 1).expand(n, -1, -1, -1)
                out[i] += F.grid_sample(out[i + 1], grid, align_corners=True)
            else:
                n, _, h, w = out[i].shape
                grid = K.create_meshgrid(
                    h, w, normalized_coordinates=False, device=out[i].device)
                scale = out[i].new_tensor([w - 1, h - 1])
                grid = (2 * (grid - 0.5) / scale - 1).expand(n, -1, -1, -1)
                out[i] = F.grid_sample(out[i], grid, align_corners=True)

                out[i] += F.interpolate(
                    out[i + 1], scale_factor=2, mode="bilinear",
                    align_corners=False)

            out[i] = self.downs[i](out[i])
        return out
