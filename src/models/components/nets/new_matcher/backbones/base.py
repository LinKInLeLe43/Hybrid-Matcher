from typing import List, Optional

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
        layer_depths0: List[Optional[int]],
        layer_depths1: Optional[List[Optional[int]]] = None,
        type: str = "loftr"
    ) -> None:
        super().__init__()
        self.type = type

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        if type == "loftr":
            for i in range(len(layer_depths0[:-1])):
                up = None
                if layer_depths0[i] is not None:
                    up = self._create_up_branch(
                        layer_depths0[i], layer_depths0[i + 1])
                self.ups.append(up)
                down_out_depth = layer_depths0[i]
                if down_out_depth is None:
                    for depth in layer_depths0[i + 1:]:
                        if depth is not None:
                            down_out_depth = depth
                            break
                down_in_depth = layer_depths0[i + 1]
                if down_in_depth is None:
                    down_in_depth = down_out_depth
                if layer_depths1 is not None and layer_depths1[i] is not None:
                    down_in_depth = down_in_depth + layer_depths1[i]
                self.downs.append(self._create_down_branch(
                    down_in_depth, down_out_depth))
            self.ups.append(self._create_up_branch(
                layer_depths0[-1], layer_depths0[-1]))
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
        xs0: List[Optional[torch.Tensor]],
        xs1: Optional[List[Optional[torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        n = len(xs0)
        out = (n - 1) * [None] + [self.ups[-1](xs0[-1])]
        for i in reversed(range(n - 1)):
            out[i] = F.interpolate(
                out[i + 1], scale_factor=2, mode="bilinear",
                align_corners=False)
            if xs0[i] is not None:
                out[i] = out[i] + self.ups[i](xs0[i])
            if xs1 is not None and xs1[i] is not None:
                out[i] = torch.cat([out[i], xs1[i]], dim=1)
            out[i] = self.downs[i](out[i])
        return out
