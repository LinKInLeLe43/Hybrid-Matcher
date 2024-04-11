from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _conv1x1(in_depth: int, out_depth: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_depth, out_depth, 1, stride=stride, bias=False)


def _conv3x3(in_depth: int, out_depth: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_depth, out_depth, 3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, in_depth: int, out_depth: int, stride: int = 1) -> None:
        super().__init__()
        self.conv0 = _conv3x3(in_depth, out_depth, stride=stride)
        self.norm0 = nn.BatchNorm2d(out_depth)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = _conv3x3(out_depth, out_depth)
        self.norm1 = nn.BatchNorm2d(out_depth)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                _conv1x1(in_depth, out_depth, stride=stride),
                nn.BatchNorm2d(out_depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.norm1(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNetFpn82(nn.Module):
    def __init__(
        self,
        initial_depth: int,
        layer_depths: Tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.in_depth = initial_depth
        self.scales = 8, 2

        self.conv = nn.Conv2d(
            1, initial_depth, 7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(initial_depth)
        self.relu = nn.ReLU(inplace=True)

        self.layer0 = self._make_layer(layer_depths[0])
        self.layer1 = self._make_layer(layer_depths[1], stride=2)
        self.layer2 = self._make_layer(layer_depths[2], stride=2)

        # self.layer2_up = _conv1x1(layer_depths[2], layer_depths[2])
        # self.layer1_up = _conv1x1(layer_depths[1], layer_depths[2])
        # self.layer1_out = nn.Sequential(
        #     _conv3x3(layer_depths[2], layer_depths[2]),
        #     nn.BatchNorm2d(layer_depths[2]),
        #     nn.LeakyReLU(inplace=True),
        #     _conv3x3(layer_depths[2], layer_depths[1]))
        # self.layer0_up = _conv1x1(layer_depths[0], layer_depths[1])
        # self.layer0_out = nn.Sequential(
        #     _conv3x3(layer_depths[1], layer_depths[1]),
        #     nn.BatchNorm2d(layer_depths[1]),
        #     nn.LeakyReLU(inplace=True),
        #     _conv3x3(layer_depths[1], layer_depths[0]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, depth: int, stride: int = 1) -> nn.Module:
        layer = nn.Sequential(
            _BasicBlock(self.in_depth, depth, stride=stride),
            _BasicBlock(depth, depth))
        self.in_depth = depth
        return layer

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)

        # x2_out = self.layer2_up(x2)
        # x1_out = self.layer1_up(x1)
        # x1_out += F.interpolate(
        #     x2_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x1_out = self.layer1_out(x1_out)
        # x0_out = self.layer0_up(x0)
        # x0_out += F.interpolate(
        #     x1_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x0_out = self.layer0_out(x0_out)
        # return x2_out, x0_out
        return [x0, x1], x2
