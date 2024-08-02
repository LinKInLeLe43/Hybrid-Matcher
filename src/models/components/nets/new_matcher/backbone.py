from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(layer_depths[0])
        self.layer1 = self._make_layer(layer_depths[1], stride=2)
        self.layer2 = self._make_layer(layer_depths[2], stride=2)
        self.layer3 = self._make_layer(layer_depths[3], stride=2)

        self._conv = nn.Conv2d(
            1, 128, 7, stride=2, padding=3, bias=False)
        self._norm = nn.BatchNorm2d(128)
        self.in_depth = 128
        self._layer0 = self._make_layer(128)
        self._layer1 = self._make_layer(128, stride=2)
        self._layer2 = self._make_layer(128, stride=2)

        self.layer3_up = _conv1x1(layer_depths[3], 256)

        self.layer1_up = _conv1x1(layer_depths[1] + 128, 256)
        self.layer1_out = nn.Sequential(
            _conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            _conv3x3(256, 256))

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        y = self._conv(y)
        y = self._norm(y)
        y = self.relu(y)

        y0 = self._layer0(y)
        y1 = self._layer1(y0)
        y2 = self._layer2(y1)

        x3_out = self.layer3_up(x3)
        x1_out = self.layer1_up(torch.cat([x1, y2], dim=1))
        x1_out += F.interpolate(x2, scale_factor=2.0, mode="bilinear")
        x1_out = self.layer1_out(x1_out)

        return y0, x1_out, x3_out


class PretrainedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = 8, 2

        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad = False

        self._conv = nn.Conv2d(
            1, 128, 7, stride=2, padding=3, bias=False)
        self._norm = nn.BatchNorm2d(128)
        self._relu = nn.ReLU(inplace=True)
        self.in_depth = 128
        self._layer0 = self._make_layer(128)
        self._layer1 = self._make_layer(128, stride=2)
        self._layer2 = self._make_layer(128, stride=2)

        self.layer4_up = _conv1x1(512, 256)

        self.layer2_up = _conv1x1(256, 256)
        self.layer2_out = nn.Sequential(
            _conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            _conv3x3(256, 256))

    def _make_layer(self, depth: int, stride: int = 1) -> nn.Module:
        layer = nn.Sequential(
            _BasicBlock(self.in_depth, depth, stride=stride),
            _BasicBlock(depth, depth))
        self.in_depth = depth
        return layer

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = x.mean(dim=1, keepdim=True)
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)

        y = self._conv(y)
        y = self._norm(y)
        y = self._relu(y)

        y0 = self._layer0(y)
        y1 = self._layer1(y0)
        y2 = self._layer2(y1)

        x4_out = self.layer4_up(x4)
        x2_out = self.layer2_up(torch.cat([x2, y2], dim=1))
        x2_out += F.interpolate(x3, scale_factor=2.0, mode="bilinear")
        x2_out = self.layer2_out(x2_out)

        return y0, x2_out, x4_out
