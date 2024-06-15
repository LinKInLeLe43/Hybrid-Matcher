import torch
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_depth: int, out_depth: int, stride: int = 1) -> None:
        super().__init__()

        self.conv0 = self._create_conv_bn(in_depth, out_depth, 3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = self._create_conv_bn(out_depth, out_depth, 3)

        self.skip = nn.Identity()
        if out_depth != in_depth or stride != 1:
            self.skip = self._create_conv_bn(
                in_depth, out_depth, 1, stride=stride)

    def _create_conv_bn(
        self,
        in_depth: int,
        out_depth: int,
        kernel_size: int,
        stride: int = 1
    ) -> nn.Module:
        block = nn.Sequential()
        block.add_module(
            "conv",
            nn.Conv2d(
                in_depth, out_depth, kernel_size, stride=stride,
                padding=kernel_size // 2, bias=False))
        block.add_module("bn", nn.BatchNorm2d(out_depth))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)

        out += self.skip(x)
        out = self.relu(out)
        return out
