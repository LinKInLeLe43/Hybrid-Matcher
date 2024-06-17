from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, depth: int, ratio: float = 0.0625) -> None:
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv2d(depth, int(depth * ratio), 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(int(depth * ratio), depth, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.pooling(x)
        weight = self.conv0(weight)
        weight = self.relu(weight)
        weight = self.conv1(weight)
        weight.sigmoid_()
        out = weight * x
        return out


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        out_depth: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        conv_branch_count: int = 1,
        act: nn.Module = nn.GELU()
    ) -> None:
        super().__init__()
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.inference_mode = inference_mode
        self.conv_branch_count = conv_branch_count

        self.se = SEBlock(out_depth) if use_se else nn.Identity()
        self.act = act if use_act else nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_depth, out_depth, kernel_size, stride=stride,
                padding=kernel_size // 2, groups=groups, bias=True)
        else:
            self.rbr_convs = None
            if conv_branch_count > 0:
                self.rbr_convs = nn.ModuleList()
                for _ in range(conv_branch_count):
                    self.rbr_convs.append(self._create_conv_bn(kernel_size))

            self.rbr_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.rbr_scale = self._create_conv_bn(1)

            self.rbr_skip = None
            if out_depth == in_depth and stride == 1:
                self.rbr_skip = nn.BatchNorm2d(out_depth)

    def _create_conv_bn(self, kernel_size: int) -> nn.Module:
        block = nn.Sequential()
        block.add_module(
            "conv",
            nn.Conv2d(
                self.in_depth, self.out_depth, kernel_size, stride=self.stride,
                padding=kernel_size // 2, groups=self.groups, bias=False))
        block.add_module("bn", nn.BatchNorm2d(self.out_depth))
        return block

    def _fuse_bn(self, branch: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            bn = branch.bn
            weight = branch.conv.weight
            if branch.conv.kernel_size == (1, 1):
                weight = F.pad(weight, (1, 1, 1, 1))
        elif isinstance(branch, nn.BatchNorm2d):
            bn = branch
            depth, depth_per_group = self.in_depth, self.in_depth // self.groups
            weight = torch.zeros(
                (depth, depth_per_group, 3, 3), dtype=branch.weight.dtype,
                device=bn.weight.device)
            for i in range(depth):
                weight[i, i % depth_per_group, 1, 1] = 1.0
        else:
            assert False
        std = (bn.running_var + bn.eps).sqrt()
        weight = weight * (bn.weight / std)[:, None, None, None]
        bias = bn.bias - bn.running_mean * bn.weight / std
        return weight, bias

    def _get_eq_weight_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight, bias = 0.0, 0.0
        if self.rbr_convs is not None:
            for conv in self.rbr_convs:
                conv_weight, conv_bias = self._fuse_bn(conv)
                weight += conv_weight
                bias += conv_bias

        if self.rbr_scale is not None:
            scale_weight, scale_bias = self._fuse_bn(self.rbr_scale)
            weight += scale_weight
            bias += scale_bias

        if self.rbr_skip is not None:
            skip_weight, skip_bias = self._fuse_bn(self.rbr_skip)
            weight += skip_weight
            bias += skip_bias
        return weight, bias

    def reparameterize(self) -> None:
        if self.inference_mode:
            return

        conv = nn.Conv2d(
            self.in_depth, self.out_depth, self.kernel_size, stride=self.stride,
            padding=self.kernel_size // 2, groups=self.groups, bias=True)
        conv.weight.data, conv.bias.data = self._get_eq_weight_bias()
        self.reparam_conv = conv

        for p in self.parameters():
            p.detach_()
        del self.rbr_convs
        del self.rbr_scale
        del self.rbr_skip

        self.inference_mode = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            out = self.reparam_conv(x)
        else:
            out = 0.0
            if self.rbr_convs is not None:
                for conv in self.rbr_convs:
                    out += conv(x)

            if self.rbr_scale is not None:
                out += self.rbr_scale(x)

            if self.rbr_skip is not None:
                out += self.rbr_skip(x)

        out = self.se(out)
        out = self.act(out)
        return out
