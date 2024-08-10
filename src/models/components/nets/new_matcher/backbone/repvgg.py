from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _create_conv_bn_branch(
    in_depth: int,
    out_depth: int,
    kernel_size,
    stride: int = 1,
    padding: int = 0
) -> nn.Module:
    branch = nn.Sequential()
    branch.add_module(
        "conv",
        nn.Conv2d(
            in_depth, out_depth, kernel_size, stride=stride, padding=padding,
            bias=False))
    branch.add_module("bn", nn.BatchNorm2d(out_depth))
    return branch


class RepVggBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        out_depth: int,
        stride: int = 1,
        deploy: bool = False
    ) -> None:
        super().__init__()
        self.deploy = deploy

        if deploy:
            self.branch_reparam = nn.Conv2d(
                in_depth,  out_depth, 3, stride=stride, padding=1)
        else:
            self.branch_3x3 = _create_conv_bn_branch(
                in_depth, out_depth, 3, stride=stride, padding=1)
            self.branch_1x1 = _create_conv_bn_branch(
                in_depth, out_depth, 1, stride=stride)
            self.branch_identity = None
            if out_depth == in_depth and stride == 1:
                self.branch_identity = nn.BatchNorm2d(in_depth)
        self.relu = nn.ReLU(inplace=True)

    def _fuse_bn(
        self,
        branch: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            bn = branch.bn
            weight = branch.conv.weight
            if branch.conv.kernel_size == (1, 1):
                weight = F.pad(weight, (1, 1, 1, 1))
        elif isinstance(branch, nn.BatchNorm2d):
            assert isinstance(branch, nn.BatchNorm2d)
            bn = branch
            if not hasattr(self, "template_identity"):
                depth = bn.num_features
                self.template_identity = torch.zeros(
                    (depth, depth, 3, 3), device=bn.weight.device)
                self.template_identity[range(depth), range(depth), 1, 1] = 1.0
            weight = self.template_identity
        else:
            assert False
        std = (bn.running_var + bn.eps).sqrt()
        fused_weight = (bn.weight / std)[:, None, None, None] * weight
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_weight, fused_bias

    def _get_eq_weight_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_3x3, bias_3x3 = self._fuse_bn(self.branch_3x3)
        weight_1x1, bias_1x1 = self._fuse_bn(self.branch_1x1)
        weight = weight_3x3 + weight_1x1
        bias = bias_3x3 + bias_1x1

        if self.branch_identity is not None:
            weight_identity, bias_identity = self._fuse_bn(self.branch_identity)
            weight += weight_identity
            bias += bias_identity
        return weight, bias

    def switch_to_deploy(self) -> None:
        if hasattr(self, "branch_reparam"):
            return

        conv_3x3 = self.branch_3x3.conv
        self.branch_reparam = nn.Conv2d(
            conv_3x3.in_channels, conv_3x3.out_channels, conv_3x3.kernel_size,
            stride=conv_3x3.stride, padding=conv_3x3.padding)
        (self.branch_reparam.weight.data,
         self.branch_reparam.bias.data) = self._get_eq_weight_bias()

        del self.branch_3x3
        del self.branch_1x1
        del self.branch_identity
        if hasattr(self, "template_identity"):
            del self.template_identity
        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            out = self.branch_reparam(x)
        else:
            out = self.branch_3x3(x) + self.branch_1x1(x)
            if self.branch_identity is not None:
                out += self.branch_identity(x)
        out = self.relu(out)
        return out


class RepVgg82(nn.Module):
    def __init__(
        self,
        block_counts: List[int],
        layer_depths: List[int],
        deploy: bool = False
    ) -> None:
        super().__init__()
        self.deploy = deploy
        self.in_depth = layer_depths[0]
        self.scales = (8, 2)

        self.layer0 = RepVggBlock(  # 1/2
            1, self.in_depth, stride=2, deploy=self.deploy)
        self.layer1 = self._make_layer(  # 1/2
            layer_depths[0], block_counts[0])
        self.layer2 = self._make_layer(  # 1/4
            layer_depths[1], block_counts[1], stride=2)
        self.layer3 = self._make_layer(  # 1/8
            layer_depths[2], block_counts[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(
        self,
        out_depth: int,
        block_count: int,
        stride: int = 1
    ) -> nn.Module:
        strides = [stride] + (block_count - 1) * [1]
        layer = nn.Sequential()
        for stride in strides:
            layer.append(RepVggBlock(
                self.in_depth, out_depth, stride=stride, deploy=self.deploy))
            self.in_depth = out_depth
        return layer

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [x1, x2], x3
