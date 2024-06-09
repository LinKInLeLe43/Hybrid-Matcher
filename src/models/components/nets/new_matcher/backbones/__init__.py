import functools
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.nets.new_matcher.backbones import modules


class Backbone(nn.Module):
    def __init__(
        self,
        block_builders: List,
        block_counts: List[int],
        layer_depths: List[int],
        strides: List[int],
        use_fpn: bool = True,
        coarse_scale: int = 16
    ) -> None:
        super().__init__()
        assert len(block_counts) == len(layer_depths) == len(strides)
        self.strides = strides

        self.layers = nn.ModuleList()
        self.in_depth = 1
        start_idx = 0
        if isinstance(block_builders[0], nn.Module):
            assert block_counts[0] == 1
            self.layers.append(block_builders[0])
            self.in_depth = layer_depths[0]
            start_idx = 1
        for i in range(start_idx, len(layer_depths)):
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
                self.ups.append(self._make_up_branch(
                    fpn_layer_depths[i], fpn_layer_depths[i + 1]))
                self.downs.append(self._make_down_branch(
                    fpn_layer_depths[i + 1], fpn_layer_depths[i]))
            self.ups.append(self._make_up_branch(
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

    def _make_up_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        branch = nn.Conv2d(in_depth, out_depth, 1, bias=False)
        return branch

    def _make_down_branch(self, in_depth: int, out_depth: int) -> nn.Module:
        branch = nn.Sequential(
            nn.Conv2d(in_depth, in_depth, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_depth),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_depth, out_depth, 3, padding=1, bias=False))
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

    def fpn(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features[-1] = self.ups[-1](features[-1])
        for i in reversed(range(len(features[:-1]))):
            features[i] = self.ups[i](features[i])
            features[i] += F.interpolate(
                features[i + 1], scale_factor=self.fpn_scale[i],
                mode="bilinear")
            features[i] = self.downs[i](features[i])
        return features


def create_resnet_selfcoc_fpn_1x():
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [1, 2, 2, 2, 1, 2, 2]
    stem = nn.Sequential(
        nn.Conv2d(
            1, layer_depths[0], 7, stride=strides[0], padding=7 // 2,
            bias=False),
        nn.BatchNorm2d(layer_depths[0]),
        nn.ReLU(inplace=True))
    _SelfClusterBlock_8x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=8, fold_size=1)
    _SelfClusterBlock_16x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=4, fold_size=1)
    _SelfClusterBlock_32x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=2, fold_size=1)
    block_builders = [
        stem, modules.ResNetBlock, modules.ResNetBlock, modules.ResNetBlock,
        _SelfClusterBlock_8x, _SelfClusterBlock_16x, _SelfClusterBlock_32x]
    block_counts = [1, 2, 2, 2, 2, 2, 2]
    net = Backbone(block_builders, block_counts, layer_depths, strides)
    return net


def create_repvgg_selfcoc_fpn_1x():
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [1, 2, 2, 2, 1, 2, 2]
    _SelfClusterBlock_8x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=8, fold_size=1)
    _SelfClusterBlock_16x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=4, fold_size=1)
    _SelfClusterBlock_32x = functools.partial(
        modules.SelfClusterBlock, head_count=8, center_size=2, fold_size=1)
    block_builders = [
        modules.RepVggBlock, modules.RepVggBlock, modules.RepVggBlock,
        modules.RepVggBlock, _SelfClusterBlock_8x, _SelfClusterBlock_16x,
        _SelfClusterBlock_32x]
    block_counts = [1, 2, 4, 14, 2, 2, 2]
    net = Backbone(block_builders, block_counts, layer_depths, strides)
    return net
