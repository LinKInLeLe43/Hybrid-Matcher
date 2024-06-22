from typing import Tuple
import functools

from torch import nn

from .base import Backbone, Fusion
from .mobileone_block import MobileOneBlock
from .resnet_block import ResNetBlock
from .self_cluster_block import SelfClusterBlock

RepVggBlock = functools.partial(
    MobileOneBlock, kernel_size=3, act=nn.ReLU(inplace=True))


def create_resnet_selfcoc_fpn() -> Tuple[nn.Module, nn.Module]:
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [1, 2, 2, 2, 1, 2, 2]
    stem = nn.Sequential(
        nn.Conv2d(
            1, layer_depths[0], 7, stride=strides[0], padding=7 // 2,
            bias=False),
        nn.BatchNorm2d(layer_depths[0]),
        nn.ReLU(inplace=True))
    SelfClusterBlock8x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(8, 8))
    SelfClusterBlock16x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(4, 4))
    SelfClusterBlock32x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(2, 2))
    block_builders = [stem, ResNetBlock, ResNetBlock, ResNetBlock,
                      SelfClusterBlock8x, SelfClusterBlock16x,
                      SelfClusterBlock32x]
    block_counts = [1, 2, 2, 2, 2, 2, 2]
    net = Backbone(block_builders, block_counts, layer_depths, strides)
    fusion = Fusion([64, 64, 128, 256, 256])
    return net, fusion


def create_repvgg_selfcoc_fpn() -> Tuple[nn.Module, nn.Module]:
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [1, 2, 2, 2, 1, 2, 2]
    SelfClusterBlock8x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(8, 8))
    SelfClusterBlock16x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(4, 4))
    SelfClusterBlock32x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(2, 2))
    block_builders = [RepVggBlock, RepVggBlock, RepVggBlock, RepVggBlock,
                      SelfClusterBlock8x, SelfClusterBlock16x,
                      SelfClusterBlock32x]
    block_counts = [1, 2, 4, 14, 2, 2, 2]
    net = Backbone(block_builders, block_counts, layer_depths, strides)
    fusion = Fusion([64, 64, 128, 256, 256])
    return net, fusion


def create_seperated_backbone_fpn(
) -> Tuple[nn.ModuleList, nn.Module]:
    layer_depths0 = [64, 128, 256, 256]
    strides0 = [4, 2, 2, 2]
    SelfClusterBlock4x = functools.partial(
        SelfClusterBlock, kernel_size=4, head_count=4, fold_size=(8, 8),
        anchor_size=(2, 2), hidden_depth=128, padding=0, with_coor=True)
    SelfClusterBlock8x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=4, fold_size=(1, 1),
        anchor_size=(8, 8))
    SelfClusterBlock16x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(4, 4))
    SelfClusterBlock32x = functools.partial(
        SelfClusterBlock, kernel_size=3, head_count=8, fold_size=(1, 1),
        anchor_size=(2, 2))
    block_builders0 = [SelfClusterBlock4x, SelfClusterBlock8x,
                       SelfClusterBlock16x, SelfClusterBlock32x]
    block_counts0 = [2, 2, 6, 2]
    net0 = Backbone(block_builders0, block_counts0, layer_depths0, strides0)

    layer_depths1 = [8, 16, 32, 64]
    strides1 = [1, 2, 2, 2]
    block_builders1 = [RepVggBlock, RepVggBlock, RepVggBlock, RepVggBlock]
    block_counts1 = [1, 2, 4, 4]
    net1 = Backbone(block_builders1, block_counts1, layer_depths1, strides1)
    fusion = Fusion(
        [None, None, 64, 128, 256], layer_depths1=[8, 16, 32, 64, None])
    return nn.ModuleList([net0, net1]), fusion
