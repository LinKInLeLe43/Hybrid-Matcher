import functools

from torch import nn

from .base import Backbone
from .mobileone_block import MobileOneBlock
from .resnet_block import ResNetBlock
from .self_cluster_block import SelfClusterBlock

RepVggBlock = functools.partial(
    MobileOneBlock, kernel_size=3, act=nn.ReLU(inplace=True))


def create_resnet_selfcoc_fpn():
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [2, 1, 2, 2, 1, 2, 2]
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
    return net


def create_repvgg_selfcoc_fpn():
    layer_depths = [64, 64, 128, 256, 256, 256, 256]
    strides = [2, 1, 2, 2, 1, 2, 2]
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
    return net
