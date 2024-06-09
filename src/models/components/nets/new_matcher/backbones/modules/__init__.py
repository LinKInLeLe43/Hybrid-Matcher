import functools

from torch import nn

from .mobileone_block import MobileOneBlock
from .resnet_block import ResNetBlock
from .self_cluster_block import SelfClusterBlock


RepVggBlock = functools.partial(
    MobileOneBlock, kernel_size=3, act=nn.ReLU(inplace=True),
    conv_branch_count=1)
