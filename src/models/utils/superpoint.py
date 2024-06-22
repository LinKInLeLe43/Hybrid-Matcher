from collections import OrderedDict

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F


def _crop_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    ww = kernel_size ** 2
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(output, "n (c ww) l -> n l ww c", ww=ww)
    return output


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(OrderedDict([("conv", conv),
                                      ("activation", activation),
                                      ("bn", bn)]))


class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [64, 64, 128, 128, 256]
        self.stride = 2 ** (len(self.channels) - 2)
        channels = [1, *self.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride ** 2 + 1, 1, relu=False))

        self.load_state_dict(
            torch.load("weights/superpoint_v6_from_tf.pth"), strict=False)
        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        w = window_size

        features = self.backbone(x)
        heatmap = self.detector(features)[:, :-1]
        heatmap = einops.rearrange(
            heatmap, "n (sh sw) fh fw -> n (fh sh) (fw sw)", sh=8)
        heatmap = F.pad(heatmap[:, None], [w // 2, 0, w // 2, 0])
        heatmap = _crop_windows(heatmap, w, w, 0)[..., 0]
        heatmap = F.softmax(heatmap, dim=2)
        return heatmap
