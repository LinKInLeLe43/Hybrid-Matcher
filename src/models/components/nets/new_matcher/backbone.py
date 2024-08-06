from typing import List, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
import kornia as K


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.linear1 = nn.Linear(hidden_depth, out_depth, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout(x)
        return x


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


class SpecialCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        window_size: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.window_size = window_size

        self.proj0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.proj1 = nn.Linear(512, 2 * hidden_depth, bias=bias)
        self.merge = nn.Linear(hidden_depth, in_depth, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        x_point = self.proj0(x.permute(0, 2, 3, 1))
        center = self.proj1(center.permute(0, 2, 3, 1))
        x_point = einops.rearrange(
            x_point, "n (fh sh) (fw sw) (fc sc) -> n fh fw (sh sw) fc sc",
            fc=self.heads_count, sh=self.window_size, sw=self.window_size)
        center = center.unflatten(3, (self.heads_count, -1))
        center_point, center_value = center.chunk(2, dim=4)

        norm_x_point = F.normalize(x_point, dim=5)
        norm_center_point = F.normalize(center_point, dim=4)
        similarities = torch.einsum(
            "nhwkdc,nhwdc->nhwkd", norm_x_point, norm_center_point)
        similarities = self.alpha * similarities + self.beta
        similarities = similarities.sigmoid_()[..., None]

        dispatched = similarities * center_value[:, :, :, None]
        dispatched = einops.rearrange(
            dispatched, "n fh fw (sh sw) fc sc -> n (fh sh) (fw sw) (fc sc)",
            sh=self.window_size)
        dispatched = self.merge(dispatched)
        return dispatched


class SpecialClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        window_size: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.cluster = SpecialCluster(
            in_depth, hidden_depth, heads_count, window_size, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = Mlp(
            2 * in_depth, 2 * in_depth, in_depth, bias=bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(self, x: torch.Tensor, center: torch.Tensor, cat: bool = False) -> torch.Tensor:
        new_x = self.cluster(x, center)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x.permute(0, 2, 3, 1), new_x], dim=3)
        new_x = self.mlp(new_x)
        new_x = self.norm1(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()

        if cat:
            new_x = torch.cat([x, new_x], dim=1)
        else:
            new_x += x
        return new_x


class PretrainedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = 8, 2

        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad = False

        self._conv = nn.Conv2d(
            3, 128, 7, stride=2, padding=3, bias=False)
        self._norm = nn.BatchNorm2d(128)
        self._relu = nn.ReLU(inplace=True)
        self.in_depth = 128
        self._layer0 = self._make_layer(128)
        self._layer1 = self._make_layer(128, stride=2)
        self._layer2 = self._make_layer(128, stride=2)

        self.cluster_block0 = SpecialClusterBlock(128, 256, 8, 4)
        self.cluster_block1 = SpecialClusterBlock(256, 512, 16, 4)
        self.cluster_block2 = SpecialClusterBlock(256, 512, 16, 4)

    def _make_layer(self, depth: int, stride: int = 1) -> nn.Module:
        layer = nn.Sequential(
            _BasicBlock(self.in_depth, depth, stride=stride),
            _BasicBlock(depth, depth))
        self.in_depth = depth
        return layer

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = x.shape
        coors = K.create_meshgrid(h, w, device=x.device)
        coors = (coors / 2).permute(0, 3, 1, 2).expand(n, -1, -1, -1)
        y = torch.cat([x.mean(dim=1, keepdim=True), coors], dim=1)

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

        y2 = self.cluster_block0(y2, x4, cat=True)
        y2 = self.cluster_block1(y2, x4)
        y2 = self.cluster_block2(y2, x4)
        return y2, y0
