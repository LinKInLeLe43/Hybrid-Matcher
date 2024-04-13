from typing import List, Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F


def _conv_1x1(in_depth: int, out_depth: int, stride: int = 1) -> nn.Module:
    conv = nn.Conv2d(in_depth, out_depth, 1, stride=stride, bias=False)
    return conv


def _conv_3x3(in_depth: int, out_depth: int, stride: int = 1) -> nn.Module:
    conv = nn.Conv2d(
        in_depth, out_depth, 3, stride=stride, padding=1, bias=False)
    return conv


def _crop_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(
        output, "n (c ww) l -> n l ww c", ww=kernel_size ** 2)
    return output


class FinePreprocess(nn.Module):
    def __init__(
        self,
        type: str,
        window_size: int,
        stride: int,
        padding: int,
        layer_depths: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        self.type = type
        self.window_size = window_size
        self.stride = stride
        self.padding = padding

        if type == "only_crop_fine":
            pass
        elif type == "fuse_coarse_after_crop_fine":
            if layer_depths is None:
                raise ValueError("")
            self.proj = nn.Linear(layer_depths[-1], layer_depths[0])
            self.merge = nn.Linear(2 * layer_depths[0], layer_depths[0])
        elif type == "fuse_all_before_crop":
            if layer_depths is None:
                raise ValueError("")
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            for i in range(len(layer_depths[:-1])):
                self.ups.append(_conv_1x1(layer_depths[i], layer_depths[i + 1]))
                self.downs.append(
                    nn.Sequential(
                        _conv_3x3(layer_depths[i + 1], layer_depths[i + 1]),
                        nn.BatchNorm2d(layer_depths[i + 1]),
                        nn.LeakyReLU(inplace=True),
                        _conv_3x3(layer_depths[i + 1], layer_depths[i])))
            self.ups.append(_conv_1x1(layer_depths[-1], layer_depths[-1]))
        else:
            raise ValueError("")

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def _only_crop_fine(
        self,
        fine_feature0: torch.Tensor,
        fine_feature1: torch.Tensor,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idxes, i_idxes, j_idxes = idxes
        fine_feature0 = _crop_windows(
            fine_feature0, self.window_size, self.stride, self.padding)
        fine_feature1 = _crop_windows(
            fine_feature1, self.window_size, self.stride, self.padding)
        fine_feature0 = fine_feature0[b_idxes, i_idxes]
        fine_feature1 = fine_feature1[b_idxes, j_idxes]
        return fine_feature0, fine_feature1

    def _fuse_coarse_after_crop_fine(
        self,
        fine_feature0: torch.Tensor,
        fine_feature1: torch.Tensor,
        coarse_feature0: torch.Tensor,
        coarse_feature1: torch.Tensor,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idxes, i_idxes, j_idxes = idxes
        coarse_feature0 = coarse_feature0[b_idxes, i_idxes]
        coarse_feature1 = coarse_feature1[b_idxes, j_idxes]
        coarse_feature = torch.cat([coarse_feature0, coarse_feature1])
        coarse_feature = self.proj(coarse_feature)[:, None]
        coarse_feature = coarse_feature.expand(-1, self.window_size ** 2, -1)

        fine_feature0, fine_feature1 = self._only_crop_fine(
            fine_feature0, fine_feature1, idxes)
        fine_feature = torch.cat([fine_feature0, fine_feature1])
        fine_feature = torch.cat([fine_feature, coarse_feature], dim=2)
        fine_feature0, fine_feature1 = self.merge(fine_feature).chunk(2)
        return fine_feature0, fine_feature1

    def _fuse_all_impl(
        self,
        features: List[torch.Tensor],
        size: torch.Size
    ) -> torch.Tensor:
        features[-1] = features[-1].transpose(1, 2).unflatten(2, size)
        features[-1] = self.ups[-1](features[-1].contiguous())
        for i in reversed(range(len(features[:-1]))):
            features[i] = self.ups[i](features[i])
            features[i] += F.interpolate(
                features[i + 1], scale_factor=2.0, mode="bilinear")
            features[i] = self.downs[i](features[i])
        return features[0]

    def _fuse_all_before_crop(
        self,
        features0: List[torch.Tensor],
        features1: List[torch.Tensor],
        size0: torch.Size,
        size1: torch.Size,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if size0 == size1:
            features = []
            for feature0, feature1 in zip(features0, features1):
                features.append(torch.cat([feature0, feature1]))
            fine_feature = self._fuse_all_impl(features, size0)
            fine_feature0, fine_feature1 = fine_feature.chunk(2)
        else:
            fine_feature0 = self._fuse_all_impl(features0, size0)
            fine_feature1 = self._fuse_all_impl(features1, size1)
        fine_feature0, fine_feature1 = self._only_crop_fine(
            fine_feature0, fine_feature1, idxes)
        return fine_feature0, fine_feature1

    def forward(
        self,
        features0: List[torch.Tensor],
        features1: List[torch.Tensor],
        size0: torch.Size,
        size1: torch.Size,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idxes, i_idxes, j_idxes = idxes
        w, ww = self.window_size, self.window_size ** 2
        if len(b_idxes) == 0:
            fine_feature0 = fine_feature1 = features0[0].new_empty(
                (0, ww, features0[0].shape[1]))
            return fine_feature0, fine_feature1

        if self.type == "only_crop_fine":
            fine_feature0, fine_feature1 = self._only_crop_fine(
                features0[0], features1[0], idxes)
        elif self.type == "fuse_coarse_after_crop_fine":
            fine_feature0, fine_feature1 = self._fuse_coarse_after_crop_fine(
                features0[0], features1[0], features0[-1], features1[-1], idxes)
        elif self.type == "fuse_all_before_crop":
            fine_feature0, fine_feature1 = self._fuse_all_before_crop(
                features0, features1, size0, size1, idxes)
        else:
            assert False
        return fine_feature0, fine_feature1
