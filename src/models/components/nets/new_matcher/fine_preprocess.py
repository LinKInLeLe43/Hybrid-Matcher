from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO:
# - Rename self.ups and self.downs
# - Change weight init
# - Check ``align_corners`` in ``_fpn_fuse``


class FinePreprocess(nn.Module):
    def __init__(
        self,
        feat_dims: List[int],
        window_size: int,
        padding: int,
        stride: int,
        right_extra: int = 0,
        upsample_factor_before_crop: float = 1.0,
        enable_scale_before_fuse: bool = False,
    ) -> None:
        super().__init__()
        self.window_size0 = window_size
        self.window_size1 = window_size + 2 * right_extra
        self.padding0 = padding
        self.padding1 = padding + right_extra
        self.stride = stride
        self.upsample_factor_before_crop = upsample_factor_before_crop
        self.enable_scale_before_fuse = enable_scale_before_fuse
        self.scale = feat_dims[-1] ** -0.5

        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(feat_dims) - 1):
            btm_dim, top_dim = feat_dims[i], feat_dims[i + 1]
            self.ups.append(nn.Conv2d(btm_dim, top_dim, 1, bias=False))
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(top_dim, top_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(top_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(top_dim, btm_dim, 3, padding=1, bias=False),
                )
            )
        self.ups.append(nn.Conv2d(top_dim, top_dim, 1, bias=False))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def _crop_patches_by_indices(
        self,
        feat0: torch.Tensor,
        feat1: torch.Tensor,
        index_triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.upsample_factor_before_crop
        if s > 1.0:
            feat0 = F.interpolate(feat0, scale_factor=s, mode="bilinear")
            feat1 = F.interpolate(feat1, scale_factor=s, mode="bilinear")

        b_indices, i_indices, j_indices = index_triplet
        out0 = F.unfold(
            feat0, self.window_size0, padding=self.padding0, stride=self.stride
        )[b_indices, :, i_indices]
        out1 = F.unfold(
            feat1, self.window_size1, padding=self.padding1, stride=self.stride
        )[b_indices, :, j_indices]
        out0 = out0.unflatten(1, (feat0.shape[1], -1)).transpose(1, 2)
        out1 = out1.unflatten(1, (feat1.shape[1], -1)).transpose(1, 2)
        return out0, out1

    def _fpn_fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out = self.ups[-1](feats[-1])
        for i in reversed(range(len(feats) - 1)):
            out = F.interpolate(
                out, scale_factor=2.0, mode="bilinear", align_corners=True
            )
            out = out + self.ups[i](feats[i])
            out = self.downs[i](out)
        return out

    def forward(
        self,
        feats0: List[torch.Tensor],
        feats1: List[torch.Tensor],
        index_triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if index_triplet[0].shape[0] == 0:
            ww0, ww1 = self.window_size0**2, self.window_size1**2
            out0 = feats0[0].new_empty(0, ww0, feats0[0].shape[1])
            out1 = feats1[0].new_empty(0, ww1, feats1[0].shape[1])
            return out0, out1

        if self.enable_scale_before_fuse:
            feats0[-1] = feats0[-1] * self.scale
            feats1[-1] = feats1[-1] * self.scale

        if feats0[0].shape == feats1[0].shape:
            feats = [torch.cat(x) for x in zip(feats0, feats1)]
            out0, out1 = self._fpn_fuse(feats).chunk(2)
        else:
            out0, out1 = self._fpn_fuse(feats0), self._fpn_fuse(feats1)
        out0, out1 = self._crop_patches_by_indices(out0, out1, index_triplet)
        return out0, out1
