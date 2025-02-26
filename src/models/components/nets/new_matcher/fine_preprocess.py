from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinePreprocess(nn.Module):
    # TODO:
    # - Rename self.ups and self.downs
    # - Change weight init
    # - Check ``align_corners`` in ``_fpn_fuse``
    def __init__(
        self,
        feat_dims: List[int],
        window_size: int,
        stride: int,
        padding: int,
        right_extra: int = 0,
        upsample_factor_before_crop: float = 1.0,
        enable_scale_before_fuse: bool = False,
    ) -> None:
        super().__init__()
        self.upsample_factor_before_crop = upsample_factor_before_crop
        self.enable_scale_before_fuse = enable_scale_before_fuse

        self.scale = feat_dims[-1] ** -0.5
        self.W0, self.W1 = window_size, window_size + 2 * right_extra
        self.WW0, self.WW1 = self.W0**2, self.W1**2
        self.P0, self.P1 = padding, padding + right_extra
        self.S = stride

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
        r = self.upsample_factor_before_crop
        if r > 1.0:
            feat0 = F.interpolate(feat0, scale_factor=r, mode="bilinear")
            feat1 = F.interpolate(feat1, scale_factor=r, mode="bilinear")

        b_indices, i_indices, j_indices = index_triplet
        out0 = (
            F.unfold(feat0, self.W0, padding=self.P0, stride=self.S)
            .unflatten(1, (-1, self.WW0))
            .transpose(1, 2)
        )[b_indices, :, :, i_indices]
        out1 = (
            F.unfold(feat1, self.W1, padding=self.P1, stride=self.S)
            .unflatten(1, (-1, self.WW1))
            .transpose(1, 2)
        )[b_indices, :, :, j_indices]
        return out0, out1

    def _fpn_fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out = self.ups[-1](feats[-1])
        for i in reversed(range(len(feats) - 1)):
            out = self.downs[i](
                self.ups[i](feats[i])
                + F.interpolate(
                    out, scale_factor=2.0, mode="bilinear", align_corners=True
                )
            )
        return out

    def forward(
        self,
        feats0: List[torch.Tensor],
        feats1: List[torch.Tensor],
        index_triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if index_triplet[0].shape[0] == 0:
            out0 = feats0[0].new_empty(0, self.WW0, feats0[0].shape[1])
            out1 = feats1[0].new_empty(0, self.WW1, feats1[0].shape[1])
            return out0, out1

        if self.enable_scale_before_fuse:
            feats0[-1] = self.scale * feats0[-1]
            feats1[-1] = self.scale * feats1[-1]

        if feats0[0].shape == feats1[0].shape:
            out0, out1 = self._fpn_fuse(
                [torch.cat(x) for x in zip(feats0, feats1)]
            ).chunk(2)
        else:
            out0, out1 = self._fpn_fuse(feats0), self._fpn_fuse(feats1)
        out0, out1 = self._crop_patches_by_indices(out0, out1, index_triplet)
        return out0, out1
