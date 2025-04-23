# TODO:
# - Rename
# - Change weight init
# - Check `align_corners` in `fpn_fuse`
# - Remove `upscale_before_crop`

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FinePreprocess(nn.Module):
    def __init__(
        self,
        dims: List[int],
        window_size: int,
        stride: int,
        padding: int,
        right_extra: int = 0,
        upscale_before_crop: float = 1.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.upscale_before_crop = upscale_before_crop

        self.window_size0 = window_size
        self.window_size1 = window_size + 2 * right_extra
        self.padding0, self.padding1 = padding, padding + right_extra

        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims) - 1):
            dim0, dim1 = dims[i], dims[i + 1]
            self.ups.append(nn.Conv2d(dim0, dim1, 1, bias=False))
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(dim1, dim1, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(dim1, dim0, 3, padding=1, bias=False),
                )
            )
        self.ups.append(nn.Conv2d(dim1, dim1, 1, bias=False))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def crop_by_indices(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        index_triplet: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w0, w1 = self.window_size0, self.window_size1
        upscale = self.upscale_before_crop
        b_indices, i_indices, j_indices = index_triplet.chunk(3, dim=-1)

        if upscale > 1.0:
            x0 = F.interpolate(
                x0, scale_factor=upscale, mode="bilinear", align_corners=False
            )
            x1 = F.interpolate(
                x1, scale_factor=upscale, mode="bilinear", align_corners=False
            )

        cropped0 = F.unfold(x0, w0, stride=self.stride, padding=self.padding0)[
            b_indices, :, i_indices
        ]
        cropped1 = F.unfold(x1, w1, stride=self.stride, padding=self.padding1)[
            b_indices, :, j_indices
        ]
        m = cropped0.shape[0]
        cropped0 = cropped0.reshape(m, -1, int(w0**2)).transpose(-1, -2)
        cropped1 = cropped1.reshape(m, -1, int(w1**2)).transpose(-1, -2)
        # cropped0 = rearrange(cropped0, "m (c ww) -> m ww c", ww=w0**2)
        # cropped1 = rearrange(cropped1, "m (c ww) -> m ww c", ww=w1**2)
        return cropped0, cropped1

    def fpn_fuse(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        x = self.ups[-1](x_list[-1])
        x, y = self.ups[1](x_list[1]), x
        # FIXME: align_corners=False
        x = x + F.interpolate(
            y, scale_factor=2.0, mode="bilinear", align_corners=True
        )
        x = self.downs[1](x)
        x, y = self.ups[0](x_list[0]), x
        # FIXME: align_corners=False
        x = x + F.interpolate(
            y, scale_factor=2.0, mode="bilinear", align_corners=True
        )
        x = self.downs[0](x)
        return x

    def forward(
        self,
        x0_list: List[torch.Tensor],
        x1_list: List[torch.Tensor],
        index_triplet: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x0_list[0], x1_list[0]

        if index_triplet.shape[0] == 0:
            cropped0 = x0.new_empty(0, int(self.window_size0**2), x0.shape[1])
            cropped1 = x1.new_empty(0, int(self.window_size1**2), x1.shape[1])
            return cropped0, cropped1

        # if x0.shape == x1.shape:
        x_list = [torch.cat(x) for x in zip(x0_list, x1_list)]
        x0, x1 = self.fpn_fuse(x_list).chunk(2)
        # else:
        #     x0, x1 = self.fpn_fuse(x0_list), self.fpn_fuse(x1_list)
        cropped0, cropped1 = self.crop_by_indices(x0, x1, index_triplet)
        return cropped0, cropped1
