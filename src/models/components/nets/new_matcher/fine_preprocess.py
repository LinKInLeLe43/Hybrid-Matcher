from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class FinePreprocess(nn.Module):
    def __init__(
        self,
        type: str,
        window_size: int,
        stride: int,
        padding: int,
        depths: List[int],
        right_extra: int = 0,
        scale_before_crop: int = 1,
        norm_before_fuse: bool = False
    ) -> None:
        super().__init__()
        self.type = type
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.depths = depths
        self.right_extra = right_extra
        self.scale_before_crop = scale_before_crop
        self.norm_before_fuse = norm_before_fuse

        if type == "loftr":
            self.proj = nn.Linear(depths[-1], depths[0])
            self.merge = nn.Linear(2 * depths[0], depths[0])
        elif type == "eloftr":
            self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
            for i in range(len(depths[:-1])):
                c0, c1 = depths[i], depths[i + 1]
                self.ups.append(nn.Conv2d(c0, 2 * c0, 1, bias=False))
                self.downs.append(nn.Sequential(
                    nn.Conv2d(2 * c0, 2 * c0, 3, padding=1, bias=False),
                    nn.BatchNorm2d(2 * c0),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(
                        2 * c0, c0 if i == 0 else 2 * depths[i - 1], 3,
                        padding=1, bias=False)))
            self.ups.append(nn.Conv2d(c1, 2 * c0, 1, bias=False))
        else:
            raise ValueError("")

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def _crop_by_idxes(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s, e, scale = self.stride, self.right_extra, self.scale_before_crop
        w0, w1 = self.window_size, self.window_size + 2 * e
        p0, p1 = self.padding, self.padding + e
        ww0, ww1 = w0 ** 2, w1 ** 2
        b_idxes, i_idxes, j_idxes = idxes

        if scale > 1:
            x0 = F.interpolate(x0, scale_factor=scale, mode="bilinear")
            x1 = F.interpolate(x1, scale_factor=scale, mode="bilinear")

        out0 = F.unfold(x0, w0, padding=p0, stride=s)[b_idxes, :, i_idxes]
        out1 = F.unfold(x1, w1, padding=p1, stride=s)[b_idxes, :, j_idxes]
        out0 = out0.unflatten(1, (-1, ww0)).transpose(1, 2)
        out1 = out1.unflatten(1, (-1, ww1)).transpose(1, 2)
        return out0, out1

    def _fuse_loftr(
        self,
        x0s: List[torch.Tensor],
        x1s: List[torch.Tensor],
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.right_extra
        w0, w1 = self.window_size, self.window_size + 2 * e
        ww0, ww1 = w0 ** 2, w1 ** 2
        b_idxes, i_idxes, j_idxes = idxes

        x0 = x0s[-1].flatten(start_dim=2).transpose(1, 2)
        x1 = x1s[-1].flatten(start_dim=2).transpose(1, 2)
        y0, y1 = x0s[0], x1s[0]
        x = torch.cat([x0[b_idxes, i_idxes], x1[b_idxes, j_idxes]])
        x = self.proj(x)[:, None]
        y0, y1 = self._crop_by_idxes(y0, y1, idxes)

        if e == 0:
            x = x.expand(-1, ww0, -1)
            y = torch.cat([y0, y1])
            y = torch.cat([y, x], dim=2)
            out0, out1 = self.merge(y).chunk(2)
        else:
            x0, x1 = x.chunk(2)
            x0, x1 = x0.expand(-1, ww0, -1), x1.expand(-1, ww1, -1)
            y0, y1 = torch.cat([y0, x0], dim=2), torch.cat([y1, x1], dim=2)
            out0, out1 = self.merge(y0), self.merge(y1)
        return out0, out1

    def _fuse_eloftr_impl(self, xs: List[torch.Tensor]) -> torch.Tensor:
        xs[-1] = self.ups[-1](xs[-1])
        for i in reversed(range(len(xs[:-1]))):
            xs[i] = self.ups[i](xs[i])
            xs[i] += F.interpolate(
                xs[i + 1], scale_factor=2.0, mode="bilinear",
                align_corners=True)
            xs[i] = self.downs[i](xs[i])
        return xs[0]

    def _fuse_eloftr(
        self,
        x0s: List[torch.Tensor],
        x1s: List[torch.Tensor],
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0s[0].shape == x1s[0].shape:
            xs = [torch.cat(x) for x in zip(x0s, x1s)]
            out0, out1 = self._fuse_eloftr_impl(xs).chunk(2)
        else:
            out0 = self._fuse_eloftr_impl(x0s)
            out1 = self._fuse_eloftr_impl(x1s)
        out0, out1 = self._crop_by_idxes(out0, out1, idxes)
        return out0, out1

    def forward(
        self,
        x0s: List[torch.Tensor],
        x1s: List[torch.Tensor],
        idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ww0 = self.window_size ** 2
        ww1 = (self.window_size + 2 * self.right_extra) ** 2
        m, c = len(idxes[0]), self.depths[0]

        if m == 0:
            out0 = x0s[0].new_empty((0, ww0, c))
            out1 = x1s[0].new_empty((0, ww1, c))
            return out0, out1

        if self.norm_before_fuse:
            x0s[-1] = x0s[-1] / self.depths[-1] ** 0.5
            x1s[-1] = x1s[-1] / self.depths[-1] ** 0.5

        if self.type == "loftr":
            out0, out1 = self._fuse_loftr(x0s, x1s, idxes)
        elif self.type == "eloftr":
            out0, out1 = self._fuse_eloftr(x0s, x1s, idxes)
        else:
            assert False
        return out0, out1
