from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .fine_matching import FineMatching
from .fine_preprocess import FinePreprocess
from .homo.fine_homo import FineHomo
from .utils import crop_by_mask, pad_by_mask


class NewMatcherNet(Module):
    def __init__(
        self,
        backbone: Module,
        rope: Module,
        local_coc: Module,
        coarse_module: Module,
        coarse_matching: Module,
        extra_scale: Optional[int] = None,
        enable_crop: bool = False,
    ) -> None:
        super().__init__()
        self.scales = (8, 2)
        self.dim_list = [64, 128, 192]
        self.fine_w = 5
        grid = create_meshgrid(
            self.scales[0], self.scales[0], normalized_coordinates=False
        )
        grid = (grid + 0.5) * 2.0 / self.scales[0] - 1.0
        self.register_buffer("fine_cls_grid", grid, persistent=False)

        self.fine_preprocess = FinePreprocess(
            self.dim_list,
            self.fine_w,
            self.scales[0] // self.scales[1],
            self.fine_w // 2,
        )
        self.fine_cls_matching = FineMatching(
            "cls", self.dim_list[0], self.scales[0]
        )
        self.fine_reg_matching = FineHomo(self.fine_w)
        self.backbone = backbone
        self.rope = rope
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.extra_scale = extra_scale
        self.enable_crop = enable_crop

    def _transform_feature(
        self,
        x0: Tensor,
        x1: Tensor,
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        x0_8x, x1_8x = self.rope.abs_pe(x0), self.rope.abs_pe(x1)
        if self.enable_crop and mask0 is not None and mask1 is not None:
            x0_16x, x1_16x = [], []
            x0_8x_list = crop_by_mask(x0_8x, mask0)
            x1_8x_list = crop_by_mask(x1_8x, mask1)
            for b in range(x0_8x.shape[0]):
                b_x0_16x, b_x1_16x = self.coarse_module(
                    self.local_coc(x0_8x_list[b]),
                    self.local_coc(x1_8x_list[b]),
                    rope=self.rope,
                )
                b_mask0 = F.max_pool2d(mask0[[b]].float(), 2).bool()
                b_mask1 = F.max_pool2d(mask1[[b]].float(), 2).bool()
                x0_16x.append(pad_by_mask(b_x0_16x, b_mask0))
                x1_16x.append(pad_by_mask(b_x1_16x, b_mask1))
            x0_16x, x1_16x = torch.cat(x0_16x), torch.cat(x1_16x)
        else:
            x0_16x, x1_16x = self.coarse_module(
                self.local_coc(x0_8x, mask0),
                self.local_coc(x1_8x, mask1),
                rope=self.rope,
                mask0=mask0,
                mask1=mask1,
            )
        return x0_16x, x1_16x

    def _sample_for_cls(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        x = (
            torch.cat([x0, x1])
            .transpose(-2, -1)
            .unflatten(-1, (self.fine_w, self.fine_w))
        )
        grid = self.fine_cls_grid.expand(x.shape[0], -1, -1, -1)
        x0, x1 = (
            F.grid_sample(x, grid, mode="bilinear", align_corners=True)
            .flatten(start_dim=-2)
            .transpose(-2, -1)
            .chunk(2)
        )
        return x0, x1

    @torch.no_grad()
    def _get_points(
        self, data: Dict[str, Any], out: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        b_indices, i_indices, j_indices = out["coarse_cls_indices"]

        w0 = data["image0"].shape[-1] // self.scales[0]
        w1 = data["image1"].shape[-1] // self.scales[0]
        points0 = (
            torch.stack([i_indices % w0, i_indices // w0], dim=-1).float()
            * self.scales[0]
        )
        points1 = (
            torch.stack([j_indices % w1, j_indices // w1], dim=-1).float()
            * self.scales[0]
        )

        points0 = points0 + out["fine_cls_biases0"]
        points1 = (
            points1
            + out["fine_cls_biases1"]
            + out["fine_reg_biases"] * self.scales[1] * (self.fine_w // 2)
        )

        if "scale0" in data and "scale1" in data:
            points0 = points0 * data["scale0"][b_indices]
            points1 = points1 * data["scale1"][b_indices]
        return points0, points1

    def forward(
        self,
        data: Dict[str, Any],
        gt_indices_list: Optional[List[Tuple[Tensor, Tensor, Tensor]]] = None,
    ) -> Dict[str, Any]:
        image0, image1 = data["image0"], data["image1"]
        mask0, mask1 = data.get("mask0"), data.get("mask1")
        out = {}

        if image0.shape == image1.shape:
            x_list = self.backbone(torch.cat([image0, image1]))
            x0_list, x1_list = zip(*(x.chunk(2) for x in x_list))
        else:
            x0_list, x1_list = self.backbone(image0), self.backbone(image1)
        x0_list, x1_list = list(x0_list), list(x1_list)
        x0_8x, x1_8x = x0_list.pop(-1), x1_list.pop(-1)

        x0_16x, x1_16x = self._transform_feature(x0_8x, x1_8x, mask0, mask1)
        out.update(
            self.coarse_matching(
                [x0_8x, x0_16x],
                [x1_8x, x1_16x],
                mask0=mask0,
                mask1=mask1,
                gt_indices_list=gt_indices_list,
            )
        )
        x0_8x, x1_8x = out.pop("x_8x")

        x0_fine, x1_fine = self.fine_preprocess(
            [*x0_list, x0_8x], [*x1_list, x1_8x], out["coarse_cls_indices"]
        )
        out.update(
            self.fine_cls_matching(*self._sample_for_cls(x0_fine, x1_fine))
        )
        biases = torch.cat(
            [out["fine_cls_biases0"], out["fine_cls_biases1"]], dim=-1
        )
        biases = biases / self.scales[0] + 0.5
        out.update(self.fine_reg_matching(x0_fine, x1_fine, 1, biases=biases))

        out["points0"], out["points1"] = self._get_points(data, out)
        return out
