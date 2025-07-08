from typing import Any, Dict, Optional, Sequence, Tuple

import kornia as K
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import crop_by_mask, pad_by_mask


class NewMatcherNet(Module):
    def __init__(
        self,
        type: str,
        backbone: Module,
        rope: Module,
        local_coc: Module,
        coarse_module: Module,
        coarse_matching: Module,
        fine_preprocess: Module,
        fine_cls_matching: Module,
        fine_reg_matching: Module,
        extra_scale: Optional[int] = None,
        enable_crop: bool = False,
    ) -> None:
        super().__init__()
        self.type = type
        self.backbone = backbone
        self.rope = rope
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_cls_matching = fine_cls_matching
        self.fine_reg_matching = fine_reg_matching
        self.extra_scale = extra_scale
        self.enable_crop = enable_crop

        self.scales = (
            backbone.scales[0],
            backbone.scales[1] // fine_preprocess.upscale_before_crop,
        )
        self.reg_w = fine_reg_matching.window_size

        grid = K.create_meshgrid(
            self.scales[0], self.scales[0], normalized_coordinates=False
        )
        grid = 2 * (grid + 0.5) / self.scales[0] - 1
        self.register_buffer("fine_reg_grid", grid, persistent=False)

    @torch.no_grad()
    def _refine_points(
        self,
        out: Dict[str, Any],
        scale0: Optional[Tensor],
        scale1: Optional[Tensor],
    ) -> None:
        b_idxes = out["coarse_cls_idxes"][0]

        points0 = out["points0"] * self.scales[0]
        points1 = out["points1"] * self.scales[0]
        points0 = points0 + out.pop("fine_cls_biases0")
        points1 = (
            points1
            + out.pop("fine_cls_biases1")
            + out["fine_reg_biases"] * self.scales[1] * (self.reg_w // 2)
        )
        if scale0 is not None and scale1 is not None:
            points0 = points0 * scale0[b_idxes]
            points1 = points1 * scale1[b_idxes]
        out["points0"], out["points1"] = points0, points1

    def forward(
        self,
        data: Dict[str, Any],
        gt_indices_list: Optional[
            Sequence[Tuple[Tensor, Tensor, Tensor]]
        ] = None,
    ) -> Dict[str, Any]:
        image0, image1 = data["image0"], data["image1"]
        mask0, mask1 = data.get("mask0"), data.get("mask1")
        scale0, scale1 = data.get("scale0"), data.get("scale1")

        if image0.shape == image1.shape:
            x_list = self.backbone(torch.cat([image0, image1]))
            x0_list, x1_list = zip(*(x.chunk(2) for x in x_list))
        else:
            x0_list, x1_list = self.backbone(image0), self.backbone(image1)
        x0_list, x1_list = list(x0_list), list(x1_list)

        x0_8x = self.rope.abs_pe(x0_list[-1])
        x1_8x = self.rope.abs_pe(x1_list[-1])
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
                b_x0_16x = pad_by_mask(
                    b_x0_16x, F.max_pool2d(mask0[[b]].float(), 2).bool()
                )
                b_x1_16x = pad_by_mask(
                    b_x1_16x, F.max_pool2d(mask1[[b]].float(), 2).bool()
                )
                x0_16x.append(b_x0_16x), x1_16x.append(b_x1_16x)
            x0_16x, x1_16x = torch.cat(x0_16x), torch.cat(x1_16x)
        else:
            x0_16x, x1_16x = self.coarse_module(
                self.local_coc(x0_8x, mask0),
                self.local_coc(x1_8x, mask1),
                rope=self.rope,
                mask0=mask0,
                mask1=mask1,
            )

        out = self.coarse_matching(
            [x0_list[-1], x0_16x],
            [x1_list[-1], x1_16x],
            mask0=mask0,
            mask1=mask1,
            gt_indices_list=gt_indices_list,
        )
        x0_list[-1], x1_list[-1] = out.pop("x_8x")

        x0_reg, x1_reg = self.fine_preprocess(
            x0_list, x1_list, out["coarse_cls_idxes"]
        )

        x_cls = (
            torch.cat([x0_reg, x1_reg])
            .transpose(-2, -1)
            .unflatten(-1, (self.reg_w, self.reg_w))
        )
        grid = self.fine_reg_grid.expand(x_cls.shape[0], -1, -1, -1)
        x0_cls, x1_cls = (
            F.grid_sample(x_cls, grid, mode="bilinear", align_corners=True)
            .flatten(start_dim=-2)
            .transpose(-2, -1)
            .chunk(2)
        )
        out.update(self.fine_cls_matching(x0_cls, x1_cls))

        init = torch.cat(
            [out["fine_cls_biases0"], out["fine_cls_biases1"]], dim=1
        )
        init = init / self.scales[0] + 0.5
        out.update(self.fine_reg_matching(x0_reg, x1_reg, 1, init=init))

        self._refine_points(out, scale0, scale1)
        return out
