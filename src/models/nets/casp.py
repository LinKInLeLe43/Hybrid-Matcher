from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .encoders import Encoder


class CasP(Module):
    def __init__(
        self,
        encoder: str,
        rope: Module,
        coarse_module1: Module,
        coarse_matching1: Module,
        coarse_module2: Module,
        coarse_matching2: Module,
        extra_scale: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder)
        self.encoder.scales = (8, 4)
        self.rope = rope
        self.coarse_module1 = coarse_module1
        self.coarse_matching1 = coarse_matching1
        self.coarse_module2 = coarse_module2
        self.coarse_matching2 = coarse_matching2
        self.extra_scale = extra_scale

        self.scales = (self.encoder.scales[0], self.encoder.scales[1])

        self.encoder.backbone.in_planes = 128
        self.backbone2 = self.encoder.backbone._make_stage(256, 4, 2)

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[Tensor] = None,
        scale1: Optional[Tensor] = None,
    ) -> None:
        b_idxes = result["idxes"][0]

        coarse_points0 = self.scales[0] * result["points0"]
        coarse_points1 = self.scales[0] * result["points1"]

        if scale0 is not None and scale1 is not None:
            coarse_points0 *= scale0[b_idxes]
            coarse_points1 *= scale1[b_idxes]
        result["coarse_points0"] = coarse_points0
        result["coarse_points1"] = coarse_points1
        result["points0"], result["points1"] = coarse_points0, coarse_points1

    def forward(
        self,
        data: Dict[str, Any],
        gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        extra_gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Dict[str, Any]:
        mask0_8x, mask1_8x = data.get("mask0_8x"), data.get("mask1_8x")
        mask0_16x, mask1_16x = data.get("mask0_16x"), data.get("mask1_16x")

        x0_list, x1_list = self.encoder(data["image0"], data["image1"])

        x0_16x, x1_16x = x0_list.pop(-1), x1_list.pop(-1)
        encoding = self.rope.get_encoding()
        x0_16x_t, x1_16x_t = self.coarse_module1(
            x0_16x, x1_16x, encoding, mask0=mask0_16x, mask1=mask1_16x
        )

        x0_8x, x1_8x = x0_list.pop(-1), x1_list.pop(-1)
        result1 = self.coarse_matching1(
            x0_8x,
            x1_8x,
            x0_16x_t,
            x1_16x_t,
            x0_mask=mask0_8x,
            x1_mask=mask1_8x,
            y0_mask=mask0_16x,
            y1_mask=mask1_16x,
            x_gt_idxes=gt_idxes,
            y_gt_idxes=extra_gt_idxes,
        )
        x0_8x, x1_8x = result1.pop("x_8x")

        if x0_8x.shape == x1_8x.shape:
            out = torch.cat([x0_8x, x1_8x])
            for module in self.backbone2:
                out = module(out)
            x0_16x, x1_16x = out.chunk(2)
        else:
            x0_16x, x1_16x = x0_8x, x1_8x
            for module in self.backbone2:
                x0_16x = module(x0_16x)
                x1_16x = module(x1_16x)

        x0_16x_t, x1_16x_t = self.coarse_module2(
            x0_16x, x1_16x, encoding, mask0=mask0_16x, mask1=mask1_16x
        )

        result2 = self.coarse_matching2(
            x0_8x,
            x1_8x,
            x0_16x_t,
            x1_16x_t,
            x0_mask=mask0_8x,
            x1_mask=mask1_8x,
            y0_mask=mask0_16x,
            y1_mask=mask1_16x,
            x_gt_idxes=gt_idxes,
            y_gt_idxes=extra_gt_idxes,
        )
        x0_8x, x1_8x = result2.pop("x_8x")

        result = result2
        if self.training:
            result["coarse_cls_heatmap1"] = result1.pop("coarse_cls_heatmap")
            result["extra_coarse_cls_heatmap1"] = result1.pop("extra_coarse_cls_heatmap")
            result["coarse_cls_heatmap2"] = result2.pop("coarse_cls_heatmap")
            result["extra_coarse_cls_heatmap2"] = result2.pop("extra_coarse_cls_heatmap")

        self._scale_points(result, data.get("scale0"), data.get("scale1"))
        return result
