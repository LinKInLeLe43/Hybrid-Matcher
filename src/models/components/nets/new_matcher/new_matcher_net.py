from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_module: nn.Module,
        fine_reg_matching: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_module = fine_module
        self.fine_reg_matching = fine_reg_matching

        self.scales = (backbone.scales[0],
                       backbone.scales[1] // fine_preprocess.scale_before_crop)
        self.reg_w = fine_reg_matching.window_size

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None
    ) -> None:
        b_idxes = result["idxes"][0]

        points0 = self.scales[0] * result["points0"]
        if scale0 is not None:
            points0 *= scale0[b_idxes]

        points1 = self.scales[0] * result["points1"]
        biases = result["fine_reg_biases"][:len(points0)].detach()
        biases = self.scales[1] * (self.reg_w // 2) * biases
        if scale1 is not None:
            points1 *= scale1[b_idxes]
            biases *= scale1[b_idxes]
        points1 += biases
        result["points0"], result["points1"] = points0, points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        mask0_8x, mask1_8x = batch.get("mask0_8x"), batch.get("mask1_8x")
        mask0_32x, mask1_32x = batch.get("mask0_32x"), batch.get("mask1_32x")

        if batch["image0"].shape == batch["image1"].shape:
            x = torch.cat([batch["image0"], batch["image1"]])
            xs, x_8x = self.backbone(x)
            x_8x, x_32x = self.local_coc(x_8x)

            x0s, x1s = [], []
            for x in xs:
                x0, x1 = x.chunk(2)
                x0s.append(x0)
                x1s.append(x1)
            x0_8x, x1_8x = x_8x.chunk(2)
            x0_32x, x1_32x = x_32x.chunk(2)
        else:
            x0s, x0_8x = self.backbone(batch["image0"])
            x0_8x, x0_32x = self.local_coc(x0_8x)

            x1s, x1_8x = self.backbone(batch["image1"])
            x1_8x, x1_32x = self.local_coc(x1_8x)

        x0_8x, x1_8x = self.coarse_module(
            x0_8x, x1_8x, x0_32x, x1_32x, mask0_8x=mask0_8x, mask1_8x=mask1_8x,
            mask0_32x=mask0_32x, mask1_32x=mask1_32x)

        result = self.coarse_matching(
            x0_8x, x1_8x, mask0=mask0_8x, mask1=mask1_8x, gt_idxes=gt_idxes)

        x0_1x, x1_1x = self.fine_preprocess(
            x0s + [x0_8x], x1s + [x1_8x], result["coarse_cls_idxes"])

        if len(x0_1x) != 0:
            x0_1x, x1_1x = self.fine_module(x0_1x, x1_1x)

        result.update(self.fine_reg_matching(x0_1x, x1_1x))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
