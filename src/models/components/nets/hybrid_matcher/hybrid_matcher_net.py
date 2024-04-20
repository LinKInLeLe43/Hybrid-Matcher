from typing import Any, Dict, Optional, Tuple

import kornia as K
import torch
from torch import nn


class HybridMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_module: nn.Module,
        fine_matching: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_module = fine_module
        self.fine_matching = fine_matching

        self.scales = backbone.scales
        self.use_flow = getattr(coarse_module, "use_flow", False)
        self.type = fine_matching.type
        self.cls_window_size = fine_matching.cls_window_size
        self.reg_window_size = fine_matching.reg_window_size

        if (self.cls_window_size is not None and
            self.cls_window_size != self.scales[0] // self.scales[1]):
            raise ValueError("")

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None
    ) -> None:
        points0 = self.scales[0] * result["points0"]
        if "biases0" in result:
            points0 += self.scales[1] * result["biases0"][:len(points0)]
        if scale0 is not None:
            points0 *= scale0[result["b_idxes"]]
        result["points0"] = points0

        points1 = self.scales[0] * result["points1"]
        if "biases1" in result:
            points1 += self.scales[1] * result["biases1"][:len(points0)]
        if scale1 is not None:
            points1 *= scale1[result["b_idxes"]]
        result["points1"] = points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n = batch["image0"].shape[0]
        mask0, mask1 = batch.get("mask0"), batch.get("mask1")
        center0_mask = batch.get("center0_mask")
        center1_mask = batch.get("center1_mask")
        if (mask0 is None) == (mask1 is not None):
            raise ValueError("")

        pos_feature0 = pos_feature1 = None
        if batch["image0"].shape == batch["image1"].shape:
            data = torch.cat([batch["image0"], batch["image1"]])
            fine_features, coarse_feature = self.backbone(data)
            center, coarse_feature = self.local_coc(coarse_feature)
            centers0, centers1 = center.chunk(2)
            coarse_feature0, coarse_feature1 = coarse_feature.chunk(2)
            fine_features0, fine_features1 = [], []
            for fine_feature in fine_features:
                fine_feature0, fine_feature1 = fine_feature.chunk(2)
                fine_features0.append(fine_feature0)
                fine_features1.append(fine_feature1)
            if self.use_flow:
                pos_feature = self.positional_encoding.get(
                    coarse_feature).repeat(2 * n, 1, 1, 1)
                pos_feature = pos_feature.flatten(start_dim=2).transpose(1, 2)
                pos_feature0, pos_feature1 = pos_feature.chunk(2)
        else:
            fine_features0, coarse_feature0 = self.backbone(batch["image0"])
            centers0, coarse_feature0 = self.local_coc(coarse_feature0)

            fine_features1, coarse_feature1 = self.backbone(batch["image1"])
            centers1, coarse_feature1 = self.local_coc(coarse_feature1)

            if self.use_flow:
                pos_feature0 = self.positional_encoding.get(
                    coarse_feature0).repeat(n, 1, 1, 1)
                pos_feature0 = pos_feature0.flatten(start_dim=2).transpose(1, 2)
                pos_feature1 = self.positional_encoding.get(
                    coarse_feature1).repeat(n, 1, 1, 1)
                pos_feature1 = pos_feature1.flatten(start_dim=2).transpose(1, 2)
        size0, size1 = coarse_feature0.shape[2:], coarse_feature1.shape[2:]

        coarse_feature0, _ = self.positional_encoding(coarse_feature0)
        coarse_feature1, _ = self.positional_encoding(coarse_feature1)

        coarse_feature0, coarse_feature1, centers0, centers1 = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (coarse_feature0, coarse_feature1, centers0, centers1))

        coarse_feature0, coarse_feature1, flow0, flow1 = self.coarse_module(
            coarse_feature0, coarse_feature1, centers0, centers1, size0, size1,
            pos0=pos_feature0, pos1=pos_feature1, x0_mask=mask0, x1_mask=mask1,
            center0_mask=center0_mask, center1_mask=center1_mask)

        result = self.coarse_matching(
            coarse_feature0, coarse_feature1, size0, size1, flow0=flow0,
            flow1=flow1, mask0=mask0, mask1=mask1, gt_idxes=gt_idxes)

        fine_feature0, fine_feature1 = self.fine_preprocess(
            fine_features0 + [coarse_feature0],
            fine_features1 + [coarse_feature1], size0, size1,
            result["first_stage_idxes"])
        if len(fine_feature0) != 0:
            fine_feature0, fine_feature1 = self.fine_module(
                fine_feature0, fine_feature1)
        result.update(self.fine_matching(fine_feature0, fine_feature1))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
