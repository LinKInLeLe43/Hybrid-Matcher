from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_module: nn.Module,
        fine_matching: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_module = fine_module
        self.fine_matching = fine_matching

        self.scales = 16, 1
        self.extra_scale = 8
        self.use_flow = getattr(coarse_module, "use_flow", False)
        self.type = fine_matching.type
        self.cls_window_size = fine_matching.cls_window_size
        self.reg_window_size = fine_matching.reg_window_size

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None
    ) -> None:
        n = len(result["points0"])

        coarse_points0 = self.scales[0] * result["points0"]
        fine_points0 = coarse_points0.clone()
        if "biases0" in result:
            fine_points0 += self.scales[1] * result["biases0"][:n]
        if scale0 is not None:
            coarse_points0 *= scale0[result["idxes"][0]]
            fine_points0 *= scale0[result["idxes"][0]]
        result["coarse_points0"] = coarse_points0  # for evaluate coarse precision
        result["points0"] = fine_points0

        coarse_points1 = self.scales[0] * result["points1"]
        fine_points1 = coarse_points1.clone()
        if "biases1" in result:
            fine_points1 += self.scales[1] * result["biases1"][:n]
        if scale1 is not None:
            fine_points1 *= scale1[result["idxes"][0]]
        result["points1"] = fine_points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n = batch["image0"].shape[0]
        mask0_8x, mask1_8x = batch.get("mask0"), batch.get("mask1")
        mask0_16x, mask1_16x = batch.get("mask0_16x"), batch.get("mask1_16x")
        if (mask0_8x is None) == (mask1_8x is not None):
            raise ValueError("")

        pos_feature0 = pos_feature1 = None
        if batch["image0"].shape == batch["image1"].shape:
            data = torch.cat([batch["image0"], batch["image1"]])
            features = self.backbone(data)
            centers0, centers1 = features.pop(-1).chunk(2)
            coarse_feature0, coarse_feature1 = features.pop(-1).chunk(2)
            fine_features = features
            if self.use_flow:
                pos_feature = self.positional_encoding.get(
                    coarse_feature0).repeat(2 * n, 1, 1, 1)
                pos_feature = pos_feature.flatten(start_dim=2).transpose(1, 2)
                pos_feature0, pos_feature1 = pos_feature.chunk(2)
        else:
            features0 = self.backbone(batch["image0"])
            features1 = self.backbone(batch["image1"])
            centers0, centers1 = features0.pop(-1), features1.pop(-1)
            coarse_feature0 = features0.pop(-1)
            coarse_feature1 = features1.pop(-1)
            fine_features0, fine_features1 = features0, features1

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

        (coarse_feature0, coarse_feature1, matchability0, matchability1,
         flow0, flow1) = self.coarse_module(
            coarse_feature0, coarse_feature1, centers0, centers1, size0, size1,
            pos0=pos_feature0, pos1=pos_feature1, mask0=mask0_16x,
            mask1=mask1_16x)

        result = self.coarse_matching(
            coarse_feature0, coarse_feature1, size0, size1,
            matchability0=matchability0, matchability1=matchability1,
            flow0=flow0, flow1=flow1, mask0=mask0_16x, mask1=mask1_16x,
            gt_idxes=gt_idxes)

        coarse_feature0 = coarse_feature0.transpose(1, 2).unflatten(2, size0)
        coarse_feature1 = coarse_feature1.transpose(1, 2).unflatten(2, size1)
        if batch["image0"].shape == batch["image1"].shape:
            coarse_features = torch.cat([coarse_feature0, coarse_feature1])
            fine_features = self.backbone.fpn(fine_features + [coarse_features])
            fine_features0, fine_features1 = [], []
            for fine_feature in fine_features:
                fine_feature0, fine_feature1 = fine_feature.chunk(2)
                fine_features0.append(fine_feature0)
                fine_features1.append(fine_feature1)
        else:
            fine_features0 = self.backbone.fpn(
                fine_features0 + [coarse_feature0])
            fine_features1 = self.backbone.fpn(
                fine_features1 + [coarse_feature1])

        if True:
            size0, size1 = fine_features0[-2].shape[2:], fine_features1[-2].shape[2:]
            features0_8x = fine_features0[-2].flatten(start_dim=2).transpose(1, 2)
            features1_8x = fine_features1[-2].flatten(start_dim=2).transpose(1, 2)
            result_8x = self.coarse_matching(
                features0_8x, features1_8x, size0, size1,
                disable_matchability=True, mask0=mask0_8x, mask1=mask1_8x,
                gt_idxes=None)
            result["first_stage_cls_heatmap_16x"] = result["first_stage_cls_heatmap"]
            result["first_stage_cls_heatmap_8x"] = result_8x["first_stage_cls_heatmap"]

        fine_feature0, fine_feature1 = self.fine_preprocess(
            fine_features0, fine_features1, size0, size1,
            result["first_stage_idxes"])
        result["idxes"] = (
            result["idxes"][0].repeat_interleave(self.fine_matching.cls_top_k),
            result["idxes"][1].repeat_interleave(self.fine_matching.cls_top_k),
            result["idxes"][2].repeat_interleave(self.fine_matching.cls_top_k))
        result["points0"] = result["points0"].repeat_interleave(
            self.fine_matching.cls_top_k, dim=0)
        result["points1"] = result["points1"].repeat_interleave(
            self.fine_matching.cls_top_k, dim=0)
        if len(fine_feature0) != 0:
            fine_feature0, fine_feature1 = self.fine_module(
                fine_feature0, fine_feature1)
        result.update(self.fine_matching(fine_feature0, fine_feature1))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
