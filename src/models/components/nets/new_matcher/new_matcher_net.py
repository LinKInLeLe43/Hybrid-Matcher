from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        extra_coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_matching: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.extra_coarse_matching = extra_coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_matching = fine_matching

        self.scales = 8, 1
        self.extra_scale = 16
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
        mask0_8x, mask1_8x = batch.get("mask0_8x"), batch.get("mask1_8x")
        mask0_16x = mask1_16x = mask0_32x = mask1_32x = None
        if mask0_8x is not None and mask1_8x is not None:
            mask = torch.stack([mask0_8x, mask1_8x]).float()
            mask0_16x, mask1_16x = F.max_pool2d(mask, 2, stride=2).bool()
            mask0_32x, mask1_32x = F.max_pool2d(mask, 4, stride=4).bool()

        if batch["image0"].shape == batch["image1"].shape:
            data = torch.cat([batch["image0"], batch["image1"]])
            features = self.backbone(data)
            feature0_32x, feature1_32x = features.pop(-1).chunk(2)
            feature0_16x, feature1_16x = features.pop(-1).chunk(2)
        else:
            features0 = self.backbone(batch["image0"])
            features1 = self.backbone(batch["image1"])
            feature0_32x, feature1_32x = features0.pop(-1), features1.pop(-1)
            feature0_16x, feature1_16x = features0.pop(-1), features1.pop(-1)

        size0, size1 = feature0_16x.shape[2:], feature1_16x.shape[2:]

        feature0_16x, _ = self.positional_encoding(feature0_16x)
        feature1_16x, _ = self.positional_encoding(feature1_16x)

        feature0_16x, feature1_16x, feature0_32x, feature1_32x = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (feature0_16x, feature1_16x, feature0_32x, feature1_32x))

        (feature0_16x, feature1_16x,
         matchability0, matchability1) = self.coarse_module(
            feature0_16x, feature1_16x, feature0_32x, feature1_32x, size0,
            size1, x0_mask=mask0_16x, x1_mask=mask1_16x, center0_mask=mask0_32x,
            center1_mask=mask1_32x)

        positional_encoding0 = positional_encoding1 = None
        if self.coarse_matching.type == "flow":
            positional_encoding0 = self.positional_encoding.positional_encoding[None, :, :size0[0], :size0[1]].flatten(start_dim=2)
            positional_encoding1 = self.positional_encoding.positional_encoding[None, :, :size1[0], :size1[1]].flatten(start_dim=2)
        result = self.coarse_matching(
            feature0_16x, feature1_16x, size0, size1,
            matchability0=matchability0, matchability1=matchability1,
            positional_encoding0=positional_encoding0,
            positional_encoding1=positional_encoding1, mask0=mask0_16x,
            mask1=mask1_16x, upsampled_mask0=mask0_8x,
            upsampled_mask1=mask1_8x, gt_idxes=gt_idxes)

        feature0_16x = feature0_16x.transpose(1, 2).unflatten(2, size0)
        feature1_16x = feature1_16x.transpose(1, 2).unflatten(2, size1)
        if batch["image0"].shape == batch["image1"].shape:
            features_16x = torch.cat([feature0_16x, feature1_16x])
            features = self.backbone.fuse(features + [features_16x])
            features0, features1 = [], []
            for feature in features:
                feature0, feature1 = feature.chunk(2)
                features0.append(feature0)
                features1.append(feature1)
        else:
            features0 = self.backbone.fuse(features0 + [feature0_16x])
            features1 = self.backbone.fuse(features1 + [feature1_16x])

        if True:
            size0, size1 = features0[-2].shape[2:], features1[-2].shape[2:]
            features0_8x = features0[-2].flatten(start_dim=2).transpose(1, 2)
            features1_8x = features1[-2].flatten(start_dim=2).transpose(1, 2)
            result_8x = self.extra_coarse_matching(
                features0_8x, features1_8x, size0, size1, mask0=mask0_8x,
                mask1=mask1_8x)
            result["first_stage_cls_heatmap_16x"] = result["first_stage_cls_heatmap"]
            result["first_stage_cls_heatmap_8x"] = result_8x["first_stage_cls_heatmap"]

        fine_feature0, fine_feature1 = self.fine_preprocess(
            features0[0], features1[0], result["first_stage_idxes"])

        result.update(self.fine_matching(fine_feature0, fine_feature1))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
