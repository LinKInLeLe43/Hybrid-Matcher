from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules.utils import crop_windows


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        type: str,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_cls_matching: nn.Module,
        fine_reg_matching: nn.Module,
        use_extra: bool = False
    ) -> None:
        super().__init__()

        self.type = type
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_cls_matching = fine_cls_matching
        self.fine_reg_matching = fine_reg_matching

        if use_extra:
            self.extra_scale = 8

        self.scales = 16, 2, 1
        self.use_flow = False
        self.fine_reg_window_size = fine_reg_matching.window_size

        if 2 * self.scales[1] / self.scales[2] > self.fine_reg_window_size:
            raise ValueError("")

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
        mask0_16x, mask1_16x = batch.get("mask0_16x"), batch.get("mask1_16x")
        mask0_32x, mask1_32x = batch.get("mask0_32x"), batch.get("mask1_32x")

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
            size1, mask0_16x=mask0_16x, mask1_16x=mask1_16x,
            mask0_32x=mask0_32x, mask1_32x=mask1_32x)

        result = self.coarse_matching(
            feature0_16x, feature1_16x, size0, size1,
            matchability0=matchability0, matchability1=matchability1,
            mask0=mask0_16x, mask1=mask1_16x, gt_idxes=gt_idxes)

        feature0_16x = feature0_16x.transpose(1, 2).unflatten(2, size0)
        feature1_16x = feature1_16x.transpose(1, 2).unflatten(2, size1)
        align_corners = [False, False, False, False]
        if batch["image0"].shape == batch["image1"].shape:
            features_16x = torch.cat([feature0_16x, feature1_16x])
            features = self.backbone.fuse(
                features + [features_16x], align_corners)
            features0, features1 = [], []
            for feature in features:
                feature0, feature1 = feature.chunk(2)
                features0.append(feature0)
                features1.append(feature1)
        else:
            features0 = self.backbone.fuse(
                features0 + [feature0_16x], align_corners)
            features1 = self.backbone.fuse(
                features1 + [feature1_16x], align_corners)

        if True:
            size0, size1 = features0[-2].shape[2:], features1[-2].shape[2:]
            features0_8x = features0[-2].flatten(start_dim=2).transpose(1, 2)
            features1_8x = features1[-2].flatten(start_dim=2).transpose(1, 2)
            use_matchability = self.coarse_matching.use_matchability
            self.coarse_matching.use_matchability = False
            result["coarse_extra_cls_heatmap"] = self.coarse_matching(
                features0_8x, features1_8x, size0, size1, mask0=mask0_8x,
                mask1=mask1_8x)["coarse_cls_heatmap"]
            self.coarse_matching.use_matchability = use_matchability

        b_idxes, i_idxes, j_idxes = result["coarse_cls_idxes"]
        fine_cls_w = self.fine_cls_matching.window_size
        fine_reg_w = self.fine_reg_matching.window_size
        reg_upsample_scale = (self.scales[1] // self.scales[2]
                              if len(self.scales) == 3 else 1)
        p = fine_reg_w // 2 // reg_upsample_scale
        w = fine_cls_w + 2 * p
        fine_feature0 = crop_windows(
            features0[0], w, stride=fine_cls_w, padding=p)
        fine_feature1 = crop_windows(
            features1[0], w, stride=fine_cls_w, padding=p)
        fine_feature0 = fine_feature0[b_idxes, i_idxes]
        fine_feature1 = fine_feature1[b_idxes, j_idxes]
        fine_cls_feature0, fine_cls_feature1 = fine_feature0, fine_feature1
        if p != 0:
            fine_cls_feature0 = (fine_feature0
                                 .unflatten(1, (w, w))[:, p:-p, p:-p]
                                 .flatten(start_dim=1, end_dim=2))
            fine_cls_feature1 = (fine_feature1
                                 .unflatten(1, (w, w))[:, p:-p, p:-p]
                                 .flatten(start_dim=1, end_dim=2))
        result.update(self.fine_cls_matching(
            fine_cls_feature0, fine_cls_feature1))

        m_idxes, i_idxes, j_idxes = result["fine_cls_idxes"]
        fine_feature0 = fine_feature0.transpose(1, 2).unflatten(2, (w, w))
        fine_feature1 = fine_feature1.transpose(1, 2).unflatten(2, (w, w))
        w = 2 * p + 1
        fine_feature0 = crop_windows(fine_feature0, w)
        fine_feature1 = crop_windows(fine_feature1, w)
        fine_feature0 = fine_feature0[m_idxes, i_idxes]
        fine_feature1 = fine_feature1[m_idxes, j_idxes]
        if reg_upsample_scale != 1:
            fine_feature0 = fine_feature0.transpose(1, 2).unflatten(2, (w, w))
            fine_feature1 = fine_feature1.transpose(1, 2).unflatten(2, (w, w))
            fine_feature0 = F.interpolate(
                fine_feature0, size=fine_reg_w, mode="bilinear",
                align_corners=True)
            fine_feature1 = F.interpolate(
                fine_feature1, size=fine_reg_w, mode="bilinear",
                align_corners=True)
        result.update(self.fine_reg_matching(fine_feature0, fine_feature1))

        result["biases0"] = result["fine_cls_biases0"].detach()
        result["biases1"] = (result["fine_cls_biases1"].detach() +
                             p * result["fine_reg_biases"].detach())
        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
