from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import kornia as K

from .modules.utils import crop_windows


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        type: str,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        coarse_module: nn.Module,
        coarse_matching_16x: nn.Module,
        coarse_matching_8x: nn.Module,
        fine_cls_matching_2x: nn.Module,
        fine_cls_matching_1x: nn.Module,
        fine_reg_matching: nn.Module
    ) -> None:
        super().__init__()
        self.type = type
        self.backbone, self.fusion = backbone
        self.positional_encoding = positional_encoding
        self.coarse_module = coarse_module
        self.coarse_matching_16x = coarse_matching_16x
        self.coarse_matching_8x = coarse_matching_8x
        self.fine_cls_matching_2x = fine_cls_matching_2x
        self.fine_cls_matching_1x = fine_cls_matching_1x
        self.fine_reg_matching = fine_reg_matching

        self.scales = 16, 2, 1
        self.extra_scale = 8
        self.use_flow = False
        self.fine_reg_window_size = fine_reg_matching.window_size

        p = self.fine_reg_window_size // 2
        w = fine_cls_matching_1x.window_size + 2 * p
        mask = torch.zeros((w, w), dtype=torch.bool)
        mask[p:-p, p:-p] = True
        mask = mask.flatten()
        self.register_buffer("fine_cls_1x_mask", mask, persistent=False)

        idxes = K.create_meshgrid(
            self.fine_reg_window_size, self.fine_reg_window_size,
            normalized_coordinates=False, dtype=torch.long)
        idxes = idxes.reshape(-1, 2)
        self.register_buffer("fine_reg_idxes", idxes, persistent=False)

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None
    ) -> None:
        m = len(result["points0"])
        b_idxes = result["idxes"][0]

        coarse_points0 = self.scales[0] * result["points0"]
        fine_points0 = coarse_points0.clone()
        if "biases0" in result:
            fine_points0 += self.scales[2] * result["biases0"][:m]
        if scale0 is not None:
            coarse_points0 *= scale0[b_idxes]
            fine_points0 *= scale0[b_idxes]
        result["coarse_points0"] = coarse_points0  # for evaluate coarse precision
        result["points0"] = fine_points0

        coarse_points1 = self.scales[0] * result["points1"]
        fine_points1 = coarse_points1.clone()
        if "biases1" in result:
            fine_points1 += self.scales[2] * result["biases1"][:m]
        if scale1 is not None:
            fine_points1 *= scale1[b_idxes]
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
        result = {}

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
        size0_16x, size1_16x = feature0_16x.shape[2:], feature1_16x.shape[2:]

        feature0_16x, _ = self.positional_encoding(feature0_16x)
        feature1_16x, _ = self.positional_encoding(feature1_16x)

        feature0_16x, feature1_16x, feature0_32x, feature1_32x = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (feature0_16x, feature1_16x, feature0_32x, feature1_32x))

        (feature0_16x, feature1_16x,
         matchability0, matchability1) = self.coarse_module(
            feature0_16x, feature1_16x, feature0_32x, feature1_32x, size0_16x,
            size1_16x, mask0_16x=mask0_16x, mask1_16x=mask1_16x,
            mask0_32x=mask0_32x, mask1_32x=mask1_32x)

        result_16x = self.coarse_matching_16x(
            feature0_16x, feature1_16x, size0_16x, size1_16x,
            matchability0=matchability0, matchability1=matchability1,
            mask0=mask0_16x, mask1=mask1_16x, only_return_mask=True)
        result["coarse_cls_heatmap_16x"] = result_16x["coarse_cls_heatmap"]

        feature0_16x = feature0_16x.transpose(1, 2).unflatten(2, size0_16x)
        feature1_16x = feature1_16x.transpose(1, 2).unflatten(2, size1_16x)
        aligns = [True, False, False, False]
        if batch["image0"].shape == batch["image1"].shape:
            features_16x = torch.cat([feature0_16x, feature1_16x])
            features = self.fusion(features + [features_16x], aligns)
            features0, features1 = [], []
            for feature in features:
                feature0, feature1 = feature.chunk(2)
                features0.append(feature0)
                features1.append(feature1)
        else:
            features0 = self.fusion(features0 + [feature0_16x], aligns)
            features1 = self.fusion(features1 + [feature1_16x], aligns)

        size0_8x, size1_8x = features0[-2].shape[2:], features1[-2].shape[2:]
        features0_8x = features0[-2].flatten(start_dim=2).transpose(1, 2)
        features1_8x = features1[-2].flatten(start_dim=2).transpose(1, 2)
        prior_mask = None
        if not self.training:
            prior_mask = result_16x["coarse_cls_mask"].reshape(
                -1, size0_16x[0], size0_16x[1], size1_16x[0], size1_16x[1])
            for i in range(1, 5):
                prior_mask = prior_mask.repeat_interleave(2, dim=i)
            prior_mask = prior_mask.reshape(
                -1, size0_8x[0] * size0_8x[1], size1_8x[0] * size1_8x[1])
        result_8x = self.coarse_matching_8x(
            features0_8x, features1_8x, size0_8x, size1_8x,
            prior_mask=prior_mask, mask0=mask0_8x, mask1=mask1_8x,
            gt_idxes=gt_idxes)
        if self.training:
            result["coarse_cls_heatmap_8x"] = result_8x.pop(
                "coarse_cls_heatmap")
        result["coarse_cls_idxes_16x"] = result_8x.pop("coarse_cls_idxes")
        coarse_cls_sub_idxes = result_8x.pop("coarse_cls_sub_idxes")
        result.update(result_8x)

        size0_2x, size1_2x = features0[1].shape[2:], features1[1].shape[2:]
        b_idxes, i_idxes, j_idxes = result["coarse_cls_idxes_16x"]
        cls_w_2x = self.fine_cls_matching_2x.window_size
        cls_w_1x = self.fine_cls_matching_1x.window_size
        reg_w = self.fine_reg_matching.window_size
        p = reg_w // 2
        w = cls_w_1x + 2 * p
        feature0_1x, feature1_1x = features0[0], features1[0]
        feature0_2x, feature1_2x = features0[1], features1[1]
        feature0_1x = crop_windows(
            feature0_1x, w, stride=2, padding=w // 2, enable_rearrange=False)
        feature1_1x = crop_windows(
            feature1_1x, w, stride=2, padding=w // 2, enable_rearrange=False)
        feature0_1x = feature0_1x.unflatten(2, size0_2x)
        feature1_1x = feature1_1x.unflatten(2, size1_2x)
        feature0_2x_1x = torch.cat([feature0_2x, feature0_1x], dim=1)
        feature1_2x_1x = torch.cat([feature1_2x, feature1_1x], dim=1)
        feature0_2x_1x = crop_windows(feature0_2x_1x, cls_w_2x, stride=cls_w_2x)
        feature1_2x_1x = crop_windows(feature1_2x_1x, cls_w_2x, stride=cls_w_2x)
        feature0_2x_1x = feature0_2x_1x[b_idxes, i_idxes]
        feature1_2x_1x = feature1_2x_1x[b_idxes, j_idxes]

        c0, c1 = features0[0].shape[1], features0[1].shape[1]
        feature0_2x, feature0_1x = feature0_2x_1x.split(
            [c0, c1 * w ** 2], dim=2)
        feature1_2x, feature1_1x = feature1_2x_1x.split(
            [c0, c1 * w ** 2], dim=2)
        result_2x = self.fine_cls_matching_2x(
            feature0_2x, feature1_2x, sub_idxes=coarse_cls_sub_idxes)
        result["fine_cls_heatmap_2x"] = result_2x.pop("fine_cls_heatmap")
        m_idxes, fine_i_idxes, fine_j_idxes = result_2x.pop("fine_cls_idxes")
        i_idxes = (
            size0_2x[1] * (8 * (i_idxes // size0_16x[1]) + fine_i_idxes // 8) +
            8 * (i_idxes % size0_16x[1]) + fine_i_idxes % 8)
        j_idxes = (
            size1_2x[1] * (8 * (j_idxes // size1_16x[1]) + fine_j_idxes // 8) +
            8 * (j_idxes % size1_16x[1]) + fine_j_idxes % 8)
        result["coarse_cls_idxes_2x"] = b_idxes, i_idxes, j_idxes

        feature0_1x = feature0_1x[m_idxes, fine_i_idxes]
        feature1_1x = feature1_1x[m_idxes, fine_j_idxes]
        feature0_1x = feature0_1x.unflatten(1, (c1, w ** 2)).transpose(1, 2)
        feature1_1x = feature1_1x.unflatten(1, (c1, w ** 2)).transpose(1, 2)
        result_1x = self.fine_cls_matching_1x(
            feature0_1x[:, self.fine_cls_1x_mask],
            feature1_1x[:, self.fine_cls_1x_mask])
        result["fine_cls_heatmap_1x"] = result_1x.pop("fine_cls_heatmap")
        result["fine_cls_idxes_1x"] = result_1x.pop("fine_cls_idxes")

        m_idxes, fine_i_idxes, fine_j_idxes = map(
            lambda x: x[:, None], result["fine_cls_idxes_1x"])
        fine_i_idxes = (
            w * (fine_i_idxes // cls_w_1x + self.fine_reg_idxes[:, 1]) +
            fine_i_idxes % cls_w_1x + self.fine_reg_idxes[:, 0])
        fine_j_idxes = (
            w * (fine_j_idxes // cls_w_1x + self.fine_reg_idxes[:, 1]) +
            fine_j_idxes % cls_w_1x + self.fine_reg_idxes[:, 0])
        feature0_1x = feature0_1x[m_idxes, fine_i_idxes]
        feature1_1x = feature1_1x[m_idxes, fine_j_idxes]
        result_sub = self.fine_reg_matching(feature0_1x, feature1_1x)
        result.update(result_sub)

        result["biases0"] = (2 * result_2x["fine_cls_biases0"].detach() +
                             result_1x["fine_cls_biases0"].detach())
        result["biases1"] = (2 * result_2x["fine_cls_biases1"].detach() +
                             result_1x["fine_cls_biases1"].detach() +
                             result_sub["fine_reg_biases"].detach())
        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
