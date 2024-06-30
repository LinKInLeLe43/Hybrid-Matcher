from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
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
        fine_module: nn.Module,
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
        self.fine_module = fine_module
        self.fine_cls_matching_1x = fine_cls_matching_1x
        self.fine_reg_matching = fine_reg_matching

        self.scales = 16, 2, 1
        self.extra_scale = 8
        self.use_flow = False

        self.cls_1x_window_size = fine_cls_matching_1x.window_size
        self.reg_window_size = fine_reg_matching.window_size
        assert fine_cls_matching_2x.window_size == 8
        assert self.cls_1x_window_size % 2 == 1
        assert self.reg_window_size % 2 == 1
        radius0 = self.cls_1x_window_size // 2
        radius1 = radius0 + self.reg_window_size // 2

        self.p0_2x = (radius0 + 1) // 2
        self.w0_2x = 8 + 2 * self.p0_2x
        mask0_2x = torch.zeros((self.w0_2x, self.w0_2x), dtype=torch.bool)
        mask0_2x[self.p0_2x:-self.p0_2x, self.p0_2x:-self.p0_2x] = True
        mask0_2x = mask0_2x.flatten()
        self.register_buffer("cls_mask0_2x", mask0_2x, persistent=False)

        self.p1_2x = (radius1 + 1) // 2
        self.w1_2x = 8 + 2 * self.p1_2x
        mask1_2x = torch.zeros((self.w1_2x, self.w1_2x), dtype=torch.bool)
        mask1_2x[self.p1_2x:-self.p1_2x, self.p1_2x:-self.p1_2x] = True
        mask1_2x = mask1_2x.flatten()
        self.register_buffer("cls_mask1_2x", mask1_2x, persistent=False)

        self.p0_1x = 0
        self.w0_1x = self.cls_1x_window_size

        self.p1_1x = self.reg_window_size // 2
        self.w1_1x = self.cls_1x_window_size + 2 * self.p1_1x
        mask1_1x = torch.zeros((self.w1_1x, self.w1_1x), dtype=torch.bool)
        mask1_1x[self.p1_1x:-self.p1_1x, self.p1_1x:-self.p1_1x] = True
        mask1_1x = mask1_1x.flatten()
        self.register_buffer("cls_mask1_1x", mask1_1x, persistent=False)

        reg_delta1 = K.create_meshgrid(
            self.reg_window_size, self.reg_window_size,
            normalized_coordinates=False, dtype=torch.long)
        reg_delta1 = reg_delta1.reshape(-1, 2)
        self.register_buffer("reg_delta1", reg_delta1, persistent=False)

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
            fine_points0 += self.scales[-1] * result["biases0"][:m]
        if scale0 is not None:
            coarse_points0 *= scale0[b_idxes]
            fine_points0 *= scale0[b_idxes]
        result["coarse_points0"] = coarse_points0  # for evaluate coarse precision
        result["points0"] = fine_points0

        coarse_points1 = self.scales[0] * result["points1"]
        fine_points1 = coarse_points1.clone()
        if "biases1" in result:
            fine_points1 += self.scales[-1] * result["biases1"][:m]
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
            mask0=mask0_16x, mask1=mask1_16x, gt_idxes=gt_idxes)
        result["coarse_cls_heatmap_16x"] = result_16x.pop("coarse_cls_heatmap")
        result["coarse_cls_idxes_16x"] = result_16x.pop("coarse_cls_idxes")
        result.update(result_16x)

        feature0_16x = feature0_16x.transpose(1, 2).unflatten(2, size0_16x)
        feature1_16x = feature1_16x.transpose(1, 2).unflatten(2, size1_16x)
        aligns = [False, False, False, False]
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

        if True:
            size0_8x = features0[-2].shape[2:]
            size1_8x = features1[-2].shape[2:]
            features0_8x = features0[-2].flatten(start_dim=2).transpose(1, 2)
            features1_8x = features1[-2].flatten(start_dim=2).transpose(1, 2)
            result_8x = self.coarse_matching_8x(
                features0_8x, features1_8x, size0_8x, size1_8x, mask0=mask0_8x,
                mask1=mask1_8x)
            result["coarse_cls_heatmap_8x"] = result_8x.pop(
                "coarse_cls_heatmap")

        w = 8
        b_idxes, i_idxes, j_idxes = result["coarse_cls_idxes_16x"]
        feature0_2x, feature1_2x = features0[0], features1[0]
        feature0_2x = crop_windows(
            feature0_2x, self.w0_2x, stride=w, padding=self.p0_2x)
        feature1_2x = crop_windows(
            feature1_2x, self.w1_2x, stride=w, padding=self.p1_2x)
        feature0_2x = feature0_2x[b_idxes, i_idxes]
        feature1_2x = feature1_2x[b_idxes, j_idxes]
        result_2x = self.fine_cls_matching_2x(
            feature0_2x[:, self.cls_mask0_2x],
            feature1_2x[:, self.cls_mask1_2x])
        result["fine_cls_heatmap_2x"] = result_2x.pop("fine_cls_heatmap")
        cls_idxes_2x = result_2x.pop("fine_cls_idxes")

        _, sub_i_idxes, sub_j_idxes = cls_idxes_2x
        size0_2x, size1_2x = features0[0].shape[2:], features1[0].shape[2:]
        offset = w // 2
        i_idxes = (
            size0_2x[1] *
            (w * (i_idxes // size0_16x[1]) + sub_i_idxes // w - offset) +
            w * (i_idxes % size0_16x[1]) + sub_i_idxes % w - offset)
        j_idxes = (
            size1_2x[1] *
            (w * (j_idxes // size1_16x[1]) + sub_j_idxes // w - offset) +
            w * (j_idxes % size1_16x[1]) + sub_j_idxes % w - offset)
        result["coarse_cls_idxes_2x"] = b_idxes, i_idxes, j_idxes

        topk = self.fine_cls_matching_2x.cls_topk
        if topk != 1:
            result["idxes"] = tuple(map(
                lambda x: x.repeat_interleave(topk), result["idxes"]))
            result["points0"] = result["points0"].repeat_interleave(
                self.fine_cls_matching.cls_topk, dim=0)
            result["points1"] = result["points1"].repeat_interleave(
                self.fine_cls_matching.cls_topk, dim=0)

        w = self.cls_1x_window_size
        feature0_1x = F.interpolate(
            features0[0], size=(2 * size0_2x[0] - 1, 2 * size0_2x[1] - 1),
            mode="bilinear", align_corners=True)
        feature1_1x = F.interpolate(
            features1[0], size=(2 * size1_2x[0] - 1, 2 * size1_2x[1] - 1),
            mode="bilinear", align_corners=True)
        feature0_1x = crop_windows(
            feature0_1x, self.w0_1x, stride=2, padding=self.p0_1x)
        feature1_1x = crop_windows(
            feature1_1x, self.w1_1x, stride=2, padding=self.p1_1x)
        feature0_1x = feature0_1x[b_idxes, i_idxes]
        feature1_1x = feature1_1x[b_idxes, j_idxes]

        tr_feature0_1x = feature0_1x
        tr_feature1_1x = feature1_1x[:, self.cls_mask1_1x]
        if len(feature0_1x) != 0:
            tr_feature0_1x, tr_feature1_1x = self.fine_module(
                tr_feature0_1x, tr_feature1_1x)

        result_1x = self.fine_cls_matching_1x(
            tr_feature0_1x, tr_feature1_1x)
        result["fine_cls_heatmap_1x"] = result_1x.pop("fine_cls_heatmap")
        result["cls_idxes_1x"] = result_1x.pop("fine_cls_idxes")

        m_idxes, sub_i_idxes, sub_j_idxes = result["cls_idxes_1x"]
        sub_j_idxes = sub_j_idxes[:, None]
        sub_j_idxes = (self.w1_1x * (sub_j_idxes // w + self.reg_delta1[:, 1]) +
                       sub_j_idxes % w + self.reg_delta1[:, 0])
        reg_feature0 = feature0_1x[m_idxes, sub_i_idxes]
        reg_feature1 = feature1_1x[m_idxes[:, None], sub_j_idxes]
        reg_result = self.fine_reg_matching(reg_feature0, reg_feature1)
        result.update(reg_result)

        result["biases0"] = (2 * result_2x["fine_cls_biases0"].detach() +
                             result_1x["fine_cls_biases0"].detach())
        result["biases1"] = (2 * result_2x["fine_cls_biases1"].detach() +
                             result_1x["fine_cls_biases1"].detach() +
                             reg_result["fine_reg_biases"].detach())
        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
