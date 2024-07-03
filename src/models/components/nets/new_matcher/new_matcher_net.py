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
        # positional_encoding: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        # fine_cls_matching: nn.Module,
        fine_module: nn.Module,
        fine_reg_matching: nn.Module,
        use_extra: bool = False
    ) -> None:
        super().__init__()

        self.type = type
        self.backbone = backbone
        # self.positional_encoding = positional_encoding
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        # self.fine_cls_matching = fine_cls_matching
        self.fine_module = fine_module
        self.fine_reg_matching = fine_reg_matching

        if use_extra:
            self.extra_scale = 8

        self.scales = 8, 2
        self.use_flow = False
        self.fine_reg_window_size = fine_reg_matching.window_size

        self.proj = nn.Linear(256, 128)
        self.merge = nn.Linear(256, 128)

        # p = self.fine_reg_window_size // 2
        # w = fine_cls_matching.window_size + 2 * p
        # mask = torch.zeros((w, w), dtype=torch.bool)
        # mask[p:-p, p:-p] = True
        # mask = mask.flatten()
        # self.register_buffer("fine_cls_mask", mask, persistent=False)
        #
        # delta_idxes = K.create_meshgrid(
        #     self.fine_reg_window_size, self.fine_reg_window_size,
        #     normalized_coordinates=False, dtype=torch.long)
        # delta_idxes = delta_idxes.reshape(-1, 2)
        # self.register_buffer("fine_delta_idxes", delta_idxes, persistent=False)

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
        mask0_32x, mask1_32x = batch.get("mask0_32x"), batch.get("mask1_32x")

        if batch["image0"].shape == batch["image1"].shape:
            data = torch.cat([batch["image0"], batch["image1"]])
            n, _, h, w = data.shape
            coors = K.create_meshgrid(h, w, device=data.device)
            coors = (coors / 2).permute(0, 3, 1, 2).expand(n, -1, -1, -1)
            data = torch.cat([data, coors], dim=1)
            features = self.backbone(data)
            feature0_32x, feature1_32x = features.pop(-1).chunk(2)
            feature0_8x, feature1_8x = features.pop(-2).chunk(2)
            feature0_2x, feature1_2x = features.pop(0).chunk(2)
        else:
            data = batch["image0"]
            n, _, h, w = data.shape
            coors = K.create_meshgrid(h, w, device=data.device)
            coors = (coors / 2).permute(0, 3, 1, 2).expand(n, -1, -1, -1)
            data = torch.cat([data, coors], dim=1)
            features0 = self.backbone(data)
            data = batch["image1"]
            n, _, h, w = data.shape
            coors = K.create_meshgrid(h, w, device=data.device)
            coors = (coors / 2).permute(0, 3, 1, 2).expand(n, -1, -1, -1)
            data = torch.cat([data, coors], dim=1)
            features1 = self.backbone(data)
            feature0_32x, feature1_32x = features0.pop(-1), features1.pop(-1)
            feature0_8x, feature1_8x = features0.pop(-2), features1.pop(-2)
            feature0_2x, feature1_2x = features0.pop(0), features1.pop(0)
        size0, size1 = feature0_8x.shape[2:], feature1_8x.shape[2:]

        # feature0_8x, _ = self.positional_encoding(feature0_8x)
        # feature1_8x, _ = self.positional_encoding(feature1_8x)

        feature0_8x, feature1_8x, feature0_32x, feature1_32x = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (feature0_8x, feature1_8x, feature0_32x, feature1_32x))

        (feature0_8x, feature1_8x,
         matchability0, matchability1) = self.coarse_module(
            feature0_8x, feature1_8x, feature0_32x, feature1_32x, size0, size1,
            mask0_8x=mask0_8x, mask1_8x=mask1_8x, mask0_32x=mask0_32x,
            mask1_32x=mask1_32x)

        result = self.coarse_matching(
            feature0_8x, feature1_8x, size0, size1,
            matchability0=matchability0, matchability1=matchability1,
            mask0=mask0_8x, mask1=mask1_8x, gt_idxes=gt_idxes)

        # feature0_8x = (feature0_8x.transpose(1, 2).unflatten(2, size0)
        #                .contiguous())
        # feature1_8x = (feature1_8x.transpose(1, 2).unflatten(2, size1)
        #                .contiguous())
        # aligns = [False, False, False, False]
        # if batch["image0"].shape == batch["image1"].shape:
        #     features_8x = torch.cat([feature0_8x, feature1_8x])
        #     features = self.fusion(features + [features_8x], aligns)
        #     features0, features1 = [], []
        #     for feature in features:
        #         feature0, feature1 = feature.chunk(2)
        #         features0.append(feature0)
        #         features1.append(feature1)
        # else:
        #     features0 = self.fusion(features0 + [feature0_8x], aligns)
        #     features1 = self.fusion(features1 + [feature1_8x], aligns)

        # if True:
        #     size0, size1 = features0[-2].shape[2:], features1[-2].shape[2:]
        #     features0_8x = features0[-2].flatten(start_dim=2).transpose(1, 2)
        #     features1_8x = features1[-2].flatten(start_dim=2).transpose(1, 2)
        #     use_matchability = self.coarse_matching.use_matchability
        #     self.coarse_matching.use_matchability = False
        #     result["coarse_extra_cls_heatmap"] = self.coarse_matching(
        #         features0_8x, features1_8x, size0, size1, mask0=mask0_8x,
        #         mask1=mask1_8x)["coarse_cls_heatmap"]
        #     self.coarse_matching.use_matchability = use_matchability

        # b_idxes, i_idxes, j_idxes = result["coarse_cls_idxes"]
        # fine_cls_w = self.fine_cls_matching.window_size
        # fine_reg_w = self.fine_reg_matching.window_size
        # p = fine_reg_w // 2
        # w = fine_cls_w + 2 * p
        # fine_feature0 = crop_windows(
        #     features0[0], w, stride=fine_cls_w, padding=p)
        # fine_feature1 = crop_windows(
        #     features1[0], w, stride=fine_cls_w, padding=p)
        # fine_feature0 = fine_feature0[b_idxes, i_idxes]
        # fine_feature1 = fine_feature1[b_idxes, j_idxes]
        # result.update(self.fine_cls_matching(
        #     fine_feature0[:, self.fine_cls_mask],
        #     fine_feature1[:, self.fine_cls_mask]))
        #
        # if self.fine_cls_matching.cls_topk != 1:
        #     result["idxes"] = tuple(map(
        #         lambda x: x.repeat_interleave(self.fine_cls_matching.cls_topk),
        #         result["idxes"]))
        #     result["points0"] = result["points0"].repeat_interleave(
        #         self.fine_cls_matching.cls_topk, dim=0)
        #     result["points1"] = result["points1"].repeat_interleave(
        #         self.fine_cls_matching.cls_topk, dim=0)
        #
        # m_idxes, i_idxes, j_idxes = map(
        #     lambda x: x[:, None], result["fine_cls_idxes"])
        # i_idxes = ((fine_cls_w + 2 * p) *
        #            (i_idxes // fine_cls_w + self.fine_delta_idxes[:, 1]) +
        #            i_idxes % fine_cls_w + self.fine_delta_idxes[:, 0])
        # j_idxes = ((fine_cls_w + 2 * p) *
        #            (j_idxes // fine_cls_w + self.fine_delta_idxes[:, 1]) +
        #            j_idxes % fine_cls_w + self.fine_delta_idxes[:, 0])
        # fine_feature0 = fine_feature0[m_idxes, i_idxes]
        # fine_feature1 = fine_feature1[m_idxes, j_idxes]
        # result.update(self.fine_reg_matching(fine_feature0, fine_feature1))

        w = self.fine_reg_window_size
        stride = self.scales[0] // self.scales[1]
        b_idxes, i_idxes, j_idxes = result["coarse_cls_idxes"]
        fine_feature0 = crop_windows(
            feature0_2x, w, stride=stride, padding=w // 2)
        fine_feature1 = crop_windows(
            feature1_2x, w, stride=stride, padding=w // 2)
        coarse_feature0 = feature0_8x[b_idxes, i_idxes]
        coarse_feature1 = feature1_8x[b_idxes, j_idxes]
        coarse_feature = torch.cat([coarse_feature0, coarse_feature1])
        coarse_feature = self.proj(coarse_feature)
        coarse_feature = coarse_feature[:, None].expand(-1, w ** 2, -1)
        fine_feature0 = fine_feature0[b_idxes, i_idxes]
        fine_feature1 = fine_feature1[b_idxes, j_idxes]
        fine_feature = torch.cat([fine_feature0, fine_feature1])
        feature = torch.cat([fine_feature, coarse_feature], dim=2)
        feature = self.merge(feature)
        fine_feature0, fine_feature1 = feature.chunk(2)

        if len(fine_feature0) != 0:
            fine_feature0, fine_feature1 = self.fine_module(
                fine_feature0, fine_feature1)

        result.update(self.fine_reg_matching(fine_feature0, fine_feature1))

        result["biases1"] = (self.fine_reg_window_size // 2 *
                             result["fine_reg_biases"].detach())
        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
