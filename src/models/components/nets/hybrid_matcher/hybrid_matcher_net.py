from typing import Any, Dict, Optional, Tuple

import kornia as K
import torch
from torch import nn


class HybridMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_module: nn.Module,
        fine_matching: nn.Module,
        positional_encoding: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_module = fine_module
        self.fine_matching = fine_matching

        self.positional_encoding = None
        if coarse_module.use_flow:
            if positional_encoding is None:
                raise ValueError("")
            self.positional_encoding = positional_encoding

        self.scales = backbone.scales
        self.window_size = fine_preprocess.window_size

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None
    ) -> None:
        points0 = self.scales[0] * result["points0"]
        if scale0 is not None:
            points0 *= scale0[result["b_idxes"]]

        points1 = self.scales[0] * result["points1"]
        biases = result["fine_biases"][:len(points0)].detach()
        biases = self.scales[1] * (self.window_size // 2) * biases
        if scale1 is not None:
            points1 *= scale1[result["b_idxes"]]
            biases *= scale1[result["b_idxes"]]
        points1 += biases
        result["points0"], result["points1"] = points0, points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        device = batch["image0"].device
        mask0, mask1 = batch.get("mask0"), batch.get("mask1")
        if (mask0 is None) == (mask1 is not None):
            raise ValueError("")

        pos_feature0 = pos_feature1 = None
        if batch["image0"].shape == batch["image1"].shape:
            n, _, h, w = batch["image0"].shape
            coors = K.create_meshgrid(h, w, device=device)
            coors = (coors / 2).permute(0, 3, 1, 2)
            data = torch.cat([batch["image0"], batch["image1"]])
            data = torch.cat([data, coors.repeat(2 * n, 1, 1, 1)], dim=1)
            coarse_features, fine_features = self.backbone(data)
            mask = torch.cat([mask0, mask1])
            centers, coarse_features = self.local_coc(
                coarse_features, mask=mask)
            centers0, centers1 = centers.chunk(2)
            coarse_feature0, coarse_feature1 = coarse_features.chunk(2)
            fine_feature0, fine_feature1 = fine_features.chunk(2)
            if self.positional_encoding is not None:
                pos_features = self.positional_encoding.get(
                    coarse_features).repeat(2 * n, 1, 1, 1)
                pos_features = pos_features.flatten(start_dim=2).transpose(1, 2)
                pos_feature0, pos_feature1 = pos_features.chunk(2)
        else:
            n, _, h, w = batch["image0"].shape
            coors = K.create_meshgrid(h, w, device=device)
            coors = (coors / 2).permute(0, 3, 1, 2)
            data = torch.cat([batch["image0"],
                              coors.repeat(n, 1, 1, 1)], dim=1)
            coarse_feature0, fine_feature0 = self.backbone(data)
            centers0, coarse_feature0 = self.local_coc(
                coarse_feature0, mask=mask0)

            n, _, h, w = batch["image1"].shape
            coors = K.create_meshgrid(h, w, device=device)
            coors = (coors / 2).permute(0, 3, 1, 2)
            data = torch.cat([batch["image1"],
                              coors.repeat(n, 1, 1, 1)], dim=1)
            coarse_feature1, fine_feature1 = self.backbone(data)
            centers1, coarse_feature1 = self.local_coc(
                coarse_feature1, mask=mask1)

            if self.positional_encoding is not None:
                pos_feature0 = self.positional_encoding.get(
                    coarse_feature0).repeat(n, 1, 1, 1)
                pos_feature0 = pos_feature0.flatten(start_dim=2).transpose(1, 2)
                pos_feature1 = self.positional_encoding.get(
                    coarse_feature1).repeat(n, 1, 1, 1)
                pos_feature1 = pos_feature1.flatten(start_dim=2).transpose(1, 2)
        size0, size1 = coarse_feature0.shape[2:], coarse_feature1.shape[2:]

        coarse_feature0, coarse_feature1, centers0, centers1 = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (coarse_feature0, coarse_feature1, centers0, centers1))

        coarse_feature0, coarse_feature1, flow0, flow1 = self.coarse_module(
            coarse_feature0, coarse_feature1, centers0, centers1, size0, size1,
            pos0=pos_feature0, pos1=pos_feature1, mask0=mask0, mask1=mask1)

        result = self.coarse_matching(
            coarse_feature0, coarse_feature1, size0, size1, flow0=flow0,
            flow1=flow1, mask0=mask0, mask1=mask1, gt_idxes=gt_idxes)

        fine_feature0, fine_feature1 = self.fine_preprocess(
            coarse_feature0, coarse_feature1, fine_feature0, fine_feature1,
            result["fine_idxes"], self.scales[0] // self.scales[1])
        if len(fine_feature0) != 0:
            fine_feature0, fine_feature1 = self.fine_module(
                fine_feature0, fine_feature1)
        result.update(self.fine_matching(fine_feature0, fine_feature1))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
