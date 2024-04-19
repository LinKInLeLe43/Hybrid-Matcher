from typing import Any, Dict, Optional, Tuple

from einops import einops
import torch
from torch import nn


class LoFTRNet(nn.Module):
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

        self.scales = backbone.scales
        self.use_flow = getattr(coarse_module, "use_flow", False)
        self.coarse_matching_type = coarse_matching.type
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
        biases = result["fine_reg_biases"][:len(points0)].detach()
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
        mask0, mask1 = batch.get("mask0"), batch.get("mask1")
        if (mask0 is None) == (mask1 is not None):
            raise ValueError("")

        if batch["image0"].shape == batch["image1"].shape:
            data = torch.cat([batch["image0"], batch["image1"]])
            coarse_features, fine_features = self.backbone(data)
            coarse_feature0, coarse_feature1 = coarse_features.chunk(2)
            fine_feature0, fine_feature1 = fine_features.chunk(2)
        else:
            coarse_feature0, fine_feature0 = self.backbone(batch["image0"])
            coarse_feature1, fine_feature1 = self.backbone(batch["image1"])
        size0, size1 = coarse_feature0.shape[2:], coarse_feature1.shape[2:]

        coarse_feature0 = einops.rearrange(
            self.positional_encoding(coarse_feature0), "n c h w -> n (h w) c")
        coarse_feature1 = einops.rearrange(
            self.positional_encoding(coarse_feature1), "n c h w -> n (h w) c")

        coarse_feature0, coarse_feature1 = self.coarse_module(
            coarse_feature0, coarse_feature1, mask0=mask0, mask1=mask1)
        result = self.coarse_matching(
            coarse_feature0, coarse_feature1, size0, size1, mask0=mask0,
            mask1=mask1, gt_idxes=gt_idxes)

        fine_feature0, fine_feature1 = self.fine_preprocess(
            coarse_feature0, coarse_feature1, fine_feature0, fine_feature1,
            result["fine_idxes"], self.scales[0] // self.scales[1])
        if len(fine_feature0) != 0:
            fine_feature0, fine_feature1 = self.fine_module(
                fine_feature0, fine_feature1)
        result.update(self.fine_matching(fine_feature0, fine_feature1))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
