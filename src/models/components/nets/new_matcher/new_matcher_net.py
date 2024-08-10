from typing import Any, Dict, Optional, Tuple

import kornia as K
import torch
from torch import nn


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        positional_encoding: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        fine_module: nn.Module,
        fine_cls_matching: nn.Module,
        fine_reg_matching: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.positional_encoding = positional_encoding
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        self.fine_module = fine_module
        self.fine_cls_matching = fine_cls_matching
        self.fine_reg_matching = fine_reg_matching

        self.scales = (backbone.scales[0],
                       backbone.scales[1] // fine_preprocess.scale_before_crop)
        self.cls_w = fine_cls_matching.window_size
        self.reg_w = fine_reg_matching.window_size
        self.cls_c = fine_cls_matching.depth
        self.reg_c = fine_reg_matching.depth

        e = fine_preprocess.right_extra
        self.fine_w = fine_preprocess.window_size + 2 * e
        mask = torch.zeros((self.fine_w, self.fine_w), dtype=torch.bool)
        mask[e:-e, e:-e] = True
        mask = mask.flatten()
        self.register_buffer("fine_cls_mask", mask, persistent=False)

        delta = K.create_meshgrid(
            self.reg_w, self.reg_w, normalized_coordinates=False,
            dtype=torch.long)
        delta = delta.reshape(-1, 2)
        self.register_buffer("fine_reg_delta", delta, persistent=False)

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
        size0, size1 = x0_8x.shape[2:], x1_8x.shape[2:]

        x0_8x = self.positional_encoding(x0_8x)
        x1_8x = self.positional_encoding(x1_8x)

        x0_8x, x1_8x, x0_32x, x1_32x = map(
            lambda x: x.flatten(start_dim=2).transpose(1, 2),
            (x0_8x, x1_8x, x0_32x, x1_32x))

        x0_8x, x1_8x, m0_8x, m1_8x = self.coarse_module(
            x0_8x, x1_8x, x0_32x, x1_32x, size0, size1,
            x0_mask=mask0_8x.flatten(start_dim=1), x1_mask=mask1_8x.flatten(start_dim=1),
            center0_mask=mask0_32x.flatten(start_dim=1), center1_mask=mask1_32x.flatten(start_dim=1))

        result = self.coarse_matching(
            x0_8x, x1_8x, size0, size1, m0=m0_8x, m1=m1_8x, mask0=mask0_8x,
            mask1=mask1_8x, gt_idxes=gt_idxes)

        x0_8x = x0_8x.transpose(1, 2).unflatten(2, size0)
        x1_8x = x1_8x.transpose(1, 2).unflatten(2, size1)
        x0_1x, x1_1x = self.fine_preprocess(
            x0s + [x0_8x], x1s + [x1_8x], result["coarse_cls_idxes"])
        if len(x0_1x) != 0:
            x0_1x, x1_1x = self.fine_module(x0_1x, x1_1x)

        x0_1x, x0_reg = x0_1x.split([self.cls_c, self.reg_c], dim=2)
        x1_1x, x1_reg = x1_1x.split([self.cls_c, self.reg_c], dim=2)

        result.update(self.fine_cls_matching(
            x0_1x, x1_1x[:, self.fine_cls_mask]))

        m_idxes, sub_i_idxes, sub_j_idxes = map(
            lambda x: x[:, None], result["fine_cls_idxes"])
        sub_j_idxes = (self.fine_w *
                       (sub_j_idxes // self.cls_w + self.fine_reg_delta[:, 1]) +
                       sub_j_idxes % self.cls_w + self.fine_reg_delta[:, 0])
        x0_reg = x0_reg[m_idxes[:, 0], sub_i_idxes[:, 0]]
        x1_reg = x1_reg[m_idxes, sub_j_idxes]

        result.update(self.fine_reg_matching(x0_reg, x1_reg))

        result["biases0"] = result.pop("fine_cls_biases0")
        result["biases1"] = (result.pop("fine_cls_biases1") +
                             self.reg_w // 2 *
                             result["fine_reg_biases"].detach())
        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
