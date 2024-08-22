from typing import Any, Dict, Optional, Tuple

import kornia as K
import torch
from torch import nn


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        type: str,
        backbone: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        # fine_module: nn.Module,
        fine_reg_matching: nn.Module,
        extra_scale: Optional[int] = None,
        fine_cls_matching: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.type = type
        self.backbone = backbone
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        # self.fine_module = fine_module
        self.fine_reg_matching = fine_reg_matching
        self.extra_scale = extra_scale
        self.fine_cls_matching = fine_cls_matching

        self.scales = (backbone.scales[0],
                       backbone.scales[1] // fine_preprocess.scale_before_crop)
        self.reg_w = fine_reg_matching.window_size

        if type == "two_stage":
            if fine_cls_matching is None:
                raise ValueError("")

            self.cls_w = fine_cls_matching.window_size
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
        m = len(result["points0"])
        b_idxes = result["idxes"][0]

        coarse_points0 = self.scales[0] * result["points0"]
        coarse_points1 = self.scales[0] * result["points1"]

        biases0 = 0
        biases1 = (self.reg_w // 2) * result["fine_reg_biases"][:m].detach()
        if self.type == "two_stage":
            biases0 += result.pop("fine_cls_biases0")[:m]
            biases1 += result.pop("fine_cls_biases1")[:m]
        biases0 *= self.scales[1]
        biases1 *= self.scales[1]

        fine_points0 = coarse_points0 + biases0
        fine_points1 = coarse_points1 + biases1

        if scale0 is not None and scale1 is not None:
            coarse_points0 *= scale0[b_idxes]
            fine_points0 *= scale0[b_idxes]
            fine_points1 *= scale1[b_idxes]
        result["coarse_points0"] = coarse_points0
        result["points0"], result["points1"] = fine_points0, fine_points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        extra_gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        mask0_8x, mask1_8x = batch.get("mask0_8x"), batch.get("mask1_8x")
        mask0_16x, mask1_16x = batch.get("mask0_16x"), batch.get("mask1_16x")
        mask0_32x, mask1_32x = batch.get("mask0_32x"), batch.get("mask1_32x")

        if batch["image0"].shape == batch["image1"].shape:
            x = torch.cat([batch["image0"], batch["image1"]])
            xs = self.backbone(x)
            x_16x, x_32x = self.local_coc(xs[-1])

            if self.local_coc.scales[0] == 1:
                xs.pop(-1)

            x0s, x1s = [], []
            for x in xs:
                x0, x1 = x.chunk(2)
                x0s.append(x0)
                x1s.append(x1)
            x0_16x, x1_16x = x_16x.chunk(2)
            x0_32x, x1_32x = x_32x.chunk(2)
        else:
            x0s = self.backbone(batch["image0"])
            x0_16x, x0_32x = self.local_coc(x0s[-1])

            x1s = self.backbone(batch["image1"])
            x1_16x, x1_32x = self.local_coc(x1s[-1])

            if self.local_coc.scales[0] == 1:
                x0s.pop(-1)
                x1s.pop(-1)

        x0_16x, x1_16x = self.coarse_module(
            x0_16x, x1_16x, x0_32x, x1_32x, x0_mask=mask0_16x,
            x1_mask=mask1_16x, y0_mask=mask0_32x, y1_mask=mask1_32x)

        result = self.coarse_matching(
            x0s[-1], x1s[-1], x0_16x, x1_16x, x0_mask=mask0_8x,
            x1_mask=mask1_8x, y0_mask=mask0_16x, y1_mask=mask1_16x,
            x_gt_idxes=gt_idxes, y_gt_idxes=extra_gt_idxes)
        x0s[-1], x1s[-1] = result.pop("x_8x")

        x0_1x, x1_1x = self.fine_preprocess(
            x0s, x1s, result["coarse_cls_idxes"])

        # if self.type == "one_stage":
        #     if len(x0_1x) != 0:
        #         x0_1x, x1_1x = self.fine_module(x0_1x, x1_1x)
        # elif self.type == "two_stage":
        #     if len(x0_1x) != 0:
        #         w0, w1 = self.cls_w, self.fine_w
        #         x0_1x, x1_1x = self.fine_module(
        #             x0_1x, x1_1x, size0=(w0, w0), size1=(w1, w1))
        #
        #     x0_1x, x0_reg = x0_1x.split([self.cls_c, self.reg_c], dim=2)
        #     x1_1x, x1_reg = x1_1x.split([self.cls_c, self.reg_c], dim=2)
        #
        #     result.update(self.fine_cls_matching(
        #         x0_1x, x1_1x[:, self.fine_cls_mask]))
        #
        #     m_idxes, sub_i_idxes, sub_j_idxes = map(
        #         lambda x: x[:, None], result["fine_cls_idxes"])
        #     sub_j_idxes = (
        #         self.fine_w *
        #         (sub_j_idxes // self.cls_w + self.fine_reg_delta[:, 1]) +
        #         sub_j_idxes % self.cls_w + self.fine_reg_delta[:, 0])
        #     x0_1x = x0_reg[m_idxes[:, 0], sub_i_idxes[:, 0]]
        #     x1_1x = x1_reg[m_idxes, sub_j_idxes]
        # else:
        #     assert False

        result.update(self.fine_reg_matching(x0_1x, x1_1x, 3))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
