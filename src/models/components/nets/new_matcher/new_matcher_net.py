from typing import Any, Dict, Optional, Sequence, Tuple

import kornia as K
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import crop_by_mask, pad_by_mask


class NewMatcherNet(Module):
    def __init__(
        self,
        type: str,
        backbone: Module,
        rope: Module,
        local_coc: Module,
        coarse_module: Module,
        coarse_matching: Module,
        fine_preprocess: Module,
        # fine_module: Module,
        fine_cls_matching: Module,
        fine_reg_matching: Module,
        extra_scale: Optional[int] = None,
        enable_crop: bool = False,
    ) -> None:
        super().__init__()
        self.type = type
        self.backbone = backbone
        self.rope = rope
        self.local_coc = local_coc
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.fine_preprocess = fine_preprocess
        # self.fine_module = fine_module
        self.fine_cls_matching = fine_cls_matching
        self.fine_reg_matching = fine_reg_matching
        self.extra_scale = extra_scale
        self.enable_crop = enable_crop

        self.scales = (
            backbone.scales[0],
            backbone.scales[1] // fine_preprocess.upscale_before_crop,
        )
        self.reg_w = fine_reg_matching.window_size

        if type == "two_stage":
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
                self.reg_w,
                self.reg_w,
                normalized_coordinates=False,
                dtype=torch.long,
            )
            delta = delta.reshape(-1, 2)
            self.register_buffer("fine_reg_delta", delta, persistent=False)

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[Tensor] = None,
        scale1: Optional[Tensor] = None,
    ) -> None:
        m = len(result["points0"])
        b_idxes = result["idxes"][0]

        coarse_points0 = self.scales[0] * result["points0"]
        coarse_points1 = self.scales[0] * result["points1"]

        biases0 = result.pop("fine_cls_biases0")[:m]
        biases1 = result.pop("fine_cls_biases1")[:m]
        biases1 += (
            self.scales[1]
            * (self.reg_w // 2)
            * result["fine_reg_biases"][:m].detach()
        )

        fine_points0 = coarse_points0 + biases0
        fine_points1 = coarse_points1 + biases1

        if scale0 is not None and scale1 is not None:
            coarse_points0 *= scale0[b_idxes]
            fine_points0 *= scale0[b_idxes]
            coarse_points1 *= scale1[b_idxes]
            fine_points1 *= scale1[b_idxes]
        result["coarse_points0"] = coarse_points0
        result["coarse_points1"] = coarse_points1
        result["points0"], result["points1"] = fine_points0, fine_points1

    def forward(
        self,
        batch: Dict[str, Any],
        gt_indices_list: Optional[
            Sequence[Tuple[Tensor, Tensor, Tensor]]
        ] = None,
    ) -> Dict[str, Any]:
        image0, image1 = batch["image0"], batch["image1"]
        mask0, mask1 = batch.get("mask0"), batch.get("mask1")
        scale0, scale1 = batch.get("scale0"), batch.get("scale1")

        if image0.shape == image1.shape:
            x_list = self.backbone(torch.cat([image0, image1]))
            x0_list, x1_list = zip(*(x.chunk(2) for x in x_list))
        else:
            x0_list, x1_list = self.backbone(image0), self.backbone(image1)
        x0_list, x1_list = list(x0_list), list(x1_list)

        x0_8x = self.rope.abs_pe(x0_list[-1])
        x1_8x = self.rope.abs_pe(x1_list[-1])
        if self.enable_crop and mask0 is not None and mask1 is not None:
            x0_16x, x1_16x = [], []
            x0_8x_list = crop_by_mask(x0_8x, mask0)
            x1_8x_list = crop_by_mask(x1_8x, mask1)
            for b in range(len(x0_8x.shape[0])):
                b_x0_16x, b_x1_16x = self.coarse_module(
                    self.local_coc(x0_8x_list[[b]]),
                    self.local_coc(x1_8x_list[[b]]),
                    rope=self.rope,
                )
                b_x0_16x = pad_by_mask(
                    b_x0_16x, F.max_pool2d(mask0[[b]].float(), 2).bool()
                )
                b_x1_16x = pad_by_mask(
                    b_x1_16x, F.max_pool2d(mask1[[b]].float(), 2).bool()
                )
                x0_16x.append(b_x0_16x)
                x1_16x.append(b_x1_16x)
            x0_16x, x1_16x = torch.cat(x0_16x), torch.cat(x1_16x)
        else:
            x0_16x, x1_16x = self.coarse_module(
                self.local_coc(x0_8x),
                self.local_coc(x1_8x),
                rope=self.rope,
                mask0=mask0,
                mask1=mask1,
            )

        result = self.coarse_matching(
            [x0_list[-1], x0_16x],
            [x1_list[-1], x1_16x],
            mask0=mask0,
            mask1=mask1,
            gt_indices_list=gt_indices_list,
        )
        x0_list[-1], x1_list[-1] = result.pop("x_8x")

        x0_reg, x1_reg = self.fine_preprocess(
            x0_list, x1_list, result["coarse_cls_idxes"]
        )

        (s1, s2), w = self.scales, self.reg_w
        grid = K.create_meshgrid(
            s1,
            s1,
            normalized_coordinates=False,
            device=x0_reg.device,
            dtype=x0_reg.dtype,
        )
        grid = (2 * (grid + 0.5) / s1 - 1).expand(2 * len(x0_reg), -1, -1, -1)
        x = torch.cat([x0_reg, x1_reg]).transpose(1, 2).unflatten(2, (w, w))
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
        x0_cls, x1_cls = x.flatten(start_dim=2).transpose(1, 2).chunk(2)

        result.update(self.fine_cls_matching(x0_cls, x1_cls))

        local_matches = torch.cat(
            [result["fine_cls_biases0"], result["fine_cls_biases1"]], dim=1
        )
        local_matches = local_matches / s2 + w // 2
        result.update(
            self.fine_reg_matching(
                x0_reg, x1_reg, 1, local_matches=local_matches
            )
        )

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result
