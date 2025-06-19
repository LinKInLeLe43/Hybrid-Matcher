from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .encoders import Encoder
from .tiny import tiny_roma_v1_outdoor_model


class CasP(Module):
    def __init__(
        self,
        encoder: str,
        rope: Module,
        coarse_matching: Module,
        num_coarse_matchings: int,
        extra_scale: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder)
        self.encoder.scales = (8, 4)
        self.rope = rope
        self.num_coarse_matchings = num_coarse_matchings
        self.coarse_matchings = nn.ModuleList(
            [deepcopy(coarse_matching) for _ in range(num_coarse_matchings)]
        )
        self.refiner = tiny_roma_v1_outdoor_model()
        self.extra_scale = extra_scale

        self.scales = (self.encoder.scales[0], self.encoder.scales[1])

        self.encoder.backbone.in_planes = 128
        down_module = self.encoder.backbone._make_stage(256, 4, 2)
        self.down_modules = nn.ModuleList(
            [deepcopy(down_module) for _ in range(num_coarse_matchings - 1)]
        )

    def _scale_points(
        self,
        result: Dict[str, Any],
        scale0: Optional[Tensor] = None,
        scale1: Optional[Tensor] = None,
    ) -> None:
        b_idxes = result["idxes"][0]

        coarse_points0 = self.scales[0] * result["points0"]
        coarse_points1 = self.scales[0] * result["points1"]

        if scale0 is not None and scale1 is not None:
            coarse_points0 *= scale0[b_idxes]
            coarse_points1 *= scale1[b_idxes]
        result["coarse_points0"] = coarse_points0
        result["coarse_points1"] = coarse_points1
        result["points0"], result["points1"] = coarse_points0, coarse_points1

    def forward(
        self,
        data: Dict[str, Any],
        gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        extra_gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Dict[str, Any]:
        mask0_8x, mask1_8x = data.get("mask0_8x"), data.get("mask1_8x")
        mask0_16x, mask1_16x = data.get("mask0_16x"), data.get("mask1_16x")

        x0_list, x1_list = self.encoder(data["image0"], data["image1"])

        x0_16x, x1_16x = x0_list.pop(-1), x1_list.pop(-1)
        x0_8x, x1_8x = x0_list.pop(-1), x1_list.pop(-1)
        encoding = self.rope.get_encoding()

        if self.training:
            coarse_cls_heatmap, extra_coarse_cls_heatmap = [], []
        for i in range(self.num_coarse_matchings):
            is_last = i == self.num_coarse_matchings - 1
            result = self.coarse_matchings[i](
                x0_16x,
                x1_16x,
                x0_8x,
                x1_8x,
                encoding,
                x0_mask=mask0_16x,
                x1_mask=mask1_16x,
                y0_mask=mask0_8x,
                y1_mask=mask1_8x,
                x_gt_idxes=extra_gt_idxes,
                y_gt_idxes=gt_idxes,
                only_decode=not (self.training or is_last),
            )
            if self.training:
                coarse_cls_heatmap.append(result.pop("coarse_cls_heatmap"))
                extra_coarse_cls_heatmap.append(
                    result.pop("extra_coarse_cls_heatmap")
                )

            if not is_last:
                x0_8x, x1_8x = result.pop("x_8x")
                if x0_8x.shape == x1_8x.shape:
                    out = torch.cat([x0_8x, x1_8x])
                    for module in self.down_modules[i]:
                        out = module(out)
                    x0_16x, x1_16x = out.chunk(2)
                else:
                    x0_16x, x1_16x = x0_8x, x1_8x
                    for module in self.down_modules[i]:
                        x0_16x = module(x0_16x)
                        x1_16x = module(x1_16x)

        if self.training:
            result["coarse_cls_heatmap"] = torch.stack(coarse_cls_heatmap)
            result["extra_coarse_cls_heatmap"] = torch.stack(
                extra_coarse_cls_heatmap
            )
        out = self.refiner.match(data["image0"], data["image1"], result["confidence0_to1"])
        self._scale_points(result, data.get("scale0"), data.get("scale1"))
        return result
