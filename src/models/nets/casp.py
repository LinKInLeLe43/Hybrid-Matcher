from typing import Any, Dict, Optional, Tuple

from torch import Tensor
from torch.nn import Module

from .encoders import Encoder


class CasP(Module):
    def __init__(
        self,
        encoder: str,
        rope: Module,
        coarse_module: Module,
        coarse_matching: Module,
        extra_scale: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder)
        self.encoder.scales = (8, 4)
        self.rope = rope
        self.coarse_module = coarse_module
        self.coarse_matching = coarse_matching
        self.extra_scale = extra_scale

        self.scales = (self.encoder.scales[0], self.encoder.scales[1])

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
        encoding = self.rope.get_encoding()
        x0_16x_t, x1_16x_t = self.coarse_module(
            x0_16x, x1_16x, encoding, mask0=mask0_16x, mask1=mask1_16x
        )

        x0_8x, x1_8x = x0_list.pop(-1), x1_list.pop(-1)
        result = self.coarse_matching(
            x0_8x,
            x1_8x,
            x0_16x_t,
            x1_16x_t,
            x0_mask=mask0_8x,
            x1_mask=mask1_8x,
            y0_mask=mask0_16x,
            y1_mask=mask1_16x,
            x_gt_idxes=gt_idxes,
            y_gt_idxes=extra_gt_idxes,
        )

        self._scale_points(result, data.get("scale0"), data.get("scale1"))
        return result
