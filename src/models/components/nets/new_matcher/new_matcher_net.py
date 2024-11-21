from typing import Any, Dict, List, Optional, Tuple

import kornia as K
import torch
from torch import nn
from torch.nn import functional as F


class NewMatcherNet(nn.Module):
    def __init__(
        self,
        type: str,
        backbone: nn.Module,
        rope: nn.Module,
        local_coc: nn.Module,
        coarse_module: nn.Module,
        coarse_matching: nn.Module,
        fine_preprocess: nn.Module,
        # fine_module: nn.Module,
        fine_cls_matching: nn.Module,
        fine_reg_matching: nn.Module,
        extra_scale: Optional[int] = None,
        enable_crop: bool = False
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

        self.scales = (backbone.scales[0],
                       backbone.scales[1] // fine_preprocess.scale_before_crop)
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

        biases0 = result.pop("fine_cls_biases0")[:m]
        biases1 = result.pop("fine_cls_biases1")[:m]
        biases1 += (self.scales[1] * (self.reg_w // 2) *
                    result["fine_reg_biases"][:m].detach())

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
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        extra_gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        image0, image1 = batch["image0"], batch["image1"]
        mask0, mask1 = batch.get("mask0"), batch.get("mask1")

        if image0.shape == image1.shape:
            image = torch.cat([image0, image1])
            mask = None
            if mask0 is not None and mask1 is not None:
                mask = torch.cat([mask0, mask1])
            n = len(image) // 2
            xs = self.backbone(image)

            x_8x = self.rope.abs_pe(xs[-1])

            if self.enable_crop and mask0 is not None and mask1 is not None:
                x0_16x, x1_16x = [], []
                for b in range(n):
                    b_x0_16x, b_x0_32x = self.local_coc(
                        crop_with_mask(x_8x[b + 0], mask[b + 0])[None]
                    )
                    b_x1_16x, b_x1_32x = self.local_coc(
                        crop_with_mask(x_8x[b + n], mask[b + n])[None]
                    )

                    b_x0_16x, b_x1_16x = self.coarse_module(
                        b_x0_16x, b_x1_16x, b_x0_32x, b_x1_32x, rope=self.rope)

                    x0_16x.append(
                        pad_with_mask(
                            b_x0_16x[0],
                            F.max_pool2d(
                                mask[[b + 0]].float(), 2, stride=2
                            )[0].bool()
                        )
                    )
                    x1_16x.append(
                        pad_with_mask(
                            b_x1_16x[0],
                            F.max_pool2d(
                                mask[[b + n]].float(), 2, stride=2
                            )[0].bool()
                        )
                    )
                x0_16x, x1_16x = torch.stack(x0_16x), torch.stack(x1_16x)

                result = self.coarse_matching(
                    xs[-1][:n], xs[-1][n:], x0_16x, x1_16x, mask0=mask0, mask1=mask1,
                    x_gt_idxes=gt_idxes, y_gt_idxes=extra_gt_idxes)
                xs[-1] = torch.cat(result.pop("x_8x"))
            else:
                if x0_8x.shape == x1_8x.shape:
                    x_8x = torch.cat([x0_8x, x1_8x])
                    x_16x, x_32x = self.local_coc(x_8x)
                    x0_16x, x1_16x = x_16x.chunk(2)
                    x0_32x, x1_32x = x_32x.chunk(2)
                else:
                    x0_16x, x0_32x = self.local_coc(x0_8x)
                    x1_16x, x1_32x = self.local_coc(x1_8x)

                x0_16x, x1_16x = self.coarse_module(
                    x0_16x, x1_16x, x0_32x, x1_32x, rope=self.rope,
                    mask0=mask0, mask1=mask1)
        else:
            x0s = self.backbone(image0)
            x1s = self.backbone(image1)

            x0_8x = self.rope.abs_pe(x0s[-1])
            x1_8x = self.rope.abs_pe(x1s[-1])

        idxes = result["coarse_cls_idxes"]
        x_2x = self.fine_preprocess(
            xs,
            (
                torch.cat([idxes[0], n + idxes[0]]),
                torch.cat([idxes[1], idxes[2]])
            )
        )

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

        m = len(x_2x) // 2
        (s1, s2), w = self.scales, self.reg_w
        grid = K.create_meshgrid(
            s1, s1, normalized_coordinates=False, device=x_2x.device,
            dtype=x_2x.dtype)
        grid = (2 * (grid + 0.5) / s1 - 1).expand(2 * m, -1, -1, -1)
        x = x_2x.transpose(1, 2).unflatten(2, (w, w))
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
        x0_cls, x1_cls = x.flatten(start_dim=2).transpose(1, 2).chunk(2)

        result.update(self.fine_cls_matching(x0_cls, x1_cls))

        local_matches = torch.cat([result["fine_cls_biases0"],
                                   result["fine_cls_biases1"]], dim=1)
        local_matches = local_matches / s2 + w // 2
        result.update(self.fine_reg_matching(
            x_2x[:m], x_2x[m:], 1, local_matches=local_matches))

        self._scale_points(result, batch.get("scale0"), batch.get("scale1"))
        return result


def crop_with_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h, w = mask.sum(dim=0).amax().item(), mask.sum(dim=1).amax().item()
    out = x[:, :h, :w]
    return out


def pad_with_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    c, h, w = x.shape
    out = x.new_zeros((c, *mask.shape))
    out[:, :h, :w] = x
    return out
