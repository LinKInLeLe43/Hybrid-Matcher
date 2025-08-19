from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .encoders import Encoder
from .fine_matching import FineMatching


class Conv2d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


class CasP_O(Module):
    def __init__(
        self,
        encoder: str,
        rope: Module,
        coarse_matching: Module,
        num_coarse_matchings: int,
        extra_scale: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder, [64, 128, 256], [2, 4, 14])
        self.encoder.scales = (8, 4)
        self.rope = rope
        self.num_coarse_matchings = num_coarse_matchings
        self.coarse_matchings = nn.ModuleList(
            [deepcopy(coarse_matching) for _ in range(num_coarse_matchings)]
        )
        self.fine_matching = FineMatching()
        self.extra_scale = extra_scale

        self.scales = (self.encoder.scales[0], self.encoder.scales[1])

        self.encoder.backbone.in_planes = 128
        down_module = self.encoder.backbone._make_stage(256, 4, 2)
        self.down_modules = nn.ModuleList(
            [deepcopy(down_module) for _ in range(num_coarse_matchings - 1)]
        )

        self.drop = None
        self.block_dims = [128, 256]
        self.fc16 = Conv2d_BN_Act(
            self.block_dims[-1], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc8 = Conv2d_BN_Act(
            self.block_dims[-2], self.block_dims[-1], 1, drop=self.drop
        )
        self.att16 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.dwconv8 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )

    def update_points(
        self, data: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        scale_coarse = self.scales[0]
        # scale_fine = self.scales[1] * (self.fine_reg_matching.window_size // 2)
        w0 = data["image0"].shape[-1] // scale_coarse
        w1 = data["image1"].shape[-1] // scale_coarse
        b_indices, i_indices, j_indices = results["idxes"]

        points0 = torch.stack([i_indices % w0, i_indices // w0], dim=-1).float()
        points1 = torch.stack([j_indices % w1, j_indices // w1], dim=-1).float()
        coarse_points0 = points0 * scale_coarse
        coarse_points1 = points1 * scale_coarse
        fine_points0 = coarse_points0 + results["fine_reg_biases0"]
        fine_points1 = coarse_points1 + results["fine_reg_biases1"]
        if "scale0" in data and "scale1" in data:
            coarse_points0 = coarse_points0 * data["scale0"][b_indices]
            coarse_points1 = coarse_points1 * data["scale1"][b_indices]
            fine_points0 = fine_points0 * data["scale0"][b_indices]
            fine_points1 = fine_points1 * data["scale1"][b_indices]
        results["coarse_points0"] = coarse_points0
        results["coarse_points1"] = coarse_points1
        results["points0"], results["points1"] = fine_points0, fine_points1

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
        x0_8x_ori, x1_8x_ori = x0_8x, x1_8x = x0_list.pop(-1), x1_list.pop(-1)
        encoding = self.rope.get_encoding()

        if self.training:
            coarse_cls_heatmap = []
        for i in range(self.num_coarse_matchings):
            is_last = i == self.num_coarse_matchings - 1
            results = self.coarse_matchings[i](
                x0_16x,
                x1_16x,
                x0_8x_ori,
                x1_8x_ori,
                encoding,
                x0_mask=mask0_16x,
                x1_mask=mask1_16x,
                y0_mask=mask0_8x,
                y1_mask=mask1_8x,
                x_gt_idxes=extra_gt_idxes,
                y_gt_idxes=gt_idxes,
                only_decode=not (self.training or is_last),
            )
            (x0_8x, x0_16x), (x1_8x, x1_16x) = results.pop("x")
            if self.training:
                coarse_cls_heatmap.append(results.pop("coarse_cls_heatmap"))

            if not is_last:
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
            results["coarse_cls_heatmap"] = torch.stack(coarse_cls_heatmap)

        f8, f16 = torch.cat([x0_8x, x1_8x]), torch.cat([x0_16x, x1_16x])

        f16 = self.fc16(f16)
        f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
        att16_up = F.interpolate(
            self.att16(f16), scale_factor=2.0, mode="bilinear"
        )
        f8 = self.fc8(f8)
        f8 = self.dwconv8(f8 * att16_up + f16_up)

        x0_8x, x1_8x = f8.chunk(2)

        results.update(
            self.fine_matching(
                x0_8x_ori, x1_8x_ori, x0_8x, x1_8x, results["coarse_cls_idxes"]
            )
        )

        self.update_points(data, results)
        return results

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k in list(state_dict.keys()):
            if k.startswith("net."):
                new_k = k.replace("net.", "", 1)
                state_dict[new_k] = state_dict.pop(k)
        return super().load_state_dict(state_dict)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/model/net/casp_o.yaml").config
    net = CasP_O(config).eval()
    mask0 = torch.zeros(1, 832 // 8, 832 // 8, dtype=torch.bool)
    mask1 = torch.zeros(1, 832 // 8, 832 // 8, dtype=torch.bool)
    mask0[:, :60, :80] = True
    mask1[:, :80, :60] = True
    data = {
        "image0": torch.rand(1, 1, 832, 832),
        "image1": torch.rand(1, 1, 832, 832),
        "mask0": mask0,
        "mask1": mask1,
    }
    net(data)
