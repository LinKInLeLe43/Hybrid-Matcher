from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module

from .coarse_matching import CoarseMatching
from .decoders import Decoder
from .encoders import Encoder
from .fine_matching import FineMatching


class CasP_O(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.scales = config["scales"]
        self.data_mode = config["data_mode"]
        self.data_factor = config["data_factor"]

        self.encoder = Encoder(**config["encoder"])
        self.decoder = Decoder(**config["decoder"])
        self.coarse_matching = CoarseMatching(**config["coarse_matching"])
        self.fine_matching = FineMatching()

    @torch.no_grad()
    def update_points(
        self, data: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        scale_coarse = self.scales[0]
        # scale_fine = self.scales[1] * (self.fine_reg_matching.window_size // 2)
        w0 = data["image0"].shape[-1] // scale_coarse
        w1 = data["image1"].shape[-1] // scale_coarse
        b_indices, i_indices, j_indices = results["coarse_cls_indices"]

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
        enable_crop: bool = True,
        gt_indices: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        image0, image1 = data["image0"], data["image1"]
        mask0, mask1 = data.get("mask0"), data.get("mask1")

        x0_list, x1_list = self.encoder(image0, image1)
        results = self.decoder(
            x0_list[:-3:-1],
            x1_list[:-3:-1],
            enable_crop=enable_crop,
            mask0=mask0,
            mask1=mask1,
        )

        results = self.coarse_matching(
            **results, mask0=mask0, mask1=mask1, gt_indices=gt_indices
        )
        x0_8x, x1_8x = results.pop("x_8x")

        results.update(
            self.fine_matching(
                x0_list[-2],
                x1_list[-2],
                x0_8x,
                x1_8x,
                results["coarse_cls_indices"],
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
