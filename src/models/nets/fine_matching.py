# TODO:
# - Change output keys
# - remove temperature, use self.scale

from typing import Any, Dict

import torch
import torch.nn as nn
from kornia.geometry import spatial_expectation2d
from kornia.utils import create_meshgrid


class FineMatching(nn.Module):
    def __init__(
        self,
        name: str,
        dim: int,
        window_size: int,
        cls_offset: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = name
        self.window_size = window_size
        self.cls_offset = cls_offset  # NOTE: used by matching_module.py
        self.scale = dim**-0.5

        if name == "classification":
            grid = create_meshgrid(
                window_size, window_size, normalized_coordinates=False
            ).flatten(end_dim=-2)
            cls_biases = grid - window_size / 2 + cls_offset
            self.register_buffer("cls_biases", cls_biases, persistent=False)
        elif name == "regression":
            pass
        else:
            raise ValueError("")

    def compute_cls_biases(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Dict[str, Any]:
        ww = self.window_size**2

        if x0.shape[0] == 0:
            out = {
                "fine_cls_heatmap": x0.new_empty(0, ww, ww),
                "fine_cls_idxes": 3 * (x0.new_empty(0, dtype=torch.long),),
                "fine_cls_biases0": x0.new_empty(0, 2),
                "fine_cls_biases1": x0.new_empty(0, 2),
            }
            return out

        x0, x1 = x0 * self.scale, x1
        similarity = torch.einsum("mlc,msc->mls", x0, x1)
        confidence = similarity.softmax(dim=-2) * similarity.softmax(dim=-1)

        with torch.no_grad():
            m_indices = torch.arange(x0.shape[0], device=x0.device)
            ij_indices = confidence.flatten(start_dim=-2).argmax(dim=-1)
            i_indices, j_indices = ij_indices // ww, ij_indices % ww
            biases0 = self.cls_biases[i_indices]
            biases1 = self.cls_biases[j_indices]

        out = {
            "fine_cls_heatmap": confidence,
            "fine_cls_idxes": (m_indices, i_indices, j_indices),
            "fine_cls_biases0": biases0,
            "fine_cls_biases1": biases1,
        }
        return out

    def compute_reg_biases(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Dict[str, Any]:
        w = self.window_size

        if x0.shape[0] == 0:
            out = {"fine_reg_biases": x0.new_empty(0, 2)}
            return out

        x0 = x0[:, w**2 // 2] * self.scale
        similarity = torch.einsum("mc,mrc->mr", x0, x1)
        confidence = similarity.softmax(dim=-1).unflatten(-1, (w, w))
        biases = spatial_expectation2d(confidence[None])[0]
        out = {"fine_reg_biases": biases}
        return out

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        if self.name == "classification":
            out = self.compute_cls_biases(x0, x1)
        elif self.name == "regression":
            out = self.compute_reg_biases(x0, x1)
        else:
            raise AssertionError("")
        return out
