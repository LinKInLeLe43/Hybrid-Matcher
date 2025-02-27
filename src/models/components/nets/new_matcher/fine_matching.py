from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix.dsnt import spatial_expectation2d
from kornia.utils.grid import create_meshgrid

# TODO:
# Change out keys


class FineMatching(nn.Module):
    def __init__(
        self,
        type: str,
        feat_dim: int,
        window_size: int,
        temperature: float = 1.0,
        cls_offset: float = 0.5,
    ) -> None:
        super().__init__()
        self.type = type
        self.window_size = window_size
        self.temperature = temperature
        self.cls_offset = cls_offset
        self.scale = feat_dim**-0.5

        if type == "cls":
            cls_biases = create_meshgrid(
                window_size, window_size, normalized_coordinates=False
            )
            cls_biases = cls_biases - window_size / 2 + cls_offset
            cls_biases = cls_biases.reshape(-1, 2)
            self.register_buffer("cls_biases", cls_biases, persistent=False)
        elif type == "reg":
            pass
        else:
            raise ValueError()

    def _compute_cls_biases(
        self, feat0: torch.Tensor, feat1: torch.Tensor
    ) -> Dict[str, Any]:
        ww = self.window_size**2
        if feat0.shape[0] == 0:
            out = {
                "fine_cls_heatmap": feat0.new_empty(0, ww, ww),
                "fine_cls_idxes": 3 * (feat0.new_empty(0, dtype=torch.long),),
                "fine_cls_biases0": feat0.new_empty(0, 2),
                "fine_cls_biases1": feat1.new_empty(0, 2),
            }
            return out

        feat0, feat1 = feat0 * self.scale, feat1 * self.scale
        similarity = (
            torch.einsum("...lc,...sc->...ls", feat0, feat1) / self.temperature
        )
        heatmap = F.softmax(similarity, dim=-2) * F.softmax(similarity, dim=-1)

        with torch.no_grad():
            m_indices = torch.arange(feat0.shape[0], device=feat0.device)
            ij_indices = heatmap.flatten(start_dim=-2).argmax(dim=-1)
            i_indices, j_indices = ij_indices // ww, ij_indices % ww
            biases0 = self.cls_biases[i_indices]
            biases1 = self.cls_biases[j_indices]

        out = {
            "fine_cls_heatmap": heatmap,
            "fine_cls_idxes": (m_indices, i_indices, j_indices),
            "fine_cls_biases0": biases0,
            "fine_cls_biases1": biases1,
        }
        return out

    def _compute_reg_biases(
        self, feat0: torch.Tensor, feat1: torch.Tensor
    ) -> Dict[str, Any]:
        if feat0.shape[0] == 0:
            out = {"fine_reg_biases": feat0.new_empty(0, 2)}
            return out

        feat0 = feat0[:, self.window_size**2 // 2]
        feat1 = feat1 * self.scale
        similarity = (
            torch.einsum("...c,...rc->...r", feat0, feat1) / self.temperature
        )
        heatmap = F.softmax(similarity, dim=-1)
        heatmap = heatmap.unflatten(-1, (self.window_size, self.window_size))
        biases = spatial_expectation2d(heatmap[None])[0]
        out = {"fine_reg_biases": biases}
        return out

    def forward(
        self, feat0: torch.Tensor, feat1: torch.Tensor
    ) -> Dict[str, Any]:
        if self.type == "cls":
            out = self._compute_cls_biases(feat0, feat1)
        elif self.type == "reg":
            out = self._compute_reg_biases(feat0, feat1)
        else:
            raise AssertionError()
        return out
