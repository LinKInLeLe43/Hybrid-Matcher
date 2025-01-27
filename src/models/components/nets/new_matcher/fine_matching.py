from typing import Any, Dict

import kornia as K
from kornia import geometry
import torch
from torch import nn
from torch.nn import functional as F


class FineMatching(nn.Module):
    def __init__(
        self,
        type: str,
        depth: int,
        window_size: int,
        temperature: float = 1.0,
        cls_offset: float = 0.5,
        reg_by_exp_with_std: bool = False
    ) -> None:
        super().__init__()
        self.type = type
        self.depth = depth
        self.window_size = window_size
        self.temperature = temperature
        self.cls_offset = cls_offset

        w = window_size
        if type == "classification":
            grid = K.create_meshgrid(w, w, normalized_coordinates=False)
            delta = (grid - w / 2 + cls_offset).reshape(-1, 2)
            self.register_buffer("cls_delta", delta, persistent=False)
        elif type == "regression_by_expectation":
            self.reg_by_exp_with_std = reg_by_exp_with_std
        else:
            raise ValueError("")

    def _compute_cls_biases(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        ww, c = self.window_size ** 2, self.depth
        m = x0.shape[0]

        x0, x1 = x0 / c ** 0.5, x1 / c ** 0.5
        similarity = torch.einsum("mlc,msc->mls", x0, x1)
        similarity /= self.temperature
        heatmap = F.softmax(similarity, dim=1) * F.softmax(similarity, dim=2)

        with torch.no_grad():
            m_idxes = torch.arange(m, device=x0.device)
            idxes = heatmap.flatten(start_dim=1).argmax(dim=1)
            idxes = m_idxes, idxes // ww, idxes % ww
            biases0 = self.cls_delta.index_select(0, idxes[1])
            biases1 = self.cls_delta.index_select(0, idxes[2])

        result = {"fine_cls_heatmap": heatmap,
                  "fine_cls_idxes": idxes,
                  "fine_cls_biases0": biases0,
                  "fine_cls_biases1": biases1}
        return result

    def _compute_reg_biases_by_expectation(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        w, c = self.window_size, self.depth
        m, ww = len(x0), w ** 2

        if m == 0:
            result = {"fine_reg_biases": x0.new_empty((0, 2))}

            if self.reg_by_exp_with_std:
                result["fine_reg_stds"] = x0.new_empty((0,))
            return result

        center0 = x0[:, ww // 2] if len(x0.shape) == 3 else x0
        center0, x1 = center0 / c ** 0.25, x1 / c ** 0.25
        similarity = torch.einsum("mc,mrc->mr", center0, x1)
        similarity /= self.temperature
        heatmap = F.softmax(similarity, dim=1).reshape(-1, w, w)
        biases = geometry.spatial_expectation2d(heatmap[None])[0]
        result = {"fine_reg_biases": biases}

        if self.reg_by_exp_with_std:
            with torch.no_grad():
                grid = K.create_meshgrid(w, w, device=x0.device)
                vars = ((heatmap[..., None] * grid ** 2).sum(dim=(1, 2)) -
                        biases ** 2)
                stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
            result["fine_reg_stds"] = stds
        return result

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        if self.type == "classification":
            result = self._compute_cls_biases(x0, x1)
        elif self.type == "regression_by_expectation":
            result = self._compute_reg_biases_by_expectation(x0, x1)
        else:
            assert False
        return result
