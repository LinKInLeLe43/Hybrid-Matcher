from typing import Any, Dict

import kornia as K
from kornia import geometry
import torch
from torch import nn
from torch.nn import functional as F


class FineMatching(nn.Module):  # TODO: change name to second stage
    def __init__(
        self,
        type: str,
        window_size: int,
        temperature: float = 1.0,
        cls_border_removal: int = 0,
        reg_with_std: bool = False
    ) -> None:
        super().__init__()
        self.type = type
        self.window_size = window_size
        self.temperature = temperature

        w = window_size
        if type == "classification":
            self.cls_border_removal = cls_border_removal

            grid = K.create_meshgrid(w, w, normalized_coordinates=False)
            bias_table = (grid - w // 2 + 0.5).reshape(-1, 2)
            self.register_buffer("cls_bias_table", bias_table, persistent=False)
        elif type == "regression":
            self.reg_with_std = reg_with_std
        else:
            raise ValueError("")

    def _compute_cls_biases(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        m, ww, c = x0.shape
        w, r = self.window_size, self.cls_border_removal

        if m == 0:
            heatmap = x0.new_empty((0, ww, ww))
            idxes = 3 * (x0.new_empty((0,), dtype=torch.long),)
            biases0 = x0.new_empty((0, 2))
            biases1 = x0.new_empty((0, 2))
            result = {"fine_cls_heatmap": heatmap,
                      "fine_cls_idxes": idxes,
                      "fine_cls_biases0": biases0,
                      "fine_cls_biases1": biases1}
            return result

        similarities = torch.einsum("mlc,msc->mls", x0, x1) / c
        similarities /= self.temperature
        heatmap = (F.softmax(similarities, dim=1) *
                   F.softmax(similarities, dim=2))

        with torch.no_grad():
            _heatmap = heatmap
            if r != 0:
                mask = x0.new_zeros((m, w, w, w, w), dtype=torch.bool)
                mask[:, r:-r, r:-r, r:-r, r:-r] = True
                mask = mask.reshape(m, ww, ww)
                _heatmap = heatmap.masked_fill(~mask, float("-inf"))

            m_idxes = torch.arange(m, device=x0.device)
            idxes = _heatmap.flatten(start_dim=1).argmax(dim=1)
            idxes = m_idxes, idxes // ww, idxes % ww
            biases0 = self.cls_bias_table.index_select(0, idxes[1])
            biases1 = self.cls_bias_table.index_select(0, idxes[2])
        result = {"fine_cls_heatmap": heatmap,
                  "fine_cls_idxes": idxes,
                  "fine_cls_biases0": biases0,
                  "fine_cls_biases1": biases1}
        return result

    def _compute_reg_biases(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        m, ww, c = x0.shape
        w = self.window_size

        if m == 0:
            biases = x0.new_empty((0, 2))
            result = {"fine_reg_biases": biases}

            if self.reg_with_std:
                stds = x0.new_empty((0,))
                result["fine_reg_stds"] = stds
            return result

        similarities = torch.einsum("mc,mrc->mr", x0[:, ww // 2], x1) / c ** 0.5
        similarities /= self.temperature
        heatmap = F.softmax(similarities, dim=1).reshape(-1, w, w)
        biases = w // 2 * geometry.spatial_expectation2d(heatmap[None])[0]
        result = {"fine_reg_biases": biases}

        if self.reg_with_std:
            with torch.no_grad():
                grid = K.create_meshgrid(w, w, device=x0.device) ** 2
                vars = (heatmap[..., None] * grid).sum(dim=(1, 2)) - biases ** 2
                stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
            result["fine_reg_stds"] = stds
        return result

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        if x0.shape[1] != self.window_size ** 2:
            raise ValueError("")

        if self.type == "classification":
            result = self._compute_cls_biases(x0, x1)
        elif self.type == "regression":
            result = self._compute_reg_biases(x0, x1)
        else:
            assert False
        return result
