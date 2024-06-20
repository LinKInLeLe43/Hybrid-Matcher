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
        depth: int,
        window_size: int,
        temperature: float = 1.0,
        cls_border_removal: int = 0,
        reg_by_exp_with_std: bool = False
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
        elif type == "regression_by_expectation":
            self.reg_by_exp_with_std = reg_by_exp_with_std
        elif type == "regression_by_mlp":
            self.compact = nn.Sequential(
                nn.Conv2d(2 * depth, depth, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(True),
                nn.Conv2d(
                    depth, depth // 2, 3, stride=2, padding=1, bias=False))
            self.mlp = nn.Sequential(
                nn.Linear(
                    round(window_size / 4 + 0.5) ** 2 * (depth // 2),
                    depth // 2, bias=False),
                nn.LeakyReLU(),
                nn.Linear(depth // 2, depth // 4, bias=False),
                nn.LeakyReLU(),
                nn.Linear(depth // 4, 2, bias=False),
                nn.Tanh())
        else:
            raise ValueError("")

    def _compute_cls_biases(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        w, r = self.window_size, self.cls_border_removal
        if len(x0.shape) == 4:
            x0 = x0.flatten(start_dim=2).transpose(1, 2)
            x1 = x1.flatten(start_dim=2).transpose(1, 2)

        m, ww, c = x0.shape
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

    def _compute_reg_biases_by_expectation(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        w = self.window_size
        if len(x0.shape) == 4:
            x0 = x0.flatten(start_dim=2).transpose(1, 2)
            x1 = x1.flatten(start_dim=2).transpose(1, 2)

        m, ww, c = x0.shape
        if m == 0:
            biases = x0.new_empty((0, 2))
            result = {"fine_reg_biases": biases}

            if self.reg_by_exp_with_std:
                stds = x0.new_empty((0,))
                result["fine_reg_stds"] = stds
            return result

        central_x0 = x0[:, ww // 2]
        similarities = torch.einsum("mc,mrc->mr", central_x0, x1) / c ** 0.5
        similarities /= self.temperature
        heatmap = F.softmax(similarities, dim=1)
        heatmap = heatmap.reshape(m, w, w)
        biases = geometry.spatial_expectation2d(heatmap[None])[0]
        result = {"fine_reg_biases": biases}

        if self.reg_by_exp_with_std:
            with torch.no_grad():
                grid = K.create_meshgrid(w, w, device=x0.device) ** 2
                vars = (heatmap[..., None] * grid).sum(dim=(1, 2)) - biases ** 2
                stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
            result["fine_reg_stds"] = stds
        return result

    def _compute_reg_biases_by_mlp(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Dict[str, Any]:
        w = self.window_size
        if len(x0.shape) == 3:
            x0 = x0.transpose(1, 2).unflatten(2, (w, w))
            x1 = x1.transpose(1, 2).unflatten(2, (w, w))

        m, c, w, w = x0.shape
        if m == 0:
            biases = x0.new_empty((0, 2))
            result = {"fine_reg_biases": biases}
            return result

        x = torch.cat([x0, x1], dim=1)
        x = self.compact(x)
        x = x.reshape(m, -1)
        biases = self.mlp(x)
        result = {"fine_reg_biases": biases}
        return result

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Dict[str, Any]:
        if self.type == "classification":
            result = self._compute_cls_biases(x0, x1)
        elif self.type == "regression_by_expectation":
            result = self._compute_reg_biases_by_expectation(x0, x1)
        elif self.type == "regression_by_mlp":
            result = self._compute_reg_biases_by_mlp(x0, x1)
        else:
            assert False
        return result
