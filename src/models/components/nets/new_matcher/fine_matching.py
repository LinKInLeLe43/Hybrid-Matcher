from typing import Any, Dict, Optional

import einops
import kornia as K
from kornia import geometry
import torch
from torch import nn
from torch.nn import functional as F


def _crop_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    ww = kernel_size ** 2
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(output, "n (c ww) l -> n l ww c", ww=ww)
    return output


class FineMatching(nn.Module):  # TODO: change name to second stage
    def __init__(
        self,
        type: str,
        cls_depth: Optional[int] = None,
        cls_window_size: Optional[int] = None,
        cls_temperature: float = 1.0,
        reg_window_size: Optional[int] = None,
        reg_with_std: Optional[bool] = None,
        reg_temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.type = type
        self.cls_depth = None
        self.cls_window_size = None
        self.cls_temperature = cls_temperature
        self.reg_window_size = None
        self.reg_with_std = None
        self.reg_temperature = reg_temperature

        if type == "one_stage":
            if reg_window_size is None or reg_with_std is None:
                raise ValueError("")

            self.reg_window_size = reg_window_size
            self.reg_with_std = reg_with_std
        elif type == "two_stage":
            if (cls_depth is None or cls_window_size is None or
                reg_window_size is None or reg_with_std is None):
                raise ValueError("")

            self.cls_depth = cls_depth
            self.cls_window_size = cls_window_size
            self.reg_window_size = reg_window_size
            self.reg_with_std = reg_with_std

            cls_w, e = cls_window_size, self.reg_window_size // 2
            w1 = cls_w + 2 * e
            mask1 = torch.zeros((w1, w1), dtype=torch.bool)
            mask1[e:-e, e:-e] = True
            mask1 = mask1.flatten()
            self.register_buffer("cls_mask1", mask1, persistent=False)

            grid = K.create_meshgrid(cls_w, cls_w, normalized_coordinates=False)
            biases = (grid - cls_w // 2 + 0.5).reshape(-1, 2)
            self.register_buffer("cls_biases", biases, persistent=False)
        else:
            raise ValueError("")

    def _compute_cls_biases(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        device = feature0.device
        m, ww, c = feature0.shape
        if m == 0:
            heatmap = torch.empty((0, ww, ww), device=device)
            _idxes = torch.empty((0,), dtype=torch.long, device=device)
            idxes = 3 * (_idxes,)
            biases0 = torch.empty((0, 2), device=device)
            biases1 = torch.empty((0, 2), device=device)
            result = {"second_stage_cls_heatmap": heatmap,
                      "second_stage_idxes": idxes,
                      "cls_biases0": biases0,
                      "cls_biases1": biases1}
            return result

        similarities = torch.einsum("mlc,msc->mls", feature0, feature1) / c
        similarities /= self.cls_temperature
        heatmap = (F.softmax(similarities, dim=1) *
                   F.softmax(similarities, dim=2))

        with torch.no_grad():
            m_idxes = torch.arange(m, device=device)
            idxes = heatmap.flatten(start_dim=1).argmax(dim=1)
            idxes = m_idxes, idxes // ww, idxes % ww
            biases0 = self.cls_biases.index_select(0, idxes[1])
            biases1 = self.cls_biases.index_select(0, idxes[2])
        result = {"second_stage_cls_heatmap": heatmap,
                  "second_stage_idxes": idxes,
                  "cls_biases0": biases0,
                  "cls_biases1": biases1}
        return result

    def _compute_reg_biases(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        device = feature0.device
        m, c = feature0.shape
        if m == 0:
            biases = torch.empty((0, 2), device=device)
            result = {"reg_biases": biases}

            if self.reg_with_std:
                stds = torch.empty((0,), device=device)
                result["reg_stds"] = stds
            return result

        w = self.reg_window_size
        similarities = torch.einsum("mc,mrc->mr", feature0, feature1) / c ** 0.5
        similarities /= self.reg_temperature
        heatmap = F.softmax(similarities, dim=1).reshape(-1, w, w)
        biases = geometry.spatial_expectation2d(heatmap[None])[0]
        result = {"reg_biases": biases}

        if self.reg_with_std:
            with torch.no_grad():
                grid = K.create_meshgrid(w, w, device=device) ** 2
                vars = (heatmap[..., None] * grid).sum(dim=(1, 2)) - biases ** 2
                stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
                result["reg_stds"] = stds
        return result

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        m, w0w0, c = feature0.shape
        _, w1w1, _ = feature1.shape

        if self.type == "one_stage":
            reg_w = self.reg_window_size
            if w0w0 != w1w1 or w0w0 != reg_w ** 2:
                raise ValueError("")

            result = self._compute_reg_biases(
                feature0[:, w0w0 // 2, :], feature1)

            biases1 = reg_w // 2 * result["reg_biases"].detach()
            result["biases1"] = biases1
        elif self.type == "two_stage":
            cls_w, reg_w = self.cls_window_size, self.reg_window_size
            w1 = cls_w + 2 * (reg_w // 2)
            if w0w0 != cls_w ** 2 or w1w1 != w1 ** 2:
                raise ValueError("")

            cls_c, reg_c = self.cls_depth, c - self.cls_depth
            cls_feature0, reg_feature0 = feature0.split([cls_c, reg_c], dim=2)
            cls_feature1, reg_feature1 = feature1.split([cls_c, reg_c], dim=2)

            result = {}
            result.update(self._compute_cls_biases(
                cls_feature0, cls_feature1[:, self.cls_mask1, :]))

            m_idxes, i_idxes, j_idxes = result["second_stage_idxes"]
            reg_feature0 = reg_feature0[m_idxes, i_idxes]
            reg_feature1 = reg_feature1.transpose(1, 2).unflatten(2, (w1, w1))
            reg_feature1 = _crop_windows(reg_feature1, reg_w, 1, 0)
            reg_feature1 = reg_feature1[m_idxes, j_idxes]
            result.update(self._compute_reg_biases(reg_feature0, reg_feature1))

            biases0 = result.pop("cls_biases0")
            biases1 = (result.pop("cls_biases1") +
                       reg_w // 2 * result["reg_biases"].detach())
            result["biases0"] = biases0
            result["biases1"] = biases1
        else:
            assert False
        return result
