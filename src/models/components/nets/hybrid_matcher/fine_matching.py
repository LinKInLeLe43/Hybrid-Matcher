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
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(
        output, "n (c ww) l -> n l ww c", ww=kernel_size ** 2)
    return output


class FineMatching(nn.Module):
    def __init__(
        self,
        type: str,
        cls_window_size: Optional[int] = None,
        cls_depth: Optional[int] = None,
        cls_top_k: Optional[int] = None,
        cls_temperature: int = 1.0,
        reg_window_size: Optional[int] = None,
        reg_temperature: int = 1.0
    ) -> None:
        super().__init__()
        self.type = type
        self.cls_window_size = None
        self.cls_depth = None
        self.cls_top_k = None
        self.cls_temperature = cls_temperature
        self.reg_window_size = None
        self.reg_with_std = None
        self.reg_temperature = reg_temperature

        if type == "reg_only":
            if reg_window_size is None or reg_window_size % 2 == 0:
                raise ValueError("")
            self.reg_window_size = reg_window_size
            self.reg_with_std = False
        elif type == "reg_only_with_std":
            if reg_window_size is None or reg_window_size % 2 == 0:
                raise ValueError("")
            self.reg_window_size = reg_window_size
            self.reg_with_std = True
        elif type == "cls_and_reg":
            if (cls_window_size is None or cls_depth is None or
                cls_top_k is None or reg_window_size is None):
                raise ValueError("")
            if cls_window_size % 2 == 1 or reg_window_size % 2 == 0:
                raise ValueError("")
            self.cls_window_size = cls_window_size
            self.cls_depth = cls_depth
            self.cls_top_k = cls_top_k
            self.reg_window_size = reg_window_size
            self.reg_with_std = False

            cls_w, p = cls_window_size, self.reg_window_size // 2
            self.cls_mask = torch.zeros(
                (cls_w + 2 * p, cls_w + 2 * p), dtype=torch.bool)
            self.cls_mask[p:-p, p:-p] = True
            self.cls_mask = self.cls_mask.flatten()
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
            heatmap = feature0.new_empty((0, ww, ww))
            idxes0 = feature0.new_empty((0, self.cls_top_k), dtype=torch.long)
            idxes1 = feature0.new_empty((0, self.cls_top_k), dtype=torch.long)
            biases0 = feature0.new_empty((0, 2))
            biases1 = feature0.new_empty((0, 2))
            result = {"fine_cls_heatmap": heatmap,
                      "fine_cls_idxes0": idxes0,
                      "fine_cls_idxes1": idxes1,
                      "fine_cls_biases0": biases0,
                      "fine_cls_biases1": biases1}
            return result
        w = self.cls_window_size

        similarities = torch.einsum("mlc,msc->mls", feature0, feature1) / c
        similarities /= self.cls_temperature
        heatmap = (F.softmax(similarities, dim=1) *
                   F.softmax(similarities, dim=2))
        with torch.no_grad():
            if self.cls_top_k == 1:
                idxes = heatmap.flatten(start_dim=1).argmax(dim=1, keepdim=True)
                idxes0, idxes1 = idxes // ww, idxes % ww
            else:
                raise NotImplementedError("")
            grid = K.create_meshgrid(
                w, w, normalized_coordinates=False, device=device)
            biases = (grid - w // 2 + 0.5).reshape(-1, 2)
            biases0 = biases.index_select(0, idxes0.flatten()).reshape(m, -1)
            biases1 = biases.index_select(0, idxes1.flatten()).reshape(m, -1)
        result = {"fine_cls_heatmap": heatmap,
                  "fine_cls_idxes0": idxes0,
                  "fine_cls_idxes1": idxes1,
                  "fine_cls_biases0": biases0,
                  "fine_cls_biases1": biases1}
        return result

    def _compute_reg_biases(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        device = feature0.device
        m, c = feature0.shape
        if m == 0:
            biases, stds = feature0.new_empty((0, 2)), feature0.new_empty((0,))
            result = {"fine_reg_biases": biases, "fine_reg_stds": stds}
            return result
        w = self.reg_window_size

        similarities = torch.einsum("mc,mrc->mr", feature0, feature1) / c ** 0.5
        similarities /= self.reg_temperature
        heatmap = F.softmax(similarities, dim=1).reshape(-1, w, w)
        biases = geometry.spatial_expectation2d(heatmap[None])[0]
        result = {"fine_reg_biases": biases}

        if self.reg_with_std:
            with torch.no_grad():
                grid = K.create_meshgrid(w, w, device=device) ** 2
                vars = (heatmap[..., None] * grid).sum(dim=(1, 2)) - biases ** 2
                stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
                result["fine_reg_stds"] = stds
        return result

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        m, w0w0, c = feature0.shape
        _, w1w1, _ = feature1.shape

        if self.type == "reg_only_with_std":
            reg_w = self.reg_window_size
            if w0w0 != w1w1 or w0w0 != reg_w ** 2:
                raise ValueError("")
            result = self._compute_reg_biases(
                feature0[:, w0w0 // 2, :], feature1)
        elif self.type == "reg_only":
            reg_w = self.reg_window_size
            if w0w0 != w1w1 or w0w0 != reg_w ** 2:
                raise ValueError("")
            result = self._compute_reg_biases(
                feature0[:, w0w0 // 2, :], feature1)
        elif self.type == "cls_and_reg":
            device = feature0.device
            cls_w, reg_w = self.cls_window_size, self.reg_window_size
            w1 = cls_w + 2 * (reg_w // 2)
            if w0w0 != cls_w ** 2 or w1w1 != w1 ** 2:
                raise ValueError("")
            cls_c, reg_c = self.cls_depth, c - self.cls_depth

            result = {}
            cls_feature0, reg_feature0 = feature0.split([cls_c, reg_c], dim=2)
            cls_feature1, reg_feature1 = feature1.split([cls_c, reg_c], dim=2)
            result.update(self._compute_cls_biases(
                cls_feature0, cls_feature1[:, self.cls_mask, :]))

            range = torch.arange(m, device=device)
            reg_feature0 = reg_feature0[range, result["fine_cls_idxes0"][:, 0]]
            reg_feature1 = reg_feature1.transpose(1, 2).unflatten(2, (w1, w1))
            reg_feature1 = _crop_windows(reg_feature1, reg_w, 1, 0)
            reg_feature1 = reg_feature1[range, result["fine_cls_idxes1"][:, 0]]
            result.update(self._compute_reg_biases(reg_feature0, reg_feature1))
        else:
            assert False
        return result
