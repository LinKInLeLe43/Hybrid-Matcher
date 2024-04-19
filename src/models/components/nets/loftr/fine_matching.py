from typing import Any, Dict

import kornia as K
import torch
from torch import nn
from torch.nn import functional as F


class FineMatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor
    ) -> Dict[str, Any]:
        m, ww, c = feature0.shape
        w = int(ww ** 0.5)
        if m == 0:
            biases, stds = feature0.new_empty((0, 2)), feature0.new_empty((0,))
            result = {"fine_reg_biases": biases, "fine_reg_stds": stds}
            return result

        similarities = torch.einsum(
            "mc,mrc->mr", feature0[:, ww // 2, :], feature1) / c ** 0.5

        heatmap = F.softmax(similarities, dim=1).reshape(-1, w, w)
        biases = K.geometry.spatial_expectation2d(heatmap[None])[0]

        with torch.no_grad():
            grid = K.create_meshgrid(w, w, device=biases.device) ** 2
            vars = (heatmap[..., None] * grid).sum(dim=(1, 2)) - biases ** 2
            stds = vars.clamp(min=1e-10).sqrt().sum(dim=1)
        result = {"fine_reg_biases": biases, "fine_reg_stds": stds}
        return result
