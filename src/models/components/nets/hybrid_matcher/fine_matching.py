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
        fine_feature0: torch.Tensor,
        fine_feature1: torch.Tensor
    ) -> Dict[str, Any]:
        m, ww, c = fine_feature0.shape
        w = int(ww ** 0.5)
        if m == 0:
            biases = fine_feature0.new_empty((0, 2))
            stddevs = fine_feature0.new_empty((0,))
            fine_matching = {"fine_biases": biases, "fine_stddevs": stddevs}
            return fine_matching

        similarities = torch.einsum(
            "mc,mrc->mr", fine_feature0[:, ww // 2, :], fine_feature1)
        similarities /= c ** 0.5

        heatmap = F.softmax(similarities, dim=1).reshape(-1, w, w)
        biases = K.geometry.spatial_expectation2d(heatmap[None])[0]

        with torch.no_grad():
            coors = K.create_meshgrid(w, w, device=biases.device)
            vars = ((heatmap[..., None] * coors ** 2).sum(dim=(1, 2)) -
                    biases ** 2)
            stddevs = vars.clamp(min=1e-10).sqrt().sum(dim=1)
        fine_matching = {"fine_biases": biases, "fine_stddevs": stddevs}
        return fine_matching
