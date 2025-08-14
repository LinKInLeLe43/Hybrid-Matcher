from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .utils import (
    gather_attended,
    remove_border_both_sides,
    remove_border_one_side,
    unpatchify,
)


class CoarseMatching(Module):
    def __init__(
        self,
        stride_list: List[int],
        threshold: float = 0.2,
        border_removal: int = 2,
        train_percent: float = 0.2,
        train_min_num_gt: int = 200,
    ) -> None:
        super().__init__()
        self.stride_list = stride_list
        self.stride = stride_list[-1]
        self.threshold = threshold
        self.border_removal = border_removal
        self.train_percent = train_percent
        self.train_min_num_gt = train_min_num_gt
        grid = create_meshgrid(
            self.stride,
            self.stride,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("grid", grid, persistent=False)

    def map_indices(
        self, x0: Tensor, size0: Tuple[int, int], size1: Tuple[int, int]
    ) -> Tensor:
        row = (x0[..., None] // size1[1]) * self.stride + self.grid[:, 1]
        col = (x0[..., None] % size1[1]) * self.stride + self.grid[:, 0]
        x0 = (
            (row * size1[1] * self.stride + col)
            .reshape(len(x0), size0[0], 1, size0[1], 1, -1)
            .expand(-1, -1, 2, -1, 2, -1)
            .flatten(start_dim=1, end_dim=4)
        )
        return x0

    def sample_for_train(self, mask: Tensor, gt_indices: Tensor) -> Tensor:
        matching_indices = mask.nonzero().transpose(-2, -1)
        num_matching, num_gt = matching_indices.shape[-1], gt_indices.shape[-1]
        num_total = int(min(mask.shape[-2:]) * self.train_percent)
        if num_matching <= num_total - self.train_min_num_gt:
            matching_subindices = torch.arange(num_matching, device=mask.device)
            gt_subindices = torch.randint(
                num_gt, (num_total - num_matching,), device=mask.device
            )
        else:
            matching_subindices = torch.randint(
                num_matching,
                (num_total - self.train_min_num_gt,),
                device=mask.device,
            )
            gt_subindices = torch.randint(
                num_gt, (self.train_min_num_gt,), device=mask.device
            )
        train_indices = torch.cat(
            [
                matching_indices[:, matching_subindices],
                gt_indices[:, gt_subindices],
            ],
            dim=-1,
        )
        return train_indices

    @torch.no_grad()
    def create_coarse_matching(
        self,
        heatmap: Union[Tensor, Tuple[Tensor, Tensor]],
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
        gt_indices: Optional[Tensor],
    ) -> Dict[str, Any]:
        if self.training and gt_indices is not None:
            mask0_to_1 = heatmap == heatmap.amax(dim=-1, keepdim=True)
            mask1_to_0 = heatmap == heatmap.amax(dim=-2, keepdim=True)
            mask = mask0_to_1 & mask1_to_0 & (heatmap > self.threshold)
            mask = remove_border_both_sides(
                mask, self.border_removal, size0, size1, mask0, mask1
            )
            indices = self.sample_for_train(mask, gt_indices)
            scores = heatmap[indices.unbind()]
        else:
            heatmap0_to_1, heatmap1_to_0 = heatmap
            score0_to_1, sub_indices0_to_1 = heatmap0_to_1.max(
                dim=-1, keepdim=True
            )
            score0_to_1 = score0_to_1[:, :, 0]
            sub_indices1_to_0 = heatmap1_to_0.argmax(dim=-2, keepdim=True)
            indices0_to_1 = indices0_to_1.gather(-1, sub_indices0_to_1)[:, :, 0]
            indices1_to_0 = indices1_to_0.gather(-2, sub_indices1_to_0)[:, 0, :]
            indices0_to_1 = remove_border_one_side(
                indices0_to_1, self.border_removal, size0, mask=mask0
            )
            indices1_to_0 = remove_border_one_side(
                indices1_to_0, self.border_removal, size1, mask=mask1
            )
            biprojection = indices1_to_0.gather(-1, indices0_to_1)
            mask0_to_1 = biprojection == torch.arange(
                heatmap0_to_1.shape[1], device=heatmap0_to_1.device
            )
            if self.border_removal > 0:
                mask0_to_1[:, 0] = False
            mask0_to_1 = mask0_to_1 & (score0_to_1 > self.threshold)
            b_indices, i_indices = mask0_to_1.nonzero(as_tuple=True)
            j_indices = indices0_to_1[b_indices, i_indices]
            indices = torch.stack([b_indices, i_indices, j_indices])
            scores = score0_to_1[mask0_to_1]
        result = {"coarse_cls_indices": indices, "scores": scores}
        return result

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        grid_size0: Tuple[int, int],
        grid_size1: Tuple[int, int],
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        similarity_list: List[Optional[Tensor]],
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
        gt_indices: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        inf = 1e9 if self.training else float("inf")
        n, _, _, c = x0.shape
        h0, w0 = grid_size0[0] * self.stride, grid_size0[1] * self.stride
        h1, w1 = grid_size1[0] * self.stride, grid_size1[1] * self.stride
        scale = c**-0.5
        results = {}

        if self.training:
            for i in reversed(range(len(similarity_list))):
                similarity = torch.stack(similarity_list[i])
                if mask0 is not None and mask1 is not None:
                    if i != len(similarity_list) - 1:
                        mask0, mask1 = mask0.float(), mask1.float()
                        mask0 = F.max_pool2d(mask0, self.stride_list[i]).bool()
                        mask1 = F.max_pool2d(mask1, self.stride_list[i]).bool()
                    mask = mask0.reshape(n, -1, 1) & mask1.reshape(n, 1, -1)
                    similarity.masked_fill_(~mask, -inf)
                heatmap0_to_1 = similarity.softmax(dim=-1)
                heatmap1_to_0 = similarity.softmax(dim=-2)
                similarity_list[i] = heatmap0_to_1 * heatmap1_to_0
            heatmap = similarity_list[-1][-1]
            results["coarse_cls_heatmap"] = similarity_list[-1]
            results["extra_coarse_cls_heatmap"] = similarity_list[-2]
        else:
            attended0_to_1 = gather_attended(x1, indices0_to_1)
            attended1_to_0 = gather_attended(x0, indices1_to_0)
            indices0_to_1 = self.map_indices(
                indices0_to_1, grid_size0, grid_size1
            )
            indices1_to_0 = self.map_indices(
                indices1_to_0, grid_size1, grid_size0
            ).transpose(-2, -1)
            similarity0_to_1 = x0 @ attended0_to_1.transpose(-2, -1) * scale
            similarity1_to_0 = x1 @ attended1_to_0.transpose(-2, -1) * scale
            heatmap0_to_1 = (
                similarity0_to_1.softmax(dim=-1)
                .reshape(n, *grid_size0, self.stride, self.stride, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(start_dim=1, end_dim=-2)
            )
            heatmap1_to_0 = (
                similarity1_to_0.softmax(dim=-1)
                .reshape(n, *grid_size1, self.stride, self.stride, -1)
                .permute(0, 5, 1, 3, 2, 4)
                .flatten(start_dim=2, end_dim=-1)
            )
            heatmap0_to_1, heatmap1_to_0 = (
                heatmap0_to_1
                * (
                    x1.new_zeros(n, h0 * w0, h1 * w1)
                    .scatter_(-2, indices1_to_0, heatmap1_to_0)
                    .gather(-1, indices0_to_1)
                ),
                heatmap1_to_0
                * (
                    x0.new_zeros(n, h0 * w0, h1 * w1)
                    .scatter_(-1, indices0_to_1, heatmap0_to_1)
                    .gather(-2, indices1_to_0)
                ),
            )
            heatmap = heatmap0_to_1, heatmap1_to_0
            x0 = unpatchify(x0, grid_size0, self.stride)
            x1 = unpatchify(x1, grid_size1, self.stride)

        results.update(
            self.create_coarse_matching(
                heatmap,
                indices0_to_1,
                indices1_to_0,
                (h0, w0),
                (h1, w1),
                mask0=mask0,
                mask1=mask1,
                gt_indices=gt_indices,
            )
        )
        results["x_8x"] = x0, x1
        return results
