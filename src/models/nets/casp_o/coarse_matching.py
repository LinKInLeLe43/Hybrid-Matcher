from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from kornia.utils.grid import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .utils import (
    gather_attended,
    remove_border_both_sides,
    remove_border_one_side,
)


class CoarseMatching(Module):
    def __init__(
        self,
        decoder: Module,
        threshold: float = 0.2,
        border_removal: int = 2,
        train_percent: float = 0.2,
        train_min_num_gt: int = 200,
    ) -> None:
        super().__init__()
        self.stride = decoder.stride
        self.decoder = decoder
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
        self.register_buffer("delta_indices", grid, persistent=False)

    def map_indices(self, x: Tensor, size: Sequence[int], fw: int) -> Tensor:
        row = (x[..., None] // fw) * self.stride + self.delta_indices[:, 1]
        col = (x[..., None] % fw) * self.stride + self.delta_indices[:, 0]
        out = row * fw * self.stride + col
        out = (
            out.unflatten(1, size)
            .repeat_interleave(self.stride, dim=1)
            .repeat_interleave(self.stride, dim=2)
            .flatten(start_dim=1, end_dim=2)
            .flatten(start_dim=-2)
        )
        return out

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
    def create_coarse_matching_for_train(
        self,
        heatmap: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
        gt_indices: Optional[Tensor],
    ) -> Dict[str, Any]:
        mask0_to_1 = heatmap == heatmap.amax(dim=-1, keepdim=True)
        mask1_to_0 = heatmap == heatmap.amax(dim=-2, keepdim=True)
        mask = mask0_to_1 & mask1_to_0 & (heatmap > self.threshold)
        mask = remove_border_both_sides(
            mask, self.border_removal, size0, size1, mask0, mask1
        )
        indices = self.sample_for_train(mask, gt_indices)
        scores = heatmap[indices.unbind()]
        results = {"coarse_cls_indices": indices, "scores": scores}
        return results

    @torch.no_grad()
    def create_coarse_matching_for_eval(
        self,
        heatmap0_to_1: Tensor,
        heatmap1_to_0: Tensor,
        indices0_to_1: Tensor,
        indices1_to_0: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
        gt_indices: Optional[Tensor],
    ) -> Dict[str, Any]:
        score0_to_1, sub_indices0_to_1 = heatmap0_to_1.max(dim=-1, keepdim=True)
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
        results = {"coarse_cls_indices": indices, "scores": scores}
        return results

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Tensor,
        y1: Tensor,
        encoding: Tensor,
        x0_mask: Optional[Tensor] = None,
        x1_mask: Optional[Tensor] = None,
        y0_mask: Optional[Tensor] = None,
        y1_mask: Optional[Tensor] = None,
        y_gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        only_decode: bool = False,
    ) -> Dict[str, Any]:
        inf = 1e9 if self.training else float("inf")
        n, c, h0, w0 = y0.shape
        _, _, h1, w1 = y1.shape
        sh = sw = self.stride
        fh0, fw0, fh1, fw1 = [t // self.stride for t in [h0, w0, h1, w1]]
        scale = c**-0.5

        out0_list, out1_list, indices0_to_1, indices1_to_0, similarity = (
            self.decoder(x0, x1, y0, y1, encoding, mask0=x0_mask, mask1=x1_mask)
        )
        _y0, _y1 = out0_list[0], out1_list[0]
        out0_list[0] = y0 = (
            _y0.reshape(n, fh0, fw0, sh, sw, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(n, c, h0, w0)
        )
        out1_list[0] = y1 = (
            _y1.reshape(n, fh1, fw1, sh, sw, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(n, c, h1, w1)
        )

        results = {}

        results["x"] = (out0_list, out1_list)
        if only_decode:
            return results

        if self.training:
            heatmap0_to_1 = similarity.softmax(dim=-1)
            heatmap1_to_0 = similarity.softmax(dim=-2)
            heatmap = heatmap0_to_1 * heatmap1_to_0
            extra_coarse_cls_heatmap = (
                heatmap.reshape(n, fh0, fw0, fh1, fw1)
                .repeat_interleave(self.stride, dim=1)
                .repeat_interleave(self.stride, dim=2)
                .repeat_interleave(self.stride, dim=3)
                .repeat_interleave(self.stride, dim=4)
                .reshape(n, h0 * w0, h1 * w1)
                .clamp(min=1e-6, max=1 - 1e-6)
                .log()
            )

            y0 = y0.flatten(start_dim=2).transpose(1, 2) * self.scale
            y1 = y1.flatten(start_dim=2).transpose(1, 2)
            similarity = y0 @ y1.transpose(-1, -2)
            if y0_mask is not None and y1_mask is not None:
                mask = y0_mask.view(n, -1, 1) & y1_mask.view(n, 1, -1)
                similarity.masked_fill_(~mask, -inf)

            heatmap0_to_1 = similarity.softmax(dim=-1)
            heatmap1_to_0 = similarity.softmax(dim=-2)
            heatmap = heatmap0_to_1 * heatmap1_to_0
            coarse_cls_heatmap = heatmap.clamp(min=1e-6, max=1 - 1e-6).log()
            results["coarse_cls_heatmap"] = (
                coarse_cls_heatmap + extra_coarse_cls_heatmap * 0.5
            )
            results.update(
                self.create_coarse_matching_for_train(
                    heatmap,
                    (h0, w0),
                    (h1, w1),
                    mask0=y0_mask,
                    mask1=y1_mask,
                    gt_indices=y_gt_idxes,
                )
            )
        else:
            attended0_to_1 = gather_attended(_y1, indices0_to_1)
            attended1_to_0 = gather_attended(_y0, indices1_to_0)
            indices0_to_1 = self.map_indices(indices0_to_1, (fh0, fw0), fw1)
            indices1_to_0 = self.map_indices(
                indices1_to_0, (fh1, fw1), fw0
            ).transpose(-2, -1)
            similarity0_to_1 = _y0 @ attended0_to_1.transpose(-2, -1) * scale
            similarity1_to_0 = _y1 @ attended1_to_0.transpose(-2, -1) * scale
            heatmap0_to_1 = (
                similarity0_to_1.softmax(dim=-1)
                .reshape(n, fh0, fw0, self.stride, self.stride, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(start_dim=1, end_dim=-2)
            )
            heatmap1_to_0 = (
                similarity1_to_0.softmax(dim=-1)
                .reshape(n, fh1, fw1, self.stride, self.stride, -1)
                .permute(0, 5, 1, 3, 2, 4)
                .flatten(start_dim=2, end_dim=-1)
            )
            heatmap0_to_1, heatmap1_to_0 = (
                heatmap0_to_1
                * (
                    y1.new_zeros(n, h0 * w0, h1 * w1)
                    .scatter_(-2, indices1_to_0, heatmap1_to_0)
                    .gather(-1, indices0_to_1)
                ),
                heatmap1_to_0
                * (
                    y0.new_zeros(n, h0 * w0, h1 * w1)
                    .scatter_(-1, indices0_to_1, heatmap0_to_1)
                    .gather(-2, indices1_to_0)
                ),
            )
            results.update(
                self.create_coarse_matching_for_eval(
                    heatmap0_to_1,
                    heatmap1_to_0,
                    indices0_to_1,
                    indices1_to_0,
                    (h0, w0),
                    (h1, w1),
                    mask0=y0_mask,
                    mask1=y1_mask,
                    gt_indices=y_gt_idxes,
                )
            )
        return results
