from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import window_unpartition


class CoarseMatching(Module):
    def __init__(
        self,
        fused_selective_module: Module,
        topk: int = 8,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200,
    ) -> None:
        super().__init__()
        self.fused_selective_module = fused_selective_module
        self.topk = topk
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        self.train_percent = train_percent
        self.train_min_gt_count = train_min_gt_count

        self.stride = fused_selective_module.stride

    def _remove_border_for_train(
        self,
        x: Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
    ) -> Tuple[Tensor, int]:
        r = self.border_removal
        (h0, w0), (h1, w1) = size0, size1

        out = x.reshape(-1, h0, w0, h1, w1)
        out[:, :r, :, :, :] = False
        out[:, :, :r, :, :] = False
        out[:, :, :, :r, :] = False
        out[:, :, :, :, :r] = False

        if mask0 is not None:
            h0s = mask0.sum(dim=1).amax(dim=1).int()
            w0s = mask0.sum(dim=2).amax(dim=1).int()
            h1s = mask1.sum(dim=1).amax(dim=1).int()
            w1s = mask1.sum(dim=2).amax(dim=1).int()
            max_count = torch.minimum(h0s * w0s, h1s * w1s).sum().item()
            for b, (_h0, _w0, _h1, _w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
                out[b, _h0 - r :, :, :, :] = False
                out[b, :, _w0 - r :, :, :] = False
                out[b, :, :, _h1 - r :, :] = False
                out[b, :, :, :, _w1 - r :] = False
        else:
            max_count = len(x) * min(h0 * w0, h1 * w1)
            if r > 0:
                out[:, -r:, :, :, :] = False
                out[:, :, -r:, :, :] = False
                out[:, :, :, -r:, :] = False
                out[:, :, :, :, -r:] = False

        out = out.reshape(-1, h0 * w0, h1 * w1)
        return out, max_count

    def _remove_border_for_eval(
        self, x: Tensor, size: Tuple[int, int], mask: Optional[Tensor]
    ) -> Tensor:
        r = self.border_removal

        if r == 0:
            return x

        out = x.unflatten(1, size)
        out[:, :r, :] = 0
        out[:, :, :r] = 0

        if mask is not None:
            hs = mask.sum(dim=1).amax(dim=1).int()
            ws = mask.sum(dim=2).amax(dim=1).int()
            for b, (h, w) in enumerate(zip(hs, ws)):
                out[b, h - r :, :] = 0
                out[b, :, w - r :] = 0
        else:
            out[:, -r:, :] = 0
            out[:, :, -r:] = 0
        out = out.flatten(start_dim=1)
        return out

    def _sample_for_train(
        self,
        max_count: int,
        matching_idxes: Tuple[Tensor, Tensor, Tensor],
        gt_idxes: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        device = matching_idxes[0].device

        train_count = int(self.train_percent * max_count)
        rest_count = train_count - self.train_min_gt_count
        matching_count, gt_count = len(matching_idxes[0]), len(gt_idxes[0])
        if matching_count <= rest_count:
            matching_subidxes = torch.arange(matching_count, device=device)
        else:
            matching_subidxes = torch.randint(
                matching_count, (rest_count,), device=device
            )
            matching_count = rest_count
        gt_subidxes = torch.randint(
            gt_count, (train_count - matching_count,), device=device
        )

        matching_idxes = tuple(
            map(lambda x: x[matching_subidxes], matching_idxes)
        )
        train_idxes = tuple(
            map(
                lambda x, y: torch.cat([x, y[gt_subidxes]]),
                matching_idxes,
                gt_idxes,
            )
        )
        return train_idxes, matching_idxes

    @torch.no_grad()
    def _create_coarse_matching(
        self,
        score: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[Tensor],
        mask1: Optional[Tensor],
        gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]],
    ) -> Dict[str, Any]:
        if self.training and gt_idxes is not None:
            score, idxes0_to_1, idxes1_to_0 = score
            mask, max_count = self._remove_border_for_train(
                score > self.threshold, size0, size1, mask0, mask1
            )
            mask &= (score == score.amax(dim=2, keepdim=True)) & (
                score == score.amax(dim=1, keepdim=True)
            )
            train_idxes, matching_idxes = self._sample_for_train(
                max_count, mask.nonzero(as_tuple=True), gt_idxes
            )
            b_idxes, i_idxes, j_idxes = train_idxes
            scores = score[train_idxes]
        else:
            score0_to_1, score1_to_0, idxes0_to_1, idxes1_to_0 = score
            values0_to_1, sub_idxes0_to_1 = score0_to_1.max(dim=2)
            sub_idxes1_to_0 = score1_to_0.argmax(dim=1)
            idxes0_to_1 = idxes0_to_1.gather(2, sub_idxes0_to_1[:, :, None])[
                :, :, 0
            ]
            idxes1_to_0 = idxes1_to_0.gather(1, sub_idxes1_to_0[:, None, :])[
                :, 0, :
            ]
            idxes0_to_1 = self._remove_border_for_eval(
                idxes0_to_1, size0, mask0
            )
            idxes1_to_0 = self._remove_border_for_eval(
                idxes1_to_0, size1, mask1
            )
            biprojection = idxes1_to_0.gather(1, idxes0_to_1)
            mask = biprojection == torch.arange(
                score0_to_1.shape[1], device=score0_to_1.device
            )
            if self.border_removal > 0:
                mask[:, 0] = False
            mask &= values0_to_1 > self.threshold
            b_idxes, i_idxes = mask.nonzero(as_tuple=True)
            j_idxes = idxes0_to_1[b_idxes, i_idxes]
            train_idxes = matching_idxes = b_idxes, i_idxes, j_idxes
            scores = values0_to_1[b_idxes, i_idxes]

        result = {
            "scores": scores,
            "coarse_cls_indices": train_idxes,
        }
        return result

    def forward(
        self,
        x0_list: Sequence[Tensor],
        x1_list: Sequence[Tensor],
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
        gt_indices_list: Optional[
            Sequence[Tuple[Tensor, Tensor, Tensor]]
        ] = None,
    ) -> Dict[str, Any]:
        assert len(x0_list) == len(x1_list) == 2
        n, c, h0, w0 = x0_list[0].shape
        _, _, h1, w1 = x1_list[0].shape
        _, _, fh0, fw0 = x0_list[1].shape
        _, _, fh1, fw1 = x1_list[1].shape
        scale = c**-0.5
        out = {}

        x0_ = x0_list[1].flatten(start_dim=-2) * scale
        x1_ = x1_list[1].flatten(start_dim=-2) * scale
        similarity = x0_.transpose(-2, -1) @ x1_
        if mask0 is not None and mask1 is not None:
            mask0_ = F.max_pool2d(mask0.float(), self.stride).bool()
            mask1_ = F.max_pool2d(mask1.float(), self.stride).bool()
            mask = mask0_.view(n, -1, 1) & mask1_.view(n, 1, -1)
            similarity.masked_fill_(~mask, -1e9)

        similarity_ = similarity
        if self.training:
            similarity = similarity / self.temperature
            confidence = similarity.softmax(dim=-1) * similarity.softmax(dim=-2)
            out["extra_coarse_cls_heatmap"] = confidence
            if gt_indices_list is not None:
                similarity_[gt_indices_list[1]] = 100

        _, indices0_to_1 = similarity_.topk(self.topk, dim=-1)
        _, indices1_to_0 = similarity_.transpose(-2, -1).topk(self.topk, dim=-1)
        x0_p, x1_p, attended0_p, attended1_p, indices0_to_1, indices1_to_0 = (
            self.fused_selective_module(
                x0_list, x1_list, indices0_to_1, indices1_to_0
            )
        )
        indices1_to_0 = indices1_to_0.transpose(-2, -1)
        x0 = window_unpartition(x0_p, (fh0, fw0), self.stride)
        x1 = window_unpartition(x1_p, (fh1, fw1), self.stride)
        out["x_8x"] = (x0, x1)

        if self.training:
            x0_ = x0.flatten(start_dim=-2) * scale
            x1_ = x1.flatten(start_dim=-2) * scale
            similarity = x0_.transpose(-2, -1) @ x1_
            if mask0 is not None and mask1 is not None:
                mask = mask0.view(n, -1, 1) & mask1.view(n, 1, -1)
                similarity.masked_fill_(~mask, -1e9)

            similarity = similarity / self.temperature
            confidence = similarity.softmax(dim=-1) * similarity.softmax(dim=-2)
            out["coarse_cls_heatmap"] = confidence
            score = confidence, indices0_to_1, indices1_to_0
        else:
            x0_p, x1_p = x0_p * scale, x1_p * scale
            attended0_p, attended1_p = attended0_p * scale, attended1_p * scale
            similarity0_to_1_p = x0_p @ attended1_p.transpose(-2, -1)
            similarity1_to_0_p = x1_p @ attended0_p.transpose(-2, -1)
            confidence0_to_1_p = (
                similarity0_to_1_p / self.temperature
            ).softmax(dim=-1)
            confidence1_to_0_p = (
                similarity1_to_0_p / self.temperature
            ).softmax(dim=-1)
            confidence0_to_1_ = (
                confidence0_to_1_p.view(
                    n, fh0, fw0, self.stride, self.stride, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(start_dim=1, end_dim=-2)
            )
            confidence1_to_0_ = (
                confidence1_to_0_p.view(
                    n, fh1, fw1, self.stride, self.stride, -1
                )
                .permute(0, 5, 1, 3, 2, 4)
                .flatten(start_dim=2, end_dim=-1)
            )
            confidence0_to_1 = confidence0_to_1_ * (
                x1.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(-2, indices1_to_0, confidence1_to_0_)
                .gather(-1, indices0_to_1)
            )
            confidence1_to_0 = confidence1_to_0_ * (
                x0.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(-1, indices0_to_1, confidence0_to_1_)
                .gather(-2, indices1_to_0)
            )
            score = (
                confidence0_to_1,
                confidence1_to_0,
                indices0_to_1,
                indices1_to_0,
            )

        out.update(
            self._create_coarse_matching(
                score,
                (h0, w0),
                (h1, w1),
                mask0=mask0,
                mask1=mask1,
                gt_idxes=gt_indices_list[0]
                if gt_indices_list is not None
                else None,
            )
        )
        return out
