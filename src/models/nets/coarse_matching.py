from typing import Any, Dict, Optional, Sequence, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from kornia import create_meshgrid
from torch import Tensor
from torch.nn import Module

from .utils import window_unpartition


class CoarseMatching(Module):
    def __init__(
        self, decoder: Module, threshold: float = 0.2, border_removal: int = 2
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.threshold = threshold
        self.border_removal = border_removal
        self.stride = decoder.stride_list[0]
        self.scale = decoder.dim_list[0] ** -0.5
        delta_indices = create_meshgrid(
            self.stride,
            self.stride,
            normalized_coordinates=False,
            dtype=torch.long,
        ).flatten(end_dim=-2)
        self.register_buffer("delta_indices", delta_indices, persistent=False)

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
        coarse_recall_mask = None
        if self.training and gt_idxes is not None:
            score, idxes0_to_1, idxes1_to_0 = score
            mask, max_count = self._remove_border_for_train(
                score > self.threshold, size0, size1, mask0, mask1
            )
            mask &= (score == score.amax(dim=2, keepdim=True)) & (
                score == score.amax(dim=1, keepdim=True)
            )
            train_idxes = matching_idxes = mask.nonzero(as_tuple=True)
            b_idxes, i_idxes, j_idxes = train_idxes
            scores = score[train_idxes]
        else:
            score0_to_1, score1_to_0, idxes0_to_1, idxes1_to_0 = score
            values0_to_1, sub_idxes0_to_1 = score0_to_1.max(dim=2)
            values1_to_0, sub_idxes1_to_0 = score1_to_0.max(dim=1)
            idxes0_to_1 = idxes0_to_1.gather(2, sub_idxes0_to_1[:, :, None])[
                :, :, 0
            ]
            idxes1_to_0 = idxes1_to_0.gather(1, sub_idxes1_to_0[:, None, :])[
                :, 0, :
            ]
            if gt_idxes is not None:
                coarse_recall_mask = self.create_bidirectional_mask(
                    idxes0_to_1[:, :, None], idxes1_to_0[:, None, :]
                )
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

        points0 = torch.stack(
            [i_idxes % size0[1], i_idxes // size0[1]], dim=1
        ).float()
        points1 = torch.stack(
            [j_idxes % size1[1], j_idxes // size1[1]], dim=1
        ).float()
        result = {
            "idxes": train_idxes,
            "points0": points0,
            "points1": points1,
            "scores": scores,
            "coarse_cls_idxes": train_idxes,
        }
        if coarse_recall_mask is not None:
            result["coarse_recall"] = coarse_recall_mask[gt_idxes]
        return result

    def create_bidirectional_mask(
        self, indices0_to_1: Tensor, indices1_to_0: Tensor
    ) -> Tensor:
        n, l0, _ = indices0_to_1.shape
        _, _, l1 = indices1_to_0.shape
        device = indices0_to_1.device

        b_indices = torch.arange(n)[:, None, None]
        i_indices = torch.arange(l0)[None, :, None]
        j_indices = torch.arange(l1)[None, None, :]

        mask = torch.zeros(n, l0, l1, device=device)
        mask[b_indices, i_indices, indices0_to_1] += 0.5
        mask[b_indices, indices1_to_0, j_indices] += 0.5
        mask.eq_(1.0)
        return mask

    def _map_indices(self, x: Tensor, size: Sequence[int], w: int) -> Tensor:
        row = (x[..., None] // w) * self.stride + self.delta_indices[:, 1]
        col = (x[..., None] % w) * self.stride + self.delta_indices[:, 0]
        x = (
            (row * w * self.stride + col)
            .view(x.shape[0], *size, -1)
            .repeat_interleave(self.stride, dim=1)
            .repeat_interleave(self.stride, dim=2)
            .flatten(start_dim=1, end_dim=2)
        )
        return x

    def forward(
        self,
        x0_list: Sequence[Tensor],
        x1_list: Sequence[Tensor],
        encoding: Tensor,
        x0_mask: Optional[Tensor] = None,
        x1_mask: Optional[Tensor] = None,
        y0_mask: Optional[Tensor] = None,
        y1_mask: Optional[Tensor] = None,
        z0_mask: Optional[Tensor] = None,
        z1_mask: Optional[Tensor] = None,
        x_gt_idxes: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        only_decode: bool = False,
    ) -> Dict[str, Any]:
        n, _, h0, w0 = x0_list[0].shape
        _, _, h1, w1 = x1_list[0].shape
        _, _, fh0, fw0 = x0_list[1].shape
        _, _, fh1, fw1 = x1_list[1].shape
        out = {}

        x0_, x1_, indices0_to_1_, indices1_to_0_, similarity_list = (
            self.decoder(
                x0_list, x1_list, encoding, mask0=z0_mask, mask1=z1_mask
            )
        )
        x0 = window_unpartition(x0_, (fh0, fw0), self.stride)
        x1 = window_unpartition(x1_, (fh1, fw1), self.stride)
        indices0_to_1 = self._map_indices(indices0_to_1_, (fh0, fw0), fw1)
        indices1_to_0 = self._map_indices(indices1_to_0_, (fh1, fw1), fw0)
        indices1_to_0 = indices1_to_0.transpose(-2, -1)
        out["extra_idxes0_to_1"] = indices0_to_1_
        out["feat"] = (x0, x1)
        if only_decode:
            return out

        if self.training:
            similarity = similarity_list.pop(-1)
            if z0_mask is not None and z1_mask is not None:
                mask = z0_mask.view(n, -1, 1) & z1_mask.view(n, 1, -1)
                similarity.masked_fill_(~mask, -float("inf"))
            confidence0_to_1 = F.softmax(similarity, dim=2).nan_to_num()
            confidence1_to_0 = F.softmax(similarity, dim=1).nan_to_num()
            confidence = confidence0_to_1 * confidence1_to_0
            confidence = (
                confidence.reshape(
                    n,
                    fh0 // self.stride,
                    fw0 // self.stride,
                    fh1 // self.stride,
                    fw1 // self.stride,
                )
                .repeat_interleave(self.stride**2, dim=1)
                .repeat_interleave(self.stride**2, dim=2)
                .repeat_interleave(self.stride**2, dim=3)
                .repeat_interleave(self.stride**2, dim=4)
                .reshape(n, h0 * w0, h1 * w1)
            )
            confidence = confidence.clamp(min=1e-6, max=1 - 1e-6).log()
            out["extra_extra_coarse_cls_heatmap"] = 0.25 * confidence

            similarity = similarity_list.pop(-1)
            if y0_mask is not None and y1_mask is not None:
                mask = y0_mask.view(n, -1, 1) & y1_mask.view(n, 1, -1)
                similarity.masked_fill_(~mask, -float("inf"))
            confidence0_to_1 = F.softmax(similarity, dim=2).nan_to_num()
            confidence1_to_0 = F.softmax(similarity, dim=1).nan_to_num()
            confidence = confidence0_to_1 * confidence1_to_0
            confidence = (
                confidence.reshape(n, fh0, fw0, fh1, fw1)
                .repeat_interleave(self.stride, dim=1)
                .repeat_interleave(self.stride, dim=2)
                .repeat_interleave(self.stride, dim=3)
                .repeat_interleave(self.stride, dim=4)
                .reshape(n, h0 * w0, h1 * w1)
            )
            confidence = confidence.clamp(min=1e-6, max=1 - 1e-6).log()
            out["extra_coarse_cls_heatmap"] = 0.25 * confidence

            x0 = x0.flatten(start_dim=2).transpose(1, 2) * self.scale
            x1 = x1.flatten(start_dim=2).transpose(1, 2)
            y_similarity = x0 @ x1.transpose(-1, -2)
            if x0_mask is not None and x1_mask is not None:
                mask = x0_mask.view(n, -1, 1) & x1_mask.view(n, 1, -1)
                y_similarity.masked_fill_(~mask, -float("inf"))

            confidence0_to_1 = F.softmax(y_similarity, dim=2).nan_to_num()
            confidence1_to_0 = F.softmax(y_similarity, dim=1).nan_to_num()
            confidence = confidence0_to_1 * confidence1_to_0
            confidence = confidence.clamp(min=1e-6, max=1 - 1e-6).log()
            score = confidence, indices0_to_1, indices1_to_0
            out["coarse_cls_heatmap"] = (
                confidence
                + out.pop("extra_coarse_cls_heatmap")
                + out.pop("extra_extra_coarse_cls_heatmap")
            )
        else:
            x0_ = x0_ * self.scale
            attended0 = x0_[
                torch.arange(n, device=x0_.device)[:, None, None],
                indices1_to_0_,
            ].flatten(start_dim=2, end_dim=3)
            attended1 = x1_[
                torch.arange(n, device=x1_.device)[:, None, None],
                indices0_to_1_,
            ].flatten(start_dim=2, end_dim=3)

            similarity0_to_1 = x0_ @ attended1.transpose(-1, -2)
            similarity1_to_0 = x1_ @ attended0.transpose(-1, -2)
            confidence0_to_1_ = F.softmax(similarity0_to_1, dim=3)
            confidence1_to_0_ = F.softmax(similarity1_to_0, dim=3)
            confidence0_to_1_ = (
                confidence0_to_1_.reshape(
                    n, fh0, fw0, self.stride, self.stride, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(start_dim=1, end_dim=-2)
            )
            confidence1_to_0_ = (
                confidence1_to_0_.reshape(
                    n, fh1, fw1, self.stride, self.stride, -1
                )
                .permute(0, 5, 1, 3, 2, 4)
                .flatten(start_dim=2, end_dim=-1)
            )
            confidence0_to_1 = confidence0_to_1_ * (
                x1.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(1, indices1_to_0, confidence1_to_0_)
                .gather(2, indices0_to_1)
            )
            confidence1_to_0 = confidence1_to_0_ * (
                x0.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(2, indices0_to_1, confidence0_to_1_)
                .gather(1, indices1_to_0)
            )
            score = (
                confidence0_to_1,
                confidence1_to_0,
                indices0_to_1,
                indices1_to_0,
            )

        out.update(
            self._create_coarse_matching(
                score, (h0, w0), (h1, w1), x0_mask, x1_mask, x_gt_idxes
            )
        )
        out["extra_idxes0_to_1"] = einops.repeat(
            out["extra_idxes0_to_1"],
            "n (fh fw) k -> n (fh sh fw sw) k",
            fh=h0 // 2,
            sh=2,
            fw=w0 // 2,
            sw=2,
        )[out["idxes"][0], out["idxes"][1]]
        return out
