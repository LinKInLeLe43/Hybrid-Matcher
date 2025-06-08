from typing import Any, Dict, Optional, Sequence, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .submodules import PyramidFuser


class CoarseMatching(nn.Module):
    def __init__(
        self,
        scale: int,
        dims: Sequence[int],
        fused_selective_module: nn.Module,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(dims) == 2
        self.scale = scale
        self.fuser = PyramidFuser(dims, **kwargs)
        self.fused_selective_module = fused_selective_module
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature

    def _remove_border_for_train(
        self,
        x: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
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
        self,
        x: torch.Tensor,
        size: Tuple[int, int],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
        score: Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
        gt_idxes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
        self, indices0_to_1: torch.Tensor, indices1_to_0: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: torch.Tensor,
        y1: torch.Tensor,
        x0_mask: Optional[torch.Tensor] = None,
        x1_mask: Optional[torch.Tensor] = None,
        y0_mask: Optional[torch.Tensor] = None,
        y1_mask: Optional[torch.Tensor] = None,
        x_gt_idxes: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        y_gt_idxes: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Dict[str, Any]:
        n, c, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape
        sh = sw = self.scale
        fh0, fw0, fh1, fw1 = [t // self.scale for t in [h0, w0, h1, w1]]

        _y0 = y0.flatten(start_dim=2).transpose(1, 2)
        _y1 = y1.flatten(start_dim=2).transpose(1, 2)
        _y0, _y1 = _y0 / c**0.5, _y1 / c**0.5
        similarity = torch.einsum("nlc,nsc->nls", _y0, _y1)
        similarity /= self.temperature
        if y0_mask is not None and y1_mask is not None:
            mask = (
                y0_mask.flatten(start_dim=1)[:, :, None]
                & y1_mask.flatten(start_dim=1)[:, None, :]
            )
            similarity.masked_fill_(~mask, -1e9)

        topk = 8
        result = {}

        if self.training:
            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)
            confidence = confidence0_to_1 * confidence1_to_0
            result["extra_coarse_cls_heatmap"] = confidence

        _similarity = similarity
        # if self.training and y_gt_idxes is not None:
        #     _similarity = similarity.clone()
        #     _similarity[y_gt_idxes] = 1e9

        _, idxes0_to_1 = _similarity.topk(topk, dim=2)
        _, idxes1_to_0 = _similarity.topk(topk, dim=1)
        result["extra_idxes0_to_1"] = idxes0_to_1
        if y_gt_idxes is not None:
            extra_coarse_topk_recall_mask = self.create_bidirectional_mask(
                idxes0_to_1, idxes1_to_0
            )
            result["extra_coarse_topk_recall"] = extra_coarse_topk_recall_mask[
                y_gt_idxes
            ]

        x0, x1 = self.fuser([x0, y0], [x1, y1])
        x0 = (
            x0.reshape(n, c, fh0, sh, fw0, sw)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh0 * fw0, sh * sw, c)
        )
        x1 = (
            x1.reshape(n, c, fh1, sh, fw1, sw)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(n, fh1 * fw1, sh * sw, c)
        )
        (_x0, _x1, _selective0, _selective1, idxes0_to_1, idxes1_to_0) = (
            self.fused_selective_module(
                x0, x1, idxes0_to_1, idxes1_to_0, (h0, w0), (h1, w1)
            )
        )
        x0 = (
            _x0.reshape(n, fh0, fw0, sh, sw, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(n, c, h0, w0)
        )
        x1 = (
            _x1.reshape(n, fh1, fw1, sh, sw, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(n, c, h1, w1)
        )
        result["x_8x"] = (x0, x1)

        if self.training:
            x0, x1 = x0 / c**0.5, x1 / c**0.5
            similarity = torch.einsum(
                "nlc,nsc->nls",
                x0.flatten(start_dim=2).transpose(1, 2),
                x1.flatten(start_dim=2).transpose(1, 2),
            )
            similarity /= self.temperature
            if x0_mask is not None and x1_mask is not None:
                mask = (
                    x0_mask.flatten(start_dim=1)[:, :, None]
                    & x1_mask.flatten(start_dim=1)[:, None, :]
                )
                similarity.masked_fill_(~mask, -1e9)

            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)
            confidence = confidence0_to_1 * confidence1_to_0
            score = confidence, idxes0_to_1, idxes1_to_0
            result["coarse_cls_heatmap"] = confidence
        else:
            _x0, _x1 = _x0 / c**0.5, _x1 / c**0.5
            _selective0 = _selective0 / c**0.5
            _selective1 = _selective1 / c**0.5
            similarity0_to_1 = torch.einsum(
                "nmlc,nmsc->nmls", _x0, _selective1
            )
            similarity1_to_0 = torch.einsum(
                "nmsc,nmlc->nmsl", _selective0, _x1
            )
            similarity0_to_1 /= self.temperature
            similarity1_to_0 /= self.temperature
            _confidence0_to_1 = F.softmax(similarity0_to_1, dim=3)
            _confidence1_to_0 = F.softmax(similarity1_to_0, dim=2)
            _confidence0_to_1 = (
                _confidence0_to_1.reshape(n, fh0, fw0, sh, sw, -1)
                .transpose(2, 3)
                .flatten(start_dim=1, end_dim=4)
            )
            _confidence1_to_0 = (
                _confidence1_to_0.reshape(n, -1, fh1, fw1, sh, sw)
                .transpose(3, 4)
                .flatten(start_dim=2, end_dim=5)
            )
            confidence0_to_1 = _confidence0_to_1 * (
                x1.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(1, idxes1_to_0, _confidence1_to_0)
                .gather(2, idxes0_to_1)
            )
            confidence1_to_0 = _confidence1_to_0 * (
                x0.new_zeros(n, h0 * w0, h1 * w1)
                .scatter_(2, idxes0_to_1, _confidence0_to_1)
                .gather(1, idxes1_to_0)
            )
            score = (
                confidence0_to_1,
                confidence1_to_0,
                idxes0_to_1,
                idxes1_to_0,
            )

        result.update(
            self._create_coarse_matching(
                score, (h0, w0), (h1, w1), x0_mask, x1_mask, x_gt_idxes
            )
        )
        result["extra_idxes0_to_1"] = einops.repeat(
            result["extra_idxes0_to_1"],
            "n (fh fw) k -> n (fh sh fw sw) k",
            fh=h0 // 2,
            sh=2,
            fw=w0 // 2,
            sw=2,
        )[result["idxes"][0], result["idxes"][1]]
        return result
