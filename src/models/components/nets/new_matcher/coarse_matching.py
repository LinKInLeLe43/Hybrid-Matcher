from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class CoarseMatching(nn.Module):
    def __init__(
        self,
        depth: int,
        fused_selective_module: nn.Module,
        use_matchability: bool = False,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200
    ) -> None:
        super().__init__()
        self.fused_selective_module = fused_selective_module
        self.use_matchability = use_matchability
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        self.train_percent = train_percent
        self.train_min_gt_count = train_min_gt_count

        if use_matchability:
            self.matchability = nn.Sequential(
                nn.Linear(depth, depth // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(depth // 2, 1, bias=False))

    def _remove_border_for_train(
        self,
        x: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor]
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
                out[b, _h0 - r:, :, :, :] = False
                out[b, :, _w0 - r:, :, :] = False
                out[b, :, :, _h1 - r:, :] = False
                out[b, :, :, :, _w1 - r:] = False
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
        mask: Optional[torch.Tensor]
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
                out[b, h - r:, :] = 0
                out[b, :, w - r:] = 0
        else:
            out[:, -r:, :] = 0
            out[:, :, -r:] = 0
        out = out.flatten(start_dim=1)
        return out

    def _sample_for_train(
        self,
        max_count: int,
        matching_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = matching_idxes[0].device

        train_count = int(self.train_percent * max_count)
        rest_count = train_count - self.train_min_gt_count
        matching_count, gt_count = len(matching_idxes[0]), len(gt_idxes[0])
        if matching_count <= rest_count:
            matching_subidxes = torch.arange(matching_count, device=device)
        else:
            matching_subidxes = torch.randint(
                matching_count, (rest_count,), device=device)
            matching_count = rest_count
        gt_subidxes = torch.randint(
            gt_count, (train_count - matching_count,), device=device)

        matching_idxes = tuple(map(
            lambda x: x[matching_subidxes], matching_idxes))
        train_idxes = tuple(map(
            lambda x, y: torch.cat([x, y[gt_subidxes]]),
            matching_idxes, gt_idxes))
        return train_idxes, matching_idxes

    @torch.no_grad()
    def _create_coarse_matching(
        self,
        score: Union[torch.Tensor,
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
        gt_idxes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        if self.training:
            mask, max_count = self._remove_border_for_train(
                score > self.threshold, size0, size1, mask0, mask1)
            mask &= ((score == score.amax(dim=2, keepdim=True)) &
                     (score == score.amax(dim=1, keepdim=True)))
            train_idxes, matching_idxes = self._sample_for_train(
                max_count, mask.nonzero(as_tuple=True), gt_idxes)
            b_idxes, i_idxes, j_idxes = train_idxes
            scores = score[train_idxes]
        else:
            score, idxes0_to_1, idxes1_to_0 = score
            n, l0, l1 = score.shape
            device = score.device
            r = torch.arange(n, device=device)[:, None, None]
            r0 = torch.arange(l0, device=device)[None, :, None]
            r1 = torch.arange(l1, device=device)[None, None, :]

            values0_to_1, sub_idxes0_to_1 = score[r, r0, idxes0_to_1].max(dim=2)
            sub_idxes1_to_0 = score[r, idxes1_to_0, r1].argmax(dim=1)
            idxes0_to_1 = idxes0_to_1[r[:, :, 0], r0[:, :, 0], sub_idxes0_to_1]
            idxes1_to_0 = idxes1_to_0[r[:, 0, :], sub_idxes1_to_0, r1[:, 0, :]]
            idxes0_to_1 = self._remove_border_for_eval(
                idxes0_to_1, size0, mask0)
            idxes1_to_0 = self._remove_border_for_eval(
                idxes1_to_0, size1, mask1)
            biprojection = torch.stack([idxes1_to_0[b, idx1]
                                        for b, idx1 in enumerate(idxes0_to_1)])
            mask = biprojection == r0[:, :, 0]
            if self.border_removal > 0:
                mask[:, 0] = False
            mask &= values0_to_1 > self.threshold
            b_idxes, i_idxes = mask.nonzero(as_tuple=True)
            j_idxes = idxes0_to_1[b_idxes, i_idxes]
            train_idxes = matching_idxes = b_idxes, i_idxes, j_idxes
            scores = values0_to_1[b_idxes, i_idxes]

        points0 = torch.stack([i_idxes % size0[1],
                               i_idxes // size0[1]], dim=1).float()
        points1 = torch.stack([j_idxes % size1[1],
                               j_idxes // size1[1]], dim=1).float()
        result = {"idxes": train_idxes,
                  "points0": points0,
                  "points1": points1,
                  "scores": scores,
                  "coarse_cls_idxes": train_idxes}
        return result

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
        x_gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        y_gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n, c, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape

        _y0 = y0.flatten(start_dim=2).transpose(1, 2)
        _y1 = y1.flatten(start_dim=2).transpose(1, 2)
        _y0, _y1 = _y0 / c ** 0.5, _y1 / c ** 0.5
        similarity = torch.einsum("nlc,nsc->nls", _y0, _y1)
        similarity /= self.temperature
        if y0_mask is not None and y1_mask is not None:
            mask = (y0_mask.flatten(start_dim=1)[:, :, None] &
                    y1_mask.flatten(start_dim=1)[:, None, :])
            similarity.masked_fill_(~mask, -1e9)

        topk = 8
        result = {}
        if self.training:
            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)
            confidence = confidence0_to_1 * confidence1_to_0
            result["extra_coarse_cls_heatmap"] = confidence
            _similarity = similarity.clone()
            _similarity[y_gt_idxes] = 1e9
        else:
            _similarity = similarity

        _, idxes0_to_1 = _similarity.topk(topk, dim=2)
        _, idxes1_to_0 = _similarity.topk(topk, dim=1)

        (x0, x1, selective0, selective1,
         idxes0_to_1, idxes1_to_0) = self.fused_selective_module(
            x0, x1, y0, y1, idxes0_to_1, idxes1_to_0)
        _x0, _x1 = x0 / c ** 0.5, x1 / c ** 0.5
        _selective0 = selective0 / c ** 0.5
        _selective1 = selective1 / c ** 0.5

        if self.training:
            similarity = torch.einsum("nlc,nsc->nls", _x0, _x1)
            similarity /= self.temperature
            if x0_mask is not None and x1_mask is not None:
                mask = (x0_mask.flatten(start_dim=1)[:, :, None] &
                        x1_mask.flatten(start_dim=1)[:, None, :])
                similarity.masked_fill_(~mask, -1e9)

            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)
        else:
            similarity0_to_1 = torch.einsum("nlc,nlkc->nlk", _x0, _selective1)
            similarity1_to_0 = torch.einsum("nlkc,nlc->nkl", _selective0, _x1)
            similarity0_to_1 /= self.temperature
            similarity1_to_0 /= self.temperature
            confidence0_to_1 = F.softmax(similarity0_to_1, dim=2)
            confidence1_to_0 = F.softmax(similarity1_to_0, dim=1)
            confidence0_to_1 = x0.new_zeros((n, h0 * w0, h1 * w1)).scatter_(
                2, idxes0_to_1, confidence0_to_1)
            confidence1_to_0 = x0.new_zeros((n, h0 * w0, h1 * w1)).scatter_(
                1, idxes1_to_0, confidence1_to_0)

        heatmap = confidence = confidence0_to_1 * confidence1_to_0
        if self.training and self.use_matchability:
            m0 = self.matchability(x0).sigmoid()
            m1 = self.matchability(x1).sigmoid().transpose(1, 2)
            confidence *= m0 * m1
            heatmap = F.pad(confidence, (0, 1, 0, 1))
            heatmap[:, :-1, -1:] = 1 - m0
            heatmap[:, -1:, :-1] = 1 - m1

        score = (confidence if self.training else
                 (confidence, idxes0_to_1, idxes1_to_0))
        result.update(self._create_coarse_matching(
            score, (h0, w0), (h1, w1), x0_mask, x1_mask, x_gt_idxes))
        result["x_8x"] = (x0.transpose(1, 2).unflatten(2, (h0, w0)),
                          x1.transpose(1, 2).unflatten(2, (h1, w1)))
        result["coarse_cls_heatmap"] = heatmap
        return result
