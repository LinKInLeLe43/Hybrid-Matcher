from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CoarseMatching(nn.Module):
    def __init__(
        self,
        use_matchability: bool = False,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200
    ) -> None:
        super().__init__()
        self.use_matchability = use_matchability
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        self.train_percent = train_percent
        self.train_min_gt_count = train_min_gt_count

    def _remove_border(
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
        score: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
        gt_idxes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        mask, max_count = self._remove_border(
            score > self.threshold, size0, size1, mask0, mask1)
        mask &= ((score == score.amax(dim=2, keepdim=True)) &
                 (score == score.amax(dim=1, keepdim=True)))

        train_idxes = matching_idxes = mask.nonzero(as_tuple=True)
        if self.training:
            train_idxes, matching_idxes = self._sample_for_train(
                max_count, matching_idxes, gt_idxes)

        b_idxes, i_idxes, j_idxes = matching_idxes
        points0 = torch.stack([i_idxes % size0[1],
                               i_idxes // size0[1]], dim=1).float()
        points1 = torch.stack([j_idxes % size1[1],
                               j_idxes // size1[1]], dim=1).float()
        scores = score[matching_idxes]
        result = {"idxes": matching_idxes,
                  "points0": points0,
                  "points1": points1,
                  "scores": scores,
                  "coarse_cls_idxes": train_idxes}
        return result

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        m0: Optional[torch.Tensor],
        m1: Optional[torch.Tensor],
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        _, c, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape

        x0 = x0.flatten(start_dim=2).transpose(1, 2)
        x1 = x1.flatten(start_dim=2).transpose(1, 2)
        x0, x1 = x0 / c ** 0.5, x1 / c ** 0.5
        similarity = torch.einsum("nlc,nsc->nls", x0, x1)
        similarity /= self.temperature
        if mask0 is not None and mask1 is not None:
            mask = (mask0.flatten(start_dim=1)[:, :, None] &
                    mask1.flatten(start_dim=1)[:, None, :])
            similarity.masked_fill_(~mask, -1e9)

        confidence0_to_1 = F.softmax(similarity, dim=2)
        confidence1_to_0 = F.softmax(similarity, dim=1)
        confidence = confidence0_to_1 * confidence1_to_0

        heatmap = confidence
        if self.training and self.use_matchability:
            if m0 is None or m1 is None:
                raise ValueError("")
            confidence = (m0[:, :, None] *
                          m1[:, None, :] * confidence)
            heatmap = F.pad(confidence, (0, 1, 0, 1))
            heatmap[:, :-1, -1] = 1 - m0
            heatmap[:, -1, :-1] = 1 - m1

        result = self._create_coarse_matching(
            confidence, (h0, w0), (h1, w1), mask0, mask1, gt_idxes)
        result["coarse_cls_heatmap"] = heatmap
        return result
