from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class CoarseMatching(nn.Module):
    def __init__(
        self,
        type: str,
        threshold: float = 0.2,
        border_removal: int = 2,
        temperature: float = 0.1,
        use_matchability: bool = False,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200
    ) -> None:
        super().__init__()
        self.type = type
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        self.use_matchability = use_matchability
        self.train_percent = train_percent
        self.train_min_gt_count = train_min_gt_count

        if type not in ("bisoftmax", "unisoftmax", "bifilter"):
            raise ValueError("")

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

        if mask0 is not None and mask1 is not None:
            hs0 = mask0.sum(dim=1).amax(dim=1)
            ws0 = mask0.sum(dim=2).amax(dim=1)
            hs1 = mask1.sum(dim=1).amax(dim=1)
            ws1 = mask1.sum(dim=2).amax(dim=1)
            max_count = torch.minimum(hs0 * ws0, hs1 * ws1).sum().item()
            for b, (_h0, _w0, _h1, _w1) in enumerate(zip(hs0, ws0, hs1, ws1)):
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
        score: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        threshold: float,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        prior_mask: Optional[torch.Tensor],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
        gt_idxes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        if self.type == "unisoftmax":
            score0_to_1, score1_to_0 = score

            mask0_to_1, _ = self._remove_border(
                score0_to_1 > threshold, size0, size1, mask0, mask1)
            mask0_to_1 &= score0_to_1 == score0_to_1.amax(dim=2, keepdim=True)

            mask1_to_0, _ = self._remove_border(
                score1_to_0 > threshold, size0, size1, mask0, mask1)
            mask1_to_0 &= score1_to_0 == score1_to_0.amax(dim=1, keepdim=True)

            mask = mask0_to_1 | mask1_to_0
            result = {"coarse_cls_mask": mask}
        elif self.type == "bisoftmax" or self.type == "bifilter":
            mask, max_count = self._remove_border(
                score > threshold, size0, size1, mask0, mask1)
            mask &= ((score == score.amax(dim=2, keepdim=True)) &
                     (score == score.amax(dim=1, keepdim=True)))
            if prior_mask is not None:
                mask &= prior_mask

            train_idxes = matching_idxes = mask.nonzero(as_tuple=True)
            if self.training and gt_idxes is not None:
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
        else:
            assert False
        return result

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        matchability0: Optional[torch.Tensor] = None,
        matchability1: Optional[torch.Tensor] = None,
        prior_mask: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n, l, c = x0.shape
        _, s, _ = x1.shape

        with torch.autocast(
            "cuda",
            enabled=(torch.is_autocast_enabled() or
                     (self.type == "bifilter" and not self.training))
        ):
            x0, x1 = x0 / c ** 0.5, x1 / c ** 0.5
            similarity = torch.einsum("nlc,nsc->nls", x0, x1)
            similarity /= self.temperature
            if mask0 is not None and mask1 is not None:
                mask = (mask0.flatten(start_dim=1)[:, :, None] &
                        mask1.flatten(start_dim=1)[:, None, :])
                similarity.masked_fill_(~mask, float("-inf"))

        coarse_cls_heatmap = None
        if (self.type == "unisoftmax" or self.type == "bisoftmax" or
            self.training):
            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)

            if mask0 is not None and mask1 is not None:
                confidence0_to_1 = confidence0_to_1.nan_to_num()
                confidence1_to_0 = confidence1_to_0.nan_to_num()

            coarse_cls_heatmap = confidence = (confidence0_to_1 *
                                               confidence1_to_0)
            score = ((confidence0_to_1, confidence1_to_0)
                     if self.type == "unisoftmax" else confidence)
            threshold = self.threshold
            prior_mask = None

            if self.training and self.use_matchability:
                if matchability0 is None or matchability1 is None:
                    raise ValueError("")

                confidence *= (matchability0[:, :, None] *
                               matchability1[:, None, :])
                confidence_with_bin = F.pad(confidence, [0, 1, 0, 1])
                confidence_with_bin[:, :-1, -1] = 1 - matchability0
                confidence_with_bin[:, -1, :-1] = 1 - matchability1
                coarse_cls_heatmap = confidence_with_bin
        elif self.type == "bifilter" and not self.training:
            score = similarity
            threshold = 0.0
            gt_idxes = None
        else:
            assert False

        result = self._create_coarse_matching(
            score, threshold, size0, size1, prior_mask, mask0, mask1, gt_idxes)
        result["coarse_cls_heatmap"] = coarse_cls_heatmap
        return result
