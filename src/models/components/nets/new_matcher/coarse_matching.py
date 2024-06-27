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
        sub_stride: Optional[int] = None,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200
    ) -> None:
        super().__init__()

        if type not in ("bi_softmax", "uni_softmax", "bi_filter"):
            raise ValueError("")

        self.type = type
        self.threshold = threshold
        self.border_removal = border_removal
        self.temperature = temperature
        self.use_matchability = use_matchability
        self.sub_stride = sub_stride
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
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        prior_mask: Optional[torch.Tensor],
        mask0: Optional[torch.Tensor],
        mask1: Optional[torch.Tensor],
        only_return_mask: bool,
        gt_idxes: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        t = self.threshold
        (_, w0), (_, w1) = size0, size1

        if isinstance(score, tuple):
            score0_to_1, score1_to_0 = score

            mask0_to_1, max_count = self._remove_border(
                score0_to_1 > t, size0, size1, mask0, mask1)
            mask0_to_1 &= score0_to_1 == score0_to_1.amax(dim=2, keepdim=True)

            mask1_to_0, max_count = self._remove_border(
                score1_to_0 > t, size0, size1, mask0, mask1)
            mask1_to_0 &= score1_to_0 == score1_to_0.amax(dim=1, keepdim=True)

            mask = mask0_to_1 | mask1_to_0
        elif isinstance(score, torch.Tensor):
            mask, max_count = self._remove_border(
                score > t, size0, size1, mask0, mask1)
            mask &= ((score == score.amax(dim=2, keepdim=True)) &
                     (score == score.amax(dim=1, keepdim=True)))
        else:
            assert False

        if prior_mask is not None:
            mask &= prior_mask

        if only_return_mask:
            result = {"coarse_cls_mask": mask}
            return result

        train_idxes = matching_idxes = mask.nonzero(as_tuple=True)
        if self.training and gt_idxes is not None:
            train_idxes, matching_idxes = self._sample_for_train(
                max_count, matching_idxes, gt_idxes)

        if isinstance(score, tuple):
            scores = torch.maximum(
                score0_to_1[matching_idxes], score1_to_0[matching_idxes])
        elif isinstance(score, torch.Tensor):
            scores = score[matching_idxes]
        else:
            assert False

        if self.sub_stride is not None:
            s = self.sub_stride
            m = len(matching_idxes[0])

            b_idxes, i_idxes, j_idxes = train_idxes
            ix_idxes, iy_idxes = i_idxes % w0, i_idxes // w0
            jx_idxes, jy_idxes = j_idxes % w1, j_idxes // w1
            i_idxes = (w0 // s) * (iy_idxes // s) + (ix_idxes // s)
            j_idxes = (w1 // s) * (jy_idxes // s) + (jx_idxes // s)
            sub_i_idxes = s * (iy_idxes % s) + (ix_idxes % s)
            sub_j_idxes = s * (jy_idxes % s) + (jx_idxes % s)
            train_idxes = b_idxes, i_idxes, j_idxes
            train_sub_idxes = sub_i_idxes, sub_j_idxes
            matching_idxes = b_idxes[:m], i_idxes[:m], j_idxes[:m]
            w0, w1 = w0 // s, w1 // s

        b_idxes, i_idxes, j_idxes = matching_idxes
        points0 = torch.stack([i_idxes % w0, i_idxes // w0], dim=1).float()
        points1 = torch.stack([j_idxes % w1, j_idxes // w1], dim=1).float()

        result = {"idxes": matching_idxes,
                  "points0": points0,
                  "points1": points1,
                  "scores": scores,
                  "coarse_cls_idxes": train_idxes}

        if self.sub_stride is not None:
            result["coarse_cls_sub_idxes"] = train_sub_idxes
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
        only_return_mask: bool = False,
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n, l, c = x0.shape
        _, s, _ = x1.shape

        with torch.autocast(
            "cuda",
            enabled=(torch.is_autocast_enabled() or
                     (self.type == "bi_filter" and not self.training))
        ):
            x0, x1 = x0 / c ** 0.5, x1 / c ** 0.5
            similarity = torch.einsum("nlc,nsc->nls", x0, x1)
            similarity /= self.temperature
            if mask0 is not None and mask1 is not None:
                mask = (mask0.flatten(start_dim=1)[:, :, None] &
                        mask1.flatten(start_dim=1)[:, None, :])
                similarity.masked_fill_(~mask, float("-inf"))

        confidence = confidence_with_bin = None
        if self.type != "bi_filter" or self.training:
            confidence0_to_1 = F.softmax(similarity, dim=2)
            confidence1_to_0 = F.softmax(similarity, dim=1)

            if mask0 is not None and mask1 is not None:
                confidence0_to_1 = confidence0_to_1.nan_to_num()
                confidence1_to_0 = confidence1_to_0.nan_to_num()

            confidence = confidence0_to_1 * confidence1_to_0

            if self.training and self.use_matchability:
                if matchability0 is None or matchability1 is None:
                    raise ValueError("")

                confidence *= (matchability0[:, :, None] *
                               matchability1[:, None, :])
                confidence_with_bin = F.pad(confidence, [0, 1, 0, 1])
                confidence_with_bin[:, :-1, -1] = 1 - matchability0
                confidence_with_bin[:, -1, :-1] = 1 - matchability1

        if self.type == "bi_softmax" or type == "uni_softmax":
            score = confidence
        elif self.type == "uni_softmax":
            score = (confidence0_to_1, confidence1_to_0)
        elif self.type == "bi_filter":
            score = similarity
        else:
            assert False

        result = self._create_coarse_matching(
            score, size0, size1, prior_mask, mask0, mask1, only_return_mask,
            gt_idxes)

        if confidence_with_bin is not None:
            result["coarse_cls_heatmap"] = confidence_with_bin
        elif confidence is not None:
            result["coarse_cls_heatmap"] = confidence
        return result
