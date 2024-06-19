from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.nets.loftr import optimal_transport


class CoarseMatching(nn.Module):  # TODO: change name to first stage
    def __init__(
        self,
        type: str,
        sparse: bool,
        use_matchability: bool = False,
        threshold: float = 0.2,
        margin_remove: int = 2,
        train_percent: float = 0.2,
        train_min_gt_count: int = 200,
        ds_temperature: float = 0.1,
        ot_filter_bin: bool = True,
        ot_bin_score: float = 1.0,
        ot_its_count: int = 3
    ) -> None:
        super().__init__()
        self.type = type
        self.sparse = sparse
        self.use_matchability = use_matchability
        self.threshold = threshold
        self.margin_remove = margin_remove
        self.train_percent = train_percent
        self.train_min_gt_count = train_min_gt_count

        if type == "dual_softmax":
            self.temp = ds_temperature
        elif type == "optimal_transport":
            self.temp = 1.0
            self.ot_filter_bin = ot_filter_bin
            self.ot_bin_score = nn.Parameter(torch.tensor(ot_bin_score))
            self.ot_its_count = ot_its_count
        else:
            raise ValueError("")

    def _remove_mask_margin(
        self,
        mask: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.margin_remove

        mask[:, :b, :, :, :] = False
        mask[:, :, :b, :, :] = False
        mask[:, :, :, :b, :] = False
        mask[:, :, :, :, :b] = False

        if mask0 is not None:
            h0s = mask0.sum(dim=1).amax(dim=1).int()
            w0s = mask0.sum(dim=2).amax(dim=1).int()
            h1s = mask1.sum(dim=1).amax(dim=1).int()
            w1s = mask1.sum(dim=2).amax(dim=1).int()
            for n, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
                mask[n, h0 - b:, :, :, :] = False
                mask[n, :, w0 - b:, :, :] = False
                mask[n, :, :, h1 - b:, :] = False
                mask[n, :, :, :, w1 - b:] = False
        else:
            n, h0, w0, h1, w1 = mask.shape
            ones = mask.new_ones((n,), dtype=torch.int)
            h0s, w0s, h1s, w1s = h0 * ones, w0 * ones, h1 * ones, w1 * ones
            if b > 0:
                mask[:, -b:, :, :, :] = False
                mask[:, :, -b:, :, :] = False
                mask[:, :, :, -b:, :] = False
                mask[:, :, :, :, -b:] = False
        return h0s, w0s, h1s, w1s

    def _sample_for_train(
        self,
        train_count: int,
        matching_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = matching_idxes[0].device

        matching_count, gt_count = len(matching_idxes[0]), len(gt_idxes[0])
        rest_count = train_count - self.train_min_gt_count
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
        confidences: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        (h0, w0), (h1, w1) = size0, size1

        mask = (confidences > self.threshold).reshape(-1, h0, w0, h1, w1)
        if mask0 is not None:
            mask0, mask1 = mask0.reshape(-1, h0, w0), mask1.reshape(-1, h1, w1)
        h0s, w0s, h1s, w1s = self._remove_mask_margin(
            mask, mask0=mask0, mask1=mask1)
        mask = mask.reshape(-1, h0 * w0, h1 * w1)

        idxes0_to_1_mask = confidences == confidences.amax(dim=2, keepdim=True)
        idxes1_to_0_mask = confidences == confidences.amax(dim=1, keepdim=True)
        mask &= idxes0_to_1_mask & idxes1_to_0_mask

        train_idxes = matching_idxes = mask.nonzero(as_tuple=True)
        if self.training:
            max_count = torch.stack(
                [h0s * w0s, h1s * w1s], dim=1).amin(dim=1).sum()
            train_count = int(self.train_percent * max_count)
            train_idxes, matching_idxes = self._sample_for_train(
                train_count, matching_idxes, gt_idxes)

        b_idxes, i_idxes, j_idxes = matching_idxes
        points0 = torch.stack([i_idxes % w0, i_idxes // w0], dim=1).float()
        points1 = torch.stack([j_idxes % w1, j_idxes // w1], dim=1).float()
        confidences = confidences[matching_idxes]
        coarse_matching = {"idxes": matching_idxes,
                           "points0": points0,
                           "points1": points1,
                           "confidences": confidences,
                           "coarse_cls_idxes": train_idxes}
        return coarse_matching

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        size0: Tuple[int, int],
        size1: Tuple[int, int],
        matchability0: Optional[torch.Tensor] = None,
        matchability1: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        gt_idxes:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        n, l, c = x0.shape
        _, s, _ = x1.shape

        similarities = torch.einsum("nlc,nsc->nls", x0, x1) / c
        similarities /= self.temp
        if mask0 is not None and mask1 is not None:
            mask = (mask0.flatten(start_dim=1)[:, :, None] &
                    mask1.flatten(start_dim=1)[:, None, :])
            similarities.masked_fill_(~mask, -1e9)

        confidences_with_bin = None
        if self.type == "dual_softmax":
            idxes0_to_1_confidences = F.softmax(similarities, dim=2)
            idxes1_to_0_confidences = F.softmax(similarities, dim=1)
            confidences = idxes0_to_1_confidences * idxes1_to_0_confidences
            if self.training and self.use_matchability:
                if matchability0 is None or matchability1 is None:
                    raise ValueError("")

                confidences = (matchability0[:, :, None] *
                               matchability1[:, None, :] * confidences)
                confidences_with_bin = F.pad(confidences, [0, 1, 0, 1])
                confidences_with_bin[:, :-1, -1] = 1 - matchability0
                confidences_with_bin[:, -1, :-1] = 1 - matchability1
        elif self.type == "optimal_transport":
            confidences_with_bin = optimal_transport.log_optimal_transport(
                similarities, self.ot_bin_score, self.ot_its_count).exp()
            confidences = confidences_with_bin[:, :l, :s]
            if not self.training and self.ot_filter_bin:
                bin0_mask = confidences_with_bin.argmax(dim=2) == s
                confidences.masked_fill_(bin0_mask[:, :l, None], 0.0)
                bin1_mask = confidences_with_bin.argmax(dim=1) == l
                confidences.masked_fill_(bin1_mask[:, None, :s], 0.0)
        else:
            assert False

        coarse_matching = self._create_coarse_matching(
            confidences, size0, size1, mask0=mask0, mask1=mask1,
            gt_idxes=gt_idxes)
        if confidences_with_bin is not None and self.sparse:
            coarse_matching["coarse_cls_heatmap"] = confidences_with_bin
        else:
            coarse_matching["coarse_cls_heatmap"] = confidences
        return coarse_matching
