from typing import Any, Dict, Optional

import torch
from torch import nn


class LoFTRLoss(nn.Module):
    def __init__(
        self,
        coarse_type: str,
        coarse_sparse: bool,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_flow: bool = False,
        flow_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.coarse_type = coarse_type
        self.coarse_sparse = coarse_sparse
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_flow = use_flow

        if use_flow:
            if flow_weight is None:
                raise ValueError("")
            self.flow_weight = flow_weight

    def _compute_focal_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.focal_alpha * (1 - x).pow(self.focal_gamma) * x.log()

    def compute_coarse_loss(
        self,
        coarse_confidences: torch.Tensor,
        gt_mask: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        valid_weight = None
        if mask0 is not None:
            valid_weight = (mask0.flatten(start_dim=1)[:, :, None] &
                            mask1.flatten(start_dim=1)[:, None, :]).float()

        pos_mask, neg_mask = gt_mask, ~gt_mask
        pos_weight = neg_weight = 1.0
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            pos_weight = 0.0
            if valid_weight is not None:
                valid_weight[0, 0, 0] = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_weight = 0.0
            if valid_weight is not None:
                valid_weight[0, 0, 0] = 0.0

        std_confidences = confidences = coarse_confidences.clamp(
            min=1e-6, max=1 - 1e-6)
        if self.coarse_type == "optimal_transport" and self.coarse_sparse:
            std_confidences = confidences[:, :-1, :-1]
        pos_losses = self._compute_focal_loss(std_confidences[pos_mask])
        if valid_weight is not None:
            pos_losses *= valid_weight[pos_mask]
        if self.coarse_sparse:
            if self.coarse_type == "dual_softmax":
                pos_loss = pos_weight * pos_losses.mean()
                return pos_loss
            elif self.coarse_type == "optimal_transport":
                bin0_mask = gt_mask.sum(dim=2) == 0
                bin1_mask = gt_mask.sum(dim=1) == 0
                bin_confidences = torch.cat(
                    [confidences[:, :-1, -1][bin0_mask],
                     confidences[:, -1, :-1][bin1_mask]])
                neg_losses = self._compute_focal_loss(bin_confidences)
                if mask0 is not None:
                    valid_bin_mask = torch.cat([mask0[bin0_mask],
                                                mask1[bin1_mask]])
                    neg_losses = neg_losses[valid_bin_mask]
            else:
                raise ValueError("")
        else:
            neg_losses = self._compute_focal_loss(1 - std_confidences[neg_mask])
            if valid_weight is not None:
                neg_losses *= valid_weight[neg_mask]
        loss = pos_weight * pos_losses.mean() + neg_weight * neg_losses.mean()
        return loss

    def compute_hard_neg_coarse_loss(
        self,
        coarse_confidences: torch.Tensor,
        gt_mask: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        valid_weight = None
        if mask0 is not None:
            valid_weight = (mask0.flatten(start_dim=1)[:, :, None] &
                            mask1.flatten(start_dim=1)[:, None, :]).float()

        pos_mask = gt_mask
        pos_weight = 1.0
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            pos_weight = 0.0
            if valid_weight is not None:
                valid_weight[0, 0, 0] = 0.0

        std_confidences = confidences = coarse_confidences.clamp(
            min=1e-6, max=1 - 1e-6)
        if self.coarse_type == "optimal_transport" and self.coarse_sparse:
            std_confidences = confidences[:, :-1, :-1]

        neg_mask = torch.ones_like(std_confidences)
        neg_mask[gt_mask] = std_confidences[gt_mask]
        neg_mask = torch.minimum(
            neg_mask.min(dim=2, keepdim=True),
            neg_mask.min(dim=1, keepdim=True))
        neg_mask = coarse_confidences > neg_mask
        neg_weight = 1.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_weight = 0.0
            if valid_weight is not None:
                valid_weight[0, 0, 0] = 0.0

        pos_losses = self._compute_focal_loss(std_confidences[pos_mask])
        if valid_weight is not None:
            pos_losses *= valid_weight[pos_mask]
        if self.coarse_sparse:
            if self.coarse_type == "dual_softmax":
                pos_loss = pos_weight * pos_losses.mean()
                return pos_loss
            elif self.coarse_type == "optimal_transport":
                bin0_mask = gt_mask.sum(dim=2) == 0
                bin1_mask = gt_mask.sum(dim=1) == 0
                bin_confidences = torch.cat(
                    [confidences[:, :-1, -1][bin0_mask],
                     confidences[:, -1, :-1][bin1_mask]])
                neg_losses = self._compute_focal_loss(bin_confidences)
                if mask0 is not None:
                    valid_bin_mask = torch.cat([mask0[bin0_mask],
                                                mask1[bin1_mask]])
                    neg_losses = neg_losses[valid_bin_mask]
            else:
                raise ValueError("")
        else:
            neg_losses = self._compute_focal_loss(1 - std_confidences[neg_mask])
            if valid_weight is not None:
                neg_losses *= valid_weight[neg_mask]
        loss = pos_weight * pos_losses.mean() + neg_weight * neg_losses.mean()
        return loss

    def compute_fine_loss(
        self,
        fine_biases: torch.Tensor,
        gt_biases: torch.Tensor,
        fine_stddevs: torch.Tensor
    ) -> torch.Tensor:
        mask = gt_biases.abs().amax(dim=1) < 1.0
        weight = 1.0 / fine_stddevs.clamp(min=1e-10)
        weight /= weight.mean()
        if not mask.any():
            if self.training:
                mask[0] = True
                weight[0] = 0.0
            else:
                return fine_biases.new_ones(())
        loss = (weight[mask] *
                ((fine_biases[mask] - gt_biases[mask]) ** 2).sum(dim=1)).mean()
        return loss

    def _compute_single_flow_loss(
        self,
        flow: torch.Tensor,
        gt_coor: torch.Tensor
    ) -> torch.Tensor:
        weight = 1.0 if len(flow) != 1 else 0.0
        flow_coor, flow_log_stddev_mul_2 = flow.chunk(2, dim=1)
        distances = (flow_coor - gt_coor) ** 2
        loss = (flow_log_stddev_mul_2 +
                (-flow_log_stddev_mul_2).exp() * distances).mean()
        loss *= weight
        return loss

    def compute_flow_loss(
        self,
        flow0_to_1: torch.Tensor,
        flow1_to_0: torch.Tensor,
        gt_coor0_to_1: torch.Tensor,
        gt_coor1_to_0: torch.Tensor
    ) -> torch.Tensor:
        loss0 = self._compute_single_flow_loss(flow0_to_1, gt_coor0_to_1)
        loss1 = self._compute_single_flow_loss(flow1_to_0, gt_coor1_to_0)
        loss = (loss0 + loss1) / 2
        loss *= self.flow_weight
        return loss

    def forward(
        self,
        coarse_confidences: torch.Tensor,
        gt_mask: torch.Tensor,
        fine_biases: torch.Tensor,
        gt_biases: torch.Tensor,
        fine_stddevs: torch.Tensor,
        flow0_to_1: Optional[torch.Tensor] = None,
        flow1_to_0: Optional[torch.Tensor] = None,
        gt_coor0_to_1: Optional[torch.Tensor] = None,
        gt_coor1_to_0: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        coarse_loss = self.compute_hard_neg_coarse_loss(
            coarse_confidences, gt_mask, mask0=mask0, mask1=mask1)
        fine_loss = self.compute_fine_loss(fine_biases, gt_biases, fine_stddevs)
        total_loss = coarse_loss + fine_loss
        loss = {"scalar": {"coarse_loss": coarse_loss.detach().cpu(),
                           "fine_loss": fine_loss.detach().cpu()}}

        if self.use_flow:
            if (flow0_to_1 is None or flow1_to_0 is None or
                gt_coor0_to_1 is None or gt_coor1_to_0 is None):
                raise ValueError("")
            flow_loss = self.compute_flow_loss(
                flow0_to_1, flow1_to_0, gt_coor0_to_1, gt_coor1_to_0)
            total_loss += flow_loss
            loss["scalar"]["flow_loss"] = flow_loss.detach().cpu()

        loss["loss"] = total_loss
        loss["scalar"]["total_loss"] = total_loss.detach().cpu()
        return loss
