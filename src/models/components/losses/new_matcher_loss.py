from typing import Any, Dict, Optional

import torch
from torch import nn


def _focal_loss(  # TODO: support NLL
    x: torch.Tensor,
    alpha: float = 0.25,  # TODO: open interface
    gamma: float = 2.0
) -> torch.Tensor:
    # TODO: check alpha in ELoFTR
    output = -alpha * (1 - x).pow(gamma) * x.log()
    return output


def _compute_cls_loss(
    sparse: bool,
    heatmap: torch.Tensor,
    gt_mask: torch.Tensor,
    loss_pos_weight: float = 1.0,
    loss_neg_weight: float = 1.0,
    mask0: Optional[torch.Tensor] = None,
    mask1: Optional[torch.Tensor] = None
) -> torch.Tensor:
    m = len(heatmap)

    if m == 0:
        return heatmap.new_tensor(1.0)

    weight = None
    if mask0 is not None:
        weight = (mask0.flatten(start_dim=1)[:, :, None] &
                  mask1.flatten(start_dim=1)[:, None, :]).float()
    pos_mask, neg_mask = gt_mask, ~gt_mask
    if not pos_mask.any():
        pos_mask[0, 0, 0] = True
        loss_pos_weight = 0.0
        if weight is not None:
            weight[0, 0, 0] = 0.0
    if not neg_mask.any():
        neg_mask[0, 0, 0] = True
        loss_neg_weight = 0.0
        if weight is not None:
            weight[0, 0, 0] = 0.0

    heatmap = heatmap.clamp(min=1e-6, max=1 - 1e-6)
    pos_losses = _focal_loss(heatmap[pos_mask])
    if weight is not None:
        pos_losses *= weight[pos_mask]
    if sparse:
        loss = loss_pos_weight * pos_losses.mean()
        return loss
    else:
        neg_losses = _focal_loss(1 - heatmap[neg_mask])
        if weight is not None:
            neg_losses *= weight[neg_mask]
    loss = (loss_pos_weight * pos_losses.mean() +
            loss_neg_weight * neg_losses.mean())
    return loss


def _compute_reg_loss(
    reg_biases: torch.Tensor,
    gt_biases: torch.Tensor,
    reg_stds: Optional[torch.Tensor],
    loss_weight: float = 1.0
) -> torch.Tensor:
    m = len(reg_biases)

    if m == 0:
        return reg_biases.new_tensor(1.0)

    weight = None
    if reg_stds is not None:
        weight = 1.0 / reg_stds.clamp(min=1e-10)
        weight /= weight.mean()
    mask = ((reg_biases.abs().amax(dim=1) < 1.0) &
            (gt_biases.abs().amax(dim=1) < 1.0))
    if not mask.any():
        mask[0] = True
        loss_weight = 0.0

    losses = ((reg_biases - gt_biases)[mask] ** 2).sum(dim=1)
    if weight is not None:
        losses *= weight[mask]
    loss = loss_weight * losses.mean()
    return loss


def _compute_dense_loss(
    reg_biases: torch.Tensor,
    gt_biases: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_weight: float = 1.0
) -> torch.Tensor:
    m = len(reg_biases)

    if m == 0:
        return reg_biases.new_tensor(1.0)

    mask = ((reg_biases.abs().amax(dim=1) < 1.0) &
            (gt_biases.abs().amax(dim=1) < 1.0) & valid_mask)
    if not mask.any():
        mask[0] = True
        loss_weight = 0.0

    losses = ((reg_biases - gt_biases)[mask] ** 2).sum(dim=1)
    loss = loss_weight * losses.mean()
    return loss


def _compute_flow_loss(  # TODO: change name to gaussian NLL
    flows_with_uncertainties: torch.Tensor,
    gt_flows: torch.Tensor,
    loss_weight: float = 1.0
) -> torch.Tensor:
    m = len(flows_with_uncertainties)

    if m <= 1:
        loss_weight = 0.0

    flows, log_stds_mul_2 = flows_with_uncertainties.chunk(2, dim=1)
    l2_distances = (flows - gt_flows) ** 2
    losses = log_stds_mul_2 + (-log_stds_mul_2).exp() * l2_distances
    loss = loss_weight * losses.mean()
    return loss


class NewMatcherLoss(nn.Module):  # TODO: change name
    def __init__(
        self,
        coarse_cls_sparse: Optional[bool] = None,
        coarse_cls_loss_pos_weight: Optional[float] = None,
        coarse_cls_loss_neg_weight: Optional[float] = None,
        extra_coarse_cls_sparse: Optional[bool] = None,
        extra_coarse_cls_loss_pos_weight: Optional[float] = None,
        extra_coarse_cls_loss_neg_weight: Optional[float] = None,
        refined_coarse_cls_sparse: Optional[bool] = None,
        refined_coarse_cls_loss_pos_weight: Optional[float] = None,
        refined_coarse_cls_loss_neg_weight: Optional[float] = None,
        fine_cls_sparse: Optional[bool] = None,
        fine_cls_loss_pos_weight: Optional[float] = None,
        fine_cls_loss_neg_weight: Optional[float] = None,
        fine_reg_loss_weight: Optional[float] = None,
        dense_reg_loss_weight: Optional[float] = None,
        flow_loss_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.coarse_cls_sparse = coarse_cls_sparse
        self.coarse_cls_loss_pos_weight = coarse_cls_loss_pos_weight
        self.coarse_cls_loss_neg_weight = coarse_cls_loss_neg_weight
        self.extra_coarse_cls_sparse = extra_coarse_cls_sparse
        self.extra_coarse_cls_loss_pos_weight = extra_coarse_cls_loss_pos_weight
        self.extra_coarse_cls_loss_neg_weight = extra_coarse_cls_loss_neg_weight
        self.refined_coarse_cls_sparse = refined_coarse_cls_sparse
        self.refined_coarse_cls_loss_pos_weight = refined_coarse_cls_loss_pos_weight
        self.refined_coarse_cls_loss_neg_weight = refined_coarse_cls_loss_neg_weight
        self.fine_cls_sparse = fine_cls_sparse
        self.fine_cls_loss_pos_weight = fine_cls_loss_pos_weight
        self.fine_cls_loss_neg_weight = fine_cls_loss_neg_weight
        self.fine_reg_loss_weight = fine_reg_loss_weight
        self.dense_reg_loss_weight = dense_reg_loss_weight
        self.flow_loss_weight = flow_loss_weight

    def forward(
        self,
        coarse_cls_heatmap: Optional[torch.Tensor] = None,
        coarse_gt_mask: Optional[torch.Tensor] = None,
        extra_coarse_cls_heatmap: Optional[torch.Tensor] = None,
        extra_coarse_gt_mask: Optional[torch.Tensor] = None,
        refined_coarse_cls_heatmap: Optional[torch.Tensor] = None,
        refined_coarse_gt_mask: Optional[torch.Tensor] = None,
        fine_cls_heatmap: Optional[torch.Tensor] = None,
        fine_gt_mask: Optional[torch.Tensor] = None,
        fine_reg_biases: Optional[torch.Tensor] = None,
        fine_gt_biases: Optional[torch.Tensor] = None,
        fine_reg_stds: Optional[torch.Tensor] = None,
        dense_reg_biases: Optional[torch.Tensor] = None,
        dense_gt_biases: Optional[torch.Tensor] = None,
        dense_valid_mask: Optional[torch.Tensor] = None,
        flows_with_uncertainties0: Optional[torch.Tensor] = None,
        flows_with_uncertainties1: Optional[torch.Tensor] = None,
        gt_flows0: Optional[torch.Tensor] = None,
        gt_flows1: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        extra_mask0: Optional[torch.Tensor] = None,
        extra_mask1: Optional[torch.Tensor] = None,
        refined_mask0: Optional[torch.Tensor] = None,
        refined_mask1: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        total_loss = 0.0
        loss = {"scalar": {}}

        if (self.coarse_cls_sparse is not None and
            self.coarse_cls_loss_pos_weight is not None and
            self.coarse_cls_loss_neg_weight is not None):
            if coarse_cls_heatmap is not None and coarse_gt_mask is not None:
                coarse_cls_loss = _compute_cls_loss(
                    self.coarse_cls_sparse, coarse_cls_heatmap, coarse_gt_mask,
                    loss_pos_weight=self.coarse_cls_loss_pos_weight,
                    loss_neg_weight=self.coarse_cls_loss_neg_weight,
                    mask0=mask0, mask1=mask1)
                total_loss += coarse_cls_loss
                loss["scalar"]["coarse_cls_loss"] = (
                    coarse_cls_loss.detach().cpu())

        if (self.extra_coarse_cls_sparse is not None and
            self.extra_coarse_cls_loss_pos_weight is not None and
            self.extra_coarse_cls_loss_neg_weight is not None):
            if (extra_coarse_cls_heatmap is not None and
                extra_coarse_gt_mask is not None):
                extra_coarse_cls_loss = _compute_cls_loss(
                    self.extra_coarse_cls_sparse, extra_coarse_cls_heatmap,
                    extra_coarse_gt_mask,
                    loss_pos_weight=self.extra_coarse_cls_loss_pos_weight,
                    loss_neg_weight=self.extra_coarse_cls_loss_neg_weight,
                    mask0=extra_mask0, mask1=extra_mask1)
                total_loss += extra_coarse_cls_loss
                loss["scalar"]["extra_coarse_cls_loss"] = (
                    extra_coarse_cls_loss.detach().cpu())

        if (self.refined_coarse_cls_sparse is not None and
            self.refined_coarse_cls_loss_pos_weight is not None and
            self.refined_coarse_cls_loss_neg_weight is not None):
            if (refined_coarse_cls_heatmap is not None and
                refined_coarse_gt_mask is not None):
                refined_coarse_cls_loss = _compute_cls_loss(
                    self.refined_coarse_cls_sparse, refined_coarse_cls_heatmap,
                    refined_coarse_gt_mask,
                    loss_pos_weight=self.refined_coarse_cls_loss_pos_weight,
                    loss_neg_weight=self.refined_coarse_cls_loss_neg_weight,
                    mask0=refined_mask0, mask1=refined_mask1)
                total_loss += refined_coarse_cls_loss
                loss["scalar"]["refined_coarse_cls_loss"] = (
                    refined_coarse_cls_loss.detach().cpu())

        if (self.fine_cls_sparse is not None and
            self.fine_cls_loss_pos_weight is not None and
            self.fine_cls_loss_neg_weight is not None):
            if fine_cls_heatmap is not None and fine_gt_mask is not None:
                fine_cls_loss = _compute_cls_loss(
                    self.fine_cls_sparse, fine_cls_heatmap, fine_gt_mask,
                    loss_pos_weight=self.fine_cls_loss_pos_weight,
                    loss_neg_weight=self.fine_cls_loss_neg_weight)
                total_loss += fine_cls_loss
                loss["scalar"]["fine_cls_loss"] = fine_cls_loss.detach().cpu()

        if self.fine_reg_loss_weight is not None:
            if fine_reg_biases is not None and fine_gt_biases is not None:
                fine_reg_loss = _compute_reg_loss(
                    fine_reg_biases, fine_gt_biases, fine_reg_stds,
                    loss_weight=self.fine_reg_loss_weight)
                total_loss += fine_reg_loss
                loss["scalar"]["fine_reg_loss"] = fine_reg_loss.detach().cpu()

        if self.dense_reg_loss_weight is not None:
            if (dense_reg_biases is not None and
                dense_gt_biases is not None and
                dense_valid_mask is not None):
                dense_reg_loss = _compute_dense_loss(
                    dense_reg_biases, dense_gt_biases, dense_valid_mask,
                    loss_weight=self.dense_reg_loss_weight)
                total_loss += dense_reg_loss
                loss["scalar"]["dense_reg_loss"] = dense_reg_loss.detach().cpu()

        if self.flow_loss_weight is not None:
            if (flows_with_uncertainties0 is not None and
                flows_with_uncertainties1 is not None and
                gt_flows0 is not None and
                gt_flows1 is not None):
                flow_loss0 = _compute_flow_loss(
                    flows_with_uncertainties0, gt_flows0,
                    loss_weight=self.flow_loss_weight)
                flow_loss1 = _compute_flow_loss(
                    flows_with_uncertainties1, gt_flows1,
                    loss_weight=self.flow_loss_weight)
                flow_loss = (flow_loss0 + flow_loss1) / 2
                total_loss += flow_loss
                loss["scalar"]["flow_loss"] = flow_loss.detach().cpu()

        loss["loss"] = total_loss
        loss["scalar"]["total_loss"] = total_loss.detach().cpu()
        return loss
