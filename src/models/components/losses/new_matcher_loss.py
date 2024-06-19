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
    device = heatmap.device
    m = len(heatmap)
    if m == 0:
        return loss_pos_weight * torch.tensor(1.0, device=device)

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
    use_bin = False
    cls_w, gt_w = heatmap.shape[2], gt_mask.shape[2]
    if cls_w == gt_w + 1:
        if sparse:
            bin0_mask = gt_mask.sum(dim=2) == 0
            bin1_mask = gt_mask.sum(dim=1) == 0
            if mask0 is not None:
                bin0_mask &= mask0.flatten(start_dim=1)
                bin1_mask &= mask1.flatten(start_dim=1)
            bins = torch.cat([heatmap[:, :-1, -1][bin0_mask],
                              heatmap[:, -1, :-1][bin1_mask]])
            use_bin = True
        heatmap = heatmap[:, :-1, :-1]

    pos_losses = _focal_loss(heatmap[pos_mask])
    if weight is not None:
        pos_losses *= weight[pos_mask]
    if sparse and not use_bin:
        loss = loss_pos_weight * pos_losses.mean()
        return loss

    if use_bin:
        neg_losses = _focal_loss(bins)
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
    reg_stds: Optional[torch.Tensor] = None,
    loss_weight: float = 1.0
) -> torch.Tensor:
    device = reg_biases.device
    m = len(reg_biases)
    if m == 0:
        return loss_weight * torch.tensor(1.0, device=device)

    weight = None
    if reg_stds is not None:
        weight = 1.0 / reg_stds.clamp(min=1e-10)
        weight /= weight.mean()
    mask = gt_biases.abs().amax(dim=1) < 1.0
    if not mask.any():
        mask[0] = True
        loss_weight = 0.0

    losses = ((reg_biases - gt_biases)[mask] ** 2).sum(dim=1)
    if weight is not None:
        losses *= weight[mask]
    loss = loss_weight * losses.mean()
    return loss


def _compute_flow_loss(  # TODO: change name to gaussian NLL
    flows_with_uncertainties: torch.Tensor,
    gt_flows: torch.Tensor,
    loss_weight: float = 1.0
) -> torch.Tensor:
    m = len(flows_with_uncertainties)
    if m == 1:
        loss_weight = 0.0

    flows, log_stds_mul_2 = flows_with_uncertainties.chunk(2, dim=1)
    l2_distances = (flows - gt_flows) ** 2
    losses = log_stds_mul_2 + (-log_stds_mul_2).exp() * l2_distances
    loss = loss_weight * losses.mean()
    return loss


class NewMatcherLoss(nn.Module):  # TODO: change name
    def __init__(
        self,
        type: str,
        coarse_cls_sparse: bool,
        coarse_cls_loss_pos_weight: float,
        coarse_cls_loss_neg_weight: float,
        fine_reg_loss_weight: float,
        fine_cls_sparse: Optional[bool] = None,
        fine_cls_loss_pos_weight: Optional[float] = None,
        fine_cls_loss_neg_weight: Optional[float] = None,
        use_flow: bool = False,
        flow_loss_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.type = type
        self.coarse_cls_sparse = coarse_cls_sparse
        self.coarse_cls_loss_pos_weight = coarse_cls_loss_pos_weight
        self.coarse_cls_loss_neg_weight = coarse_cls_loss_neg_weight
        self.fine_cls_sparse = None
        self.fine_cls_loss_pos_weight = None
        self.fine_cls_loss_neg_weight = None
        self.fine_reg_loss_weight = fine_reg_loss_weight
        self.use_flow = use_flow
        self.flow_loss_weight = None

        if type == "one_stage":
            pass
        elif type == "two_stage":
            if (fine_cls_sparse is None or
                fine_cls_loss_pos_weight is None or
                fine_cls_loss_neg_weight is None):
                raise ValueError("")

            self.fine_cls_sparse = fine_cls_sparse
            self.fine_cls_loss_pos_weight = fine_cls_loss_pos_weight
            self.fine_cls_loss_neg_weight = fine_cls_loss_neg_weight
        else:
            raise ValueError("")

        if use_flow:
            if flow_loss_weight is None:
                raise ValueError("")
            self.flow_loss_weight = flow_loss_weight

    def forward(
        self,
        coarse_cls_heatmap: torch.Tensor,
        coarse_gt_mask: torch.Tensor,
        fine_reg_biases: torch.Tensor,
        fine_gt_biases: torch.Tensor,
        fine_reg_stds: Optional[torch.Tensor] = None,
        fine_cls_heatmap: Optional[torch.Tensor] = None,
        fine_gt_mask: Optional[torch.Tensor] = None,
        flows_with_uncertainties0: Optional[torch.Tensor] = None,
        flows_with_uncertainties1: Optional[torch.Tensor] = None,
        gt_flows0: Optional[torch.Tensor] = None,
        gt_flows1: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        total_loss = 0.0
        loss = {"scalar": {}}

        coarse_cls_loss = _compute_cls_loss(
            self.coarse_cls_sparse, coarse_cls_heatmap, coarse_gt_mask,
            loss_pos_weight=self.coarse_cls_loss_pos_weight,
            loss_neg_weight=self.coarse_cls_loss_neg_weight, mask0=mask0,
            mask1=mask1)
        total_loss += coarse_cls_loss
        loss["scalar"]["coarse_cls_loss"] = coarse_cls_loss.detach().cpu()

        fine_reg_loss = _compute_reg_loss(
            fine_reg_biases, fine_gt_biases, reg_stds=fine_reg_stds,
            loss_weight=self.fine_reg_loss_weight)
        total_loss += fine_reg_loss
        loss["scalar"]["fine_reg_loss"] = fine_reg_loss.detach().cpu()

        if self.type == "two_stage":
            if fine_cls_heatmap is None or fine_gt_mask is None:
                raise ValueError("")
            fine_cls_loss = _compute_cls_loss(
                self.fine_cls_sparse, fine_cls_heatmap, fine_gt_mask,
                loss_pos_weight=self.fine_cls_loss_pos_weight,
                loss_neg_weight=self.fine_cls_loss_neg_weight)
            total_loss += fine_cls_loss
            loss["scalar"]["fine_cls_loss"] = fine_cls_loss.detach().cpu()

        if self.use_flow:
            if (flows_with_uncertainties0 is None or
                flows_with_uncertainties1 is None or gt_flows0 is None or
                gt_flows1 is None):
                raise ValueError("")

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
