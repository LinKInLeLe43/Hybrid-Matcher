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
    type: str,
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
    bin0_mask = gt_mask.sum(dim=2) == 0
    bin1_mask = gt_mask.sum(dim=1) == 0
    if type == "semi_dense":
        neg_mask[bin0_mask[:, :, None] & bin1_mask[:, None, :]] = False
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
        if type == "sparse":
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
    if type == "sparse" and not use_bin:
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
    device = flows_with_uncertainties.device
    m = len(flows_with_uncertainties)
    if m == 0:
        return loss_weight * torch.tensor(1.0, device=device)

    flows, log_stds_mul_2 = flows_with_uncertainties.chunk(2, dim=1)
    l2_distances = (flows - gt_flows) ** 2
    losses = log_stds_mul_2 + (-log_stds_mul_2).exp() * l2_distances
    loss = loss_weight * losses.mean()
    return loss


class NewMatcherLoss(nn.Module):  # TODO: change name
    def __init__(
        self,
        type: str,
        first_stage_cls_type: Optional[str] = None,
        first_stage_cls_loss_pos_weight: Optional[float] = None,
        first_stage_cls_loss_neg_weight: Optional[float] = None,
        second_stage_cls_type: Optional[str] = None,
        second_stage_cls_loss_pos_weight: Optional[float] = None,
        second_stage_cls_loss_neg_weight: Optional[float] = None,
        reg_loss_weight: Optional[float] = None,
        use_flow: bool = False,
        flow_loss_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.type = type
        self.first_stage_cls_type = None
        self.first_stage_cls_loss_pos_weight = None
        self.first_stage_cls_loss_neg_weight = None
        self.second_stage_cls_type = None
        self.second_stage_cls_loss_pos_weight = None
        self.second_stage_cls_loss_neg_weight = None
        self.reg_loss_weight = None
        self.use_flow = use_flow
        self.flow_loss_weight = None

        if type == "one_stage":
            if (first_stage_cls_type is None or
                first_stage_cls_loss_pos_weight is None or
                first_stage_cls_loss_neg_weight is None or
                reg_loss_weight is None):
                raise ValueError("")

            self.first_stage_cls_type = first_stage_cls_type
            self.first_stage_cls_loss_pos_weight = (
                first_stage_cls_loss_pos_weight)
            self.first_stage_cls_loss_neg_weight = (
                first_stage_cls_loss_neg_weight)
            self.reg_loss_weight = reg_loss_weight
        elif type == "two_stage":
            if (first_stage_cls_type is None or
                first_stage_cls_loss_pos_weight is None or
                first_stage_cls_loss_neg_weight is None or
                second_stage_cls_type is None or
                second_stage_cls_loss_pos_weight is None or
                second_stage_cls_loss_neg_weight is None or
                reg_loss_weight is None):
                raise ValueError("")

            self.first_stage_cls_type = first_stage_cls_type
            self.first_stage_cls_loss_pos_weight = (
                first_stage_cls_loss_pos_weight)
            self.first_stage_cls_loss_neg_weight = (
                first_stage_cls_loss_neg_weight)
            self.second_stage_cls_type = second_stage_cls_type
            self.second_stage_cls_loss_pos_weight = (
                second_stage_cls_loss_pos_weight)
            self.second_stage_cls_loss_neg_weight = (
                second_stage_cls_loss_neg_weight)
            self.reg_loss_weight = reg_loss_weight
        else:
            raise ValueError("")

        if use_flow:
            if flow_loss_weight is None:
                raise ValueError("")
            self.flow_loss_weight = flow_loss_weight

    def forward(
        self,
        first_stage_cls_heatmap_8x: Optional[torch.Tensor] = None,
        first_stage_gt_mask_8x: Optional[torch.Tensor] = None,
        first_stage_cls_heatmap_16x: Optional[torch.Tensor] = None,
        first_stage_gt_mask_16x: Optional[torch.Tensor] = None,
        second_stage_cls_heatmap: Optional[torch.Tensor] = None,
        second_stage_gt_mask: Optional[torch.Tensor] = None,
        reg_biases: Optional[torch.Tensor] = None,
        gt_biases: Optional[torch.Tensor] = None,
        reg_stds: Optional[torch.Tensor] = None,
        flows_with_uncertainties0: Optional[torch.Tensor] = None,
        flows_with_uncertainties1: Optional[torch.Tensor] = None,
        gt_flows0: Optional[torch.Tensor] = None,
        gt_flows1: Optional[torch.Tensor] = None,
        mask0_8x: Optional[torch.Tensor] = None,
        mask1_8x: Optional[torch.Tensor] = None,
        mask0_16x: Optional[torch.Tensor] = None,
        mask1_16x: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        total_loss = 0.0
        loss = {"scalar": {}}

        if self.type == "one_stage":
            if (first_stage_cls_heatmap_8x is None or
                first_stage_gt_mask_8x is None or
                first_stage_cls_heatmap_16x is None or
                first_stage_gt_mask_16x is None or
                reg_biases is None or
                gt_biases is None):
                raise ValueError("")

            first_stage_cls_loss_8x = _compute_cls_loss(
                self.first_stage_cls_type, first_stage_cls_heatmap_8x,
                first_stage_gt_mask_8x,
                loss_pos_weight=self.first_stage_cls_loss_pos_weight,
                loss_neg_weight=self.first_stage_cls_loss_neg_weight,
                mask0=mask0_8x, mask1=mask1_8x)
            total_loss += first_stage_cls_loss_8x
            loss["scalar"]["first_stage_cls_loss_8x"] = (
                first_stage_cls_loss_8x.detach().cpu())

            reg_loss = _compute_reg_loss(
                reg_biases, gt_biases, reg_stds=reg_stds,
                loss_weight=self.reg_loss_weight)
            total_loss += reg_loss
            loss["scalar"]["reg_loss"] = reg_loss.detach().cpu()
        elif self.type == "two_stage":
            if (first_stage_cls_heatmap_8x is None or
                first_stage_gt_mask_8x is None or
                first_stage_cls_heatmap_16x is None or
                first_stage_gt_mask_16x is None or
                second_stage_cls_heatmap is None or
                second_stage_gt_mask is None or
                reg_biases is None or
                gt_biases is None):
                raise ValueError("")

            first_stage_cls_loss_8x = _compute_cls_loss(
                self.first_stage_cls_type, first_stage_cls_heatmap_8x,
                first_stage_gt_mask_8x,
                loss_pos_weight=self.first_stage_cls_loss_pos_weight,
                loss_neg_weight=self.first_stage_cls_loss_neg_weight,
                mask0=mask0_8x, mask1=mask1_8x)
            total_loss += first_stage_cls_loss_8x
            loss["scalar"]["first_stage_cls_loss_8x"] = (
                first_stage_cls_loss_8x.detach().cpu())

            first_stage_cls_loss_16x = _compute_cls_loss(
                self.first_stage_cls_type, first_stage_cls_heatmap_16x,
                first_stage_gt_mask_16x,
                loss_pos_weight=self.first_stage_cls_loss_pos_weight,
                loss_neg_weight=self.first_stage_cls_loss_neg_weight,
                mask0=mask0_16x, mask1=mask1_16x)
            total_loss += first_stage_cls_loss_16x
            loss["scalar"]["first_stage_cls_loss_16x"] = (
                first_stage_cls_loss_16x.detach().cpu())

            second_stage_cls_loss = _compute_cls_loss(
                self.second_stage_cls_type, second_stage_cls_heatmap,
                second_stage_gt_mask,
                loss_pos_weight=self.second_stage_cls_loss_pos_weight,
                loss_neg_weight=self.second_stage_cls_loss_neg_weight)
            total_loss += second_stage_cls_loss
            loss["scalar"]["second_stage_cls_loss"] = (
                second_stage_cls_loss.detach().cpu())

            reg_loss = _compute_reg_loss(
                reg_biases, gt_biases, loss_weight=self.reg_loss_weight)
            total_loss += reg_loss
            loss["scalar"]["reg_loss"] = reg_loss.detach().cpu()

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
