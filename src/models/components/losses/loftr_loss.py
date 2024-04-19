from typing import Any, Dict, Optional

import torch
from torch import nn


def _compute_focal_loss(  # TODO: support NLL
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
    if len(heatmap) == 0:
        return loss_pos_weight * heatmap.new_ones(())

    weight = None
    if mask0 is not None:
        weight = (mask0[:, :, None] & mask1[:, None, :]).float()
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
                bin0_mask &= mask0
                bin1_mask &= mask1
            bins = torch.cat([heatmap[:, :-1, -1][bin0_mask],
                              heatmap[:, -1, :-1][bin1_mask]])
            use_bin = True
        heatmap = heatmap[:, :-1, :-1]

    pos_losses = _compute_focal_loss(heatmap[pos_mask])
    if weight is not None:
        pos_losses *= weight[pos_mask]
    if sparse and not use_bin:
        loss = loss_pos_weight * pos_losses.mean()
        return loss

    if use_bin:
        neg_losses = _compute_focal_loss(bins)
    else:
        neg_losses = _compute_focal_loss(1 - heatmap[neg_mask])
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
    if len(reg_biases) == 0:
        return loss_weight * reg_biases.new_ones(())

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
    flow: torch.Tensor,
    gt_coors: torch.Tensor,
    loss_weight: float = 1.0
) -> torch.Tensor:
    if len(flow) == 0:
        return loss_weight * flow.new_ones(())

    flow_coors, flow_log_stds_mul_2 = flow.chunk(2, dim=1)
    l2_distances = (flow_coors - gt_coors) ** 2
    losses = flow_log_stds_mul_2 + (-flow_log_stds_mul_2).exp() * l2_distances
    loss = loss_weight * losses.mean()
    return loss


class LoFTRLoss(nn.Module):  # TODO: change name
    def __init__(
        self,
        coarse_type: str,
        fine_type: str,
        coarse_cls_sparse: Optional[bool] = None,
        coarse_cls_loss_pos_weight: Optional[float] = None,
        coarse_cls_loss_neg_weight: Optional[float] = None,
        fine_cls_sparse: Optional[bool] = None,
        fine_cls_loss_pos_weight: Optional[float] = None,
        fine_cls_loss_neg_weight: Optional[float] = None,
        fine_reg_loss_weight: Optional[float] = None,
        use_flow: bool = False,
        flow_loss_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.coarse_type = coarse_type
        self.fine_type = fine_type
        self.coarse_cls_sparse = None
        self.coarse_cls_loss_pos_weight = None
        self.coarse_cls_loss_neg_weight = None
        self.fine_cls_sparse = None
        self.fine_cls_loss_pos_weight = None
        self.fine_cls_loss_neg_weight = None
        self.fine_reg_loss_weight = None
        self.use_flow = use_flow
        self.flow_loss_weight = None

        if coarse_type == "dual_softmax":  # TODO: cls_only
            if (coarse_cls_sparse is None or
                coarse_cls_loss_pos_weight is None or
                coarse_cls_loss_neg_weight is None):
                raise ValueError("")
            self.coarse_cls_sparse = coarse_cls_sparse
            self.coarse_cls_loss_pos_weight = coarse_cls_loss_pos_weight
            self.coarse_cls_loss_neg_weight = coarse_cls_loss_neg_weight
        elif coarse_type == "optimal_transport":  # TODO: cls_only
            if (coarse_cls_sparse is None or
                coarse_cls_loss_pos_weight is None or
                coarse_cls_loss_neg_weight is None):
                raise ValueError("")
            self.coarse_cls_sparse = coarse_cls_sparse
            self.coarse_cls_loss_pos_weight = coarse_cls_loss_pos_weight
            self.coarse_cls_loss_neg_weight = coarse_cls_loss_neg_weight
        else:
            raise ValueError("")
        if fine_type == "reg_only":
            if fine_reg_loss_weight is None:
                raise ValueError("")
            self.fine_reg_loss_weight = fine_reg_loss_weight
        elif self.fine_type == "reg_only_with_std":
            if fine_reg_loss_weight is None:
                raise ValueError("")
            self.fine_reg_loss_weight = fine_reg_loss_weight
        elif self.fine_type == "cls_and_reg":
            if (fine_cls_sparse is None or fine_cls_loss_pos_weight is None or
                fine_cls_loss_neg_weight is None or
                fine_reg_loss_weight is None):
                raise ValueError("")
            self.fine_cls_sparse = fine_cls_sparse
            self.fine_cls_loss_pos_weight = fine_cls_loss_pos_weight
            self.fine_cls_loss_neg_weight = fine_cls_loss_neg_weight
            self.fine_reg_loss_weight = fine_reg_loss_weight
        else:
            raise ValueError("")
        if use_flow:
            if flow_loss_weight is None:
                raise ValueError("")
            self.flow_loss_weight = flow_loss_weight

    def _compute_coarse_loss(
        self,
        cls_heatmap: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        result = {}
        if self.coarse_type == "dual_softmax":  # TODO: cls_only
            if cls_heatmap is None or gt_mask is None:
                raise ValueError("")
            cls_loss = _compute_cls_loss(
                self.coarse_cls_sparse, cls_heatmap, gt_mask,
                loss_pos_weight=self.coarse_cls_loss_pos_weight,
                loss_neg_weight=self.coarse_cls_loss_neg_weight, mask0=mask0,
                mask1=mask1)
            result["coarse_cls_loss"] = cls_loss
        elif self.coarse_type == "optimal_transport":  # TODO: cls_only
            if cls_heatmap is None or gt_mask is None:
                raise ValueError("")
            cls_loss = _compute_cls_loss(
                self.coarse_cls_sparse, cls_heatmap, gt_mask,
                loss_pos_weight=self.coarse_cls_loss_pos_weight,
                loss_neg_weight=self.coarse_cls_loss_neg_weight, mask0=mask0,
                mask1=mask1)
            result["coarse_cls_loss"] = cls_loss
        else:
            assert False
        return result

    def _compute_fine_loss(
        self,
        cls_heatmap: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        reg_biases: Optional[torch.Tensor] = None,
        gt_biases: Optional[torch.Tensor] = None,
        reg_stds: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        result = {}
        if self.fine_type == "reg_only":
            if reg_biases is None or gt_biases is None:
                raise ValueError("")
            reg_loss = _compute_reg_loss(
                reg_biases, gt_biases, loss_weight=self.fine_reg_loss_weight)
            result["fine_reg_loss"] = reg_loss
        elif self.fine_type == "reg_only_with_std":
            if reg_biases is None or gt_biases is None or reg_stds is None:
                raise ValueError("")
            reg_loss = _compute_reg_loss(
                reg_biases, gt_biases, reg_stds=reg_stds,
                loss_weight=self.fine_reg_loss_weight)
            result["fine_reg_loss"] = reg_loss
        elif self.fine_type == "cls_and_reg":
            if (cls_heatmap is None or gt_mask is None or reg_biases is None or
                gt_biases is None):
                raise ValueError()
            cls_loss = _compute_cls_loss(
                self.fine_cls_sparse, cls_heatmap, gt_mask,
                loss_pos_weight=self.fine_cls_loss_pos_weight,
                loss_neg_weight=self.fine_cls_loss_neg_weight)
            result["fine_cls_loss"] = cls_loss
            reg_loss = _compute_reg_loss(
                reg_biases, gt_biases, loss_weight=self.fine_reg_loss_weight)
            result["fine_reg_loss"] = reg_loss
        else:
            assert False
        return result

    def _compute_flow_loss(
        self,
        flow0: Optional[torch.Tensor] = None,
        flow1: Optional[torch.Tensor] = None,
        gt_coor0: Optional[torch.Tensor] = None,
        gt_coor1: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        result = {}
        if self.use_flow:
            if (flow0 is None or flow1 is None or gt_coor0 is None or
                gt_coor1 is None):
                raise ValueError("")
            loss0 = _compute_flow_loss(
                flow0, gt_coor0, loss_weight=self.flow_loss_weight)
            loss1 = _compute_flow_loss(
                flow1, gt_coor1, loss_weight=self.flow_loss_weight)
            loss = (loss0 + loss1) / 2
            result["flow_loss"] = loss
        return result

    def forward(
        self,
        coarse_cls_heatmap: Optional[torch.Tensor] = None,
        coarse_gt_mask: Optional[torch.Tensor] = None,
        fine_cls_heatmap: Optional[torch.Tensor] = None,
        fine_gt_mask: Optional[torch.Tensor] = None,
        fine_reg_biases: Optional[torch.Tensor] = None,
        fine_gt_biases: Optional[torch.Tensor] = None,
        fine_reg_stds: Optional[torch.Tensor] = None,
        flow0: Optional[torch.Tensor] = None,
        flow1: Optional[torch.Tensor] = None,
        gt_coor0: Optional[torch.Tensor] = None,
        gt_coor1: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        total_loss = 0.0
        loss = {"scalar": {}}

        coarse_loss = self._compute_coarse_loss(
            cls_heatmap=coarse_cls_heatmap, gt_mask=coarse_gt_mask, mask0=mask0,
            mask1=mask1)
        for k, v in coarse_loss.items():
            total_loss += v
            loss["scalar"][k] = v.detach().cpu()

        fine_loss = self._compute_fine_loss(
            cls_heatmap=fine_cls_heatmap, gt_mask=fine_gt_mask,
            reg_biases=fine_reg_biases, gt_biases=fine_gt_biases,
            reg_stds=fine_reg_stds)
        for k, v in fine_loss.items():
            total_loss += v
            loss["scalar"][k] = v.detach().cpu()

        flow_loss = self._compute_flow_loss(
            flow0=flow0, flow1=flow1, gt_coor0=gt_coor0, gt_coor1=gt_coor1)
        for k, v in flow_loss.items():
            total_loss += v
            loss["scalar"][k] = v.detach().cpu()

        loss["loss"] = total_loss
        loss["scalar"]["total_loss"] = total_loss.detach().cpu()
        return loss
