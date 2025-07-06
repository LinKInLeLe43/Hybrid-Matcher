from typing import Any, Dict, Optional, Tuple

import kornia as K
import torch
from torch import nn
from torch.nn import functional as F

from src.models.utils.metrics import _warp_point as _warp_point1


def _mask_out_of_bound(x: torch.Tensor, h: int, w: int) -> None:
    x[(x[:, :, 0] < 0) | (x[:, :, 0] >= w) |
      (x[:, :, 1] < 0) | (x[:, :, 1] >= h)] = 0


def _warp_point(
    image_point0: torch.Tensor,
    depth0: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T0_to_1: torch.Tensor
) -> torch.Tensor:
    _, h, w = depth0.shape

    image_grid0 = image_point0.round().long()
    _mask_out_of_bound(image_grid0, h, w)
    image_depth0 = torch.stack(
        [depth0[b, grid0[:, 1], grid0[:, 0]]
         for b, grid0 in enumerate(image_grid0)])[:, :, None]
    world_point = torch.cat([image_depth0 * image_point0, image_depth0], dim=2)
    camera_point0 = K0.inverse() @ world_point.transpose(1, 2)
    camera_point1 = T0_to_1[:, :3, :3] @ camera_point0 + T0_to_1[:, :3, 3:]
    image_point1 = (K1 @ camera_point1).transpose(1, 2)
    image_point1 = image_point1[:, :, :2] / (image_point1[:, :, 2:] + 1e-4)
    return image_point1


@torch.no_grad()
def create_coarse_supervision(
    batch: Dict[str, Any],
    scale: int,
    extra_scale: Optional[int] = None,
    extra_extra_scale: Optional[int] = None,
    return_coor: bool = False,
    return_flow: bool = False
) -> Dict[str, Any]:
    device = batch["image0"].device
    n, _, h0, w0 = batch["image0"].shape
    _, _, h1, w1 = batch["image1"].shape
    h0, w0, h1, w1 = map(lambda x: x // scale, (h0, w0, h1, w1))
    l0, l1 = h0 * w0, h1 * w1
    scale0, scale1 = batch.get("scale0"), batch.get("scale1")
    scale0 = scale * scale0[:, None] if scale0 is not None else scale
    scale1 = scale * scale1[:, None] if scale1 is not None else scale
    mask0, mask1 = batch.get(f"mask0_{scale}x"), batch.get(f"mask1_{scale}x")

    coors0 = K.create_meshgrid(
        h0, w0, normalized_coordinates=False, device=device)
    coors1 = K.create_meshgrid(
        h1, w1, normalized_coordinates=False, device=device)
    coors0 = coors0.reshape(1, -1, 2).repeat(n, 1, 1)
    coors1 = coors1.reshape(1, -1, 2).repeat(n, 1, 1)
    points0 = scale0 * coors0
    points1 = scale1 * coors1
    if mask0 is not None:
        points0[~mask0.flatten(start_dim=1)] = 0.0
        points1[~mask1.flatten(start_dim=1)] = 0.0

    points0_to_1 = _warp_point(
        points0, batch["depth0"], batch["K0"], batch["K1"], batch["T0_to_1"])
    points1_to_0 = _warp_point(
        points1, batch["depth1"], batch["K1"], batch["K0"], batch["T1_to_0"])
    flows0 = coors0_to_1 = points0_to_1 / scale1
    flows1 = coors1_to_0 = points1_to_0 / scale0

    coors0_to_1 = coors0_to_1.round().long()
    coors1_to_0 = coors1_to_0.round().long()
    _h0, _w0, _h1, _w1 = h0, w0, h1, w1
    if mask0 is not None:
        _h0 = mask0.sum(dim=1).amax(dim=1).int()[:, None]
        _w0 = mask0.sum(dim=2).amax(dim=1).int()[:, None]
        _h1 = mask1.sum(dim=1).amax(dim=1).int()[:, None]
        _w1 = mask1.sum(dim=2).amax(dim=1).int()[:, None]
    _mask_out_of_bound(coors0_to_1, _h1, _w1)
    _mask_out_of_bound(coors1_to_0, _h0, _w0)
    idxes0_to_1 = w1 * coors0_to_1[:, :, 1] + coors0_to_1[:, :, 0]
    idxes1_to_0 = w0 * coors1_to_0[:, :, 1] + coors1_to_0[:, :, 0]
    biprojection = torch.stack([idxes1_to_0[b, idx1]
                                for b, idx1 in enumerate(idxes0_to_1)])
    biprojection_mask = biprojection == torch.arange(l0, device=device)
    biprojection_mask[:, 0] = False
    b_idxes, i_idxes = biprojection_mask.nonzero(as_tuple=True)
    j_idxes = idxes0_to_1[b_idxes, i_idxes]
    gt_idxes = ((b_idxes, i_idxes, j_idxes) if len(b_idxes) != 0
                else 3 * (torch.tensor([0], device=device),))
    gt_mask = torch.zeros((n, l0, l1), dtype=torch.bool, device=device)
    gt_mask[b_idxes, i_idxes, j_idxes] = True
    supervision = {"coarse_gt_idxes": gt_idxes, "coarse_gt_mask": gt_mask}

    if extra_scale is not None:
        if extra_scale <= scale:
            raise ValueError("")

        stride = extra_scale // scale
        fh0, fw0, fh1, fw1 = map(lambda x: x // stride, (h0, w0, h1, w1))
        _gt_mask = gt_mask.reshape(
            -1, fh0, stride, fw0, stride, fh1, stride, fw1, stride
        )
        _gt_mask = _gt_mask.sum(dim=(2, 4, 6, 8)).bool()
        _gt_mask = _gt_mask.reshape(-1, fh0 * fw0, fh1 * fw1)
        _gt_idxes = _gt_mask.nonzero(as_tuple=True)
        supervision["extra_coarse_gt_mask"] = _gt_mask
        supervision["extra_coarse_gt_idxes"] = _gt_idxes

    if extra_extra_scale is not None:
        if extra_extra_scale <= scale:
            raise ValueError("")

        stride = extra_extra_scale // scale
        fh0, fw0, fh1, fw1 = map(lambda x: x // stride, (h0, w0, h1, w1))
        _gt_mask = gt_mask.reshape(
            -1, fh0, stride, fw0, stride, fh1, stride, fw1, stride
        )
        _gt_mask = _gt_mask.sum(dim=(2, 4, 6, 8)).bool()
        _gt_mask = _gt_mask.reshape(-1, fh0 * fw0, fh1 * fw1)
        _gt_idxes = _gt_mask.nonzero(as_tuple=True)
        supervision["extra_extra_coarse_gt_mask"] = _gt_mask
        supervision["extra_extra_coarse_gt_idxes"] = _gt_idxes

    if return_coor:
        if "scale0" in batch:
            points1_to_0 = points1_to_0 / batch["scale0"][:, None]
            points0 = points0 / batch["scale0"][:, None]
        if "scale1" in batch:
            points0_to_1 = points0_to_1 / batch["scale1"][:, None]
            points1 = points1 / batch["scale1"][:, None]
        supervision["gt_points1_to_0"] = points1_to_0
        supervision["gt_points0"] = points0
        supervision["gt_points0_to_1"] = points0_to_1
        supervision["gt_points1"] = points1

    if return_flow:
        supervision["gt_flows0"] = flows0[gt_idxes[0], gt_idxes[1]]
        supervision["gt_flows1"] = flows1[gt_idxes[0], gt_idxes[2]]
    return supervision


@torch.no_grad()
def create_fine_supervision(
    batch: Dict[str, Any],
    scales: Tuple[int, int],
    idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    offset: float = 0.5,
    return_coor: bool = False
) -> Dict[str, Any]:
    m, w, scale = len(idxes[0]), scales[0] // scales[1], scales[1]
    ww, x, device = w ** 2, batch["image0"], batch["image0"].device
    b_idxes, i_idxes, j_idxes = idxes

    if m == 0:
        supervision = {
            "fine_gt_mask": x.new_empty((0, ww, ww), dtype=torch.bool)}

        if return_coor:
            supervision["gt_points0_to_1"] = x.new_empty((0, ww, 2))
            supervision["gt_points1"] = x.new_empty((0, ww, 2))
        return supervision

    n, _, h0, w0 = batch["image0"].shape
    _, _, h1, w1 = batch["image1"].shape
    h0, w0, h1, w1 = map(lambda x: x // scale, (h0, w0, h1, w1))
    scale0, scale1 = batch.get("scale0"), batch.get("scale1")
    scale0 = scale * scale0[b_idxes, None] if scale0 is not None else scale
    scale1 = scale * scale1[b_idxes, None] if scale1 is not None else scale

    coors0 = K.create_meshgrid(
        h0, w0, normalized_coordinates=False, device=device)
    coors1 = K.create_meshgrid(
        h1, w1, normalized_coordinates=False, device=device)
    coors0 = coors0.repeat(n, 1, 1, 1).permute(0, 3, 1, 2)
    coors1 = coors1.repeat(n, 1, 1, 1).permute(0, 3, 1, 2)
    coors0 = F.pad(coors0, (w // 2, 0, w // 2, 0))
    coors1 = F.pad(coors1, (w // 2, 0, w // 2, 0))
    coors0 = F.unfold(coors0, w, stride=w)[b_idxes, :, i_idxes]
    coors1 = F.unfold(coors1, w, stride=w)[b_idxes, :, j_idxes]
    coors0 = coors0.unflatten(1, (2, ww)).transpose(1, 2)
    coors1 = coors1.unflatten(1, (2, ww)).transpose(1, 2)
    idxes0 = w0 * coors0[:, :, 1] + coors0[:, :, 0]
    idxes1 = w1 * coors1[:, :, 1] + coors1[:, :, 0]
    idxes0 = torch.where(idxes0 == 0, -100, idxes0 + h0 * w0 * b_idxes[:, None])
    idxes1 = torch.where(idxes1 == 0, -100, idxes1 + h1 * w1 * b_idxes[:, None])
    points0 = scale0 * (coors0 + offset)
    points1 = scale1 * (coors1 + offset)

    points0_to_1 = torch.zeros_like(points0)
    points1_to_0 = torch.zeros_like(points1)
    for b in range(n):
        b_mask = b_idxes == b
        b_points0 = points0[b_mask].reshape(1, -1, 2)
        b_points1 = points1[b_mask].reshape(1, -1, 2)
        b_points0_to_1 = _warp_point(
            b_points0, batch["depth0"][[b]], batch["K0"][[b]], batch["K1"][[b]],
            batch["T0_to_1"][[b]])
        b_points1_to_0 = _warp_point(
            b_points1, batch["depth1"][[b]], batch["K1"][[b]], batch["K0"][[b]],
            batch["T1_to_0"][[b]])
        points0_to_1[b_mask] = b_points0_to_1.reshape(-1, ww, 2)
        points1_to_0[b_mask] = b_points1_to_0.reshape(-1, ww, 2)
    coors0_to_1 = points0_to_1 / scale1
    coors1_to_0 = points1_to_0 / scale0

    coors0_to_1 = (coors0_to_1 - offset).round().long()
    coors1_to_0 = (coors1_to_0 - offset).round().long()
    _mask_out_of_bound(coors0_to_1, h1, w1)
    _mask_out_of_bound(coors1_to_0, h0, w0)
    idxes0_to_1 = w1 * coors0_to_1[:, :, 1] + coors0_to_1[:, :, 0]
    idxes1_to_0 = w0 * coors1_to_0[:, :, 1] + coors1_to_0[:, :, 0]
    idxes0_to_1 = torch.where(
        idxes0_to_1 == 0, -200, idxes0_to_1 + h1 * w1 * b_idxes[:, None])
    idxes1_to_0 = torch.where(
        idxes1_to_0 == 0, -200, idxes1_to_0 + h0 * w0 * b_idxes[:, None])
    gt_mask = ((idxes0_to_1[:, :, None] == idxes1[:, None, :]) &
               (idxes0[:, :, None] == idxes1_to_0[:, None, :]))
    supervision = {"fine_gt_mask": gt_mask}

    if return_coor:
        if "scale1" in batch:
            points0_to_1 = points0_to_1 / batch["scale1"][b_idxes, None]
            points1 = points1 / batch["scale1"][b_idxes, None]
        supervision["gt_points0_to_1"] = points0_to_1
        supervision["gt_points1"] = points1
    return supervision


@torch.no_grad()
def compute_reg_gt_biases(
    points0_to_1: torch.Tensor,
    points1_to_0: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    b_idxes, i_idxes, j_idxes = idxes

    gt_biases1 = points0_to_1[b_idxes, i_idxes] - points1[b_idxes, j_idxes]
    gt_biases1 /= 8.0
    gt_biases0 = points1_to_0[b_idxes, j_idxes] - points0[b_idxes, i_idxes]
    gt_biases0 /= 8.0
    gt_biases = torch.cat([gt_biases1, gt_biases0])
    return gt_biases


def compute_dense_gt_biases(
    batch: Dict[str, Any],
    result: Dict[str, Any],
    dense_matcher: nn.Module,
    points1: torch.Tensor,
    idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fine_scale: int,
    window_size: int
) -> Dict[str, Any]:
    b_idxes, i_idxes, j_idxes = idxes

    dense_num = 2 * 1 + 1
    data = {"flow_predictions": result.pop("flow_predictions"),
            "mkpts0_c": result["coarse_points0"],
            "mkpts1_c": result["coarse_points1"],
            "b_ids": b_idxes,
            "i_ids": i_idxes,
            "j_ids": j_idxes}
    if "scale0" in batch:
        data["scale0"] = batch["scale0"]
        data["scale1"] = batch["scale1"]
    dense_matcher.get_fine_match_dense(
        data, dense_num=dense_num, update_fmatch=False)
    mkpts0_f_dense = data["mkpts0_f_dense"]
    mkpts1_f_dense = data["mkpts1_f_dense"]

    n = len(batch["image0"])
    dense_b_idxes = b_idxes.repeat_interleave(dense_num ** 2, dim=0)
    points0_to_1 = torch.zeros_like(mkpts0_f_dense)
    valid_mask0_to_1 = torch.zeros_like(mkpts0_f_dense[:, 0], dtype=torch.bool)
    for b in range(n):
        b_mask = dense_b_idxes == b
        b_points0_to_1, b_valid_mask0_to_1 = _warp_point1(
            mkpts0_f_dense[b_mask][None], batch["depth0"][[b]],
            batch["depth1"][[b]], batch["K0"][[b]], batch["K1"][[b]],
            batch["T0_to_1"][[b]], use_bilinear=True, return_mask=True,
            consistent_depth_ratio=0.05)
        points0_to_1[b_mask] = b_points0_to_1[0]
        valid_mask0_to_1[b_mask] = b_valid_mask0_to_1[0]

    if "scale1" in batch:
        mkpts1_f_dense = mkpts1_f_dense / batch["scale1"][dense_b_idxes]
        points0_to_1 = points0_to_1 / batch["scale1"][dense_b_idxes]

    points1 = points1[b_idxes, j_idxes].repeat_interleave(dense_num ** 2, dim=0)
    reg_biases = (mkpts1_f_dense - points1) / (fine_scale * (window_size // 2))
    gt_biases = (points0_to_1 - points1) / (fine_scale * (window_size // 2))

    supervision = {"dense_reg_biases": reg_biases,
                   "dense_gt_biases": gt_biases,
                   "dense_valid_mask": valid_mask0_to_1}
    return supervision
