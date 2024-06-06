from typing import Any, Dict, Optional, Tuple

import einops
import kornia as K
import torch
from torch.nn import functional as F


def _warp_point(
    image_point0: torch.Tensor,
    depth0: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T0_to_1: torch.Tensor
) -> torch.Tensor:
    image_grid0 = image_point0.round().long()
    image_depth0 = torch.stack(
        [depth0[b, grid0[:, 1], grid0[:, 0]]
         for b, grid0 in enumerate(image_grid0)])[:, :, None]
    world_point = torch.cat([image_depth0 * image_point0, image_depth0], dim=2)
    camera_point0 = K0.inverse() @ world_point.transpose(1, 2)
    camera_point1 = T0_to_1[:, :3, :3] @ camera_point0 + T0_to_1[:, :3, 3:]
    image_point1 = (K1 @ camera_point1).transpose(1, 2)
    image_point1 = image_point1[:, :, :2] / (image_point1[:, :, 2:] + 1e-4)
    return image_point1


def _mask_out_of_bound(x: torch.Tensor, h: int, w: int) -> None:
    x[(x[:, :, 0] < 0) | (x[:, :, 0] >= w) |
      (x[:, :, 1] < 0) | (x[:, :, 1] >= h)] = 0.0


def _crop_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    ww = kernel_size ** 2
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(output, "n (c ww) l -> n l ww c", ww=ww)
    return output


@torch.no_grad()
def compute_gt_biases(
    points0_to_1: torch.Tensor,
    points1: torch.Tensor,
    idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fine_scale: int,
    window_size: int
) -> torch.Tensor:
    b_idxes, i_idxes, j_idxes = idxes

    gt_biases = points0_to_1[b_idxes, i_idxes] - points1[b_idxes, j_idxes]
    gt_biases /= fine_scale * (window_size // 2)
    return gt_biases


@torch.no_grad()
def create_first_stage_supervision(
    batch: Dict[str, Any],
    scale: int,
    use_offset: bool = False,  # TODO: whether need actually
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
    mask0, mask1 = batch.get("mask0"), batch.get("mask1")

    coors0 = K.create_meshgrid(
        h0, w0, normalized_coordinates=False, device=device)
    coors1 = K.create_meshgrid(
        h1, w1, normalized_coordinates=False, device=device)
    coors0 = coors0.reshape(1, -1, 2).repeat(n, 1, 1)
    coors1 = coors1.reshape(1, -1, 2).repeat(n, 1, 1)
    offset = 0.5 if use_offset else 0.0
    points0 = scale0 * (coors0 + offset)
    points1 = scale1 * (coors1 + offset)
    if mask0 is not None:
        points0[~mask0.flatten(start_dim=1)] = 0.0
        points1[~mask1.flatten(start_dim=1)] = 0.0
    points0_to_1 = _warp_point(
        points0, batch["depth0"], batch["K0"], batch["K1"], batch["T0_to_1"])
    points1_to_0 = _warp_point(
        points1, batch["depth1"], batch["K1"], batch["K0"], batch["T1_to_0"])
    flows0 = coors0_to_1 = (points0_to_1 / scale1) - offset
    flows1 = coors1_to_0 = (points1_to_0 / scale0) - offset

    coors0_to_1 = coors0_to_1.round().long()
    coors1_to_0 = coors1_to_0.round().long()
    _mask_out_of_bound(coors0_to_1, h1, w1)
    _mask_out_of_bound(coors1_to_0, h0, w0)
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
    supervision = {"first_stage_gt_idxes": gt_idxes,
                   "first_stage_gt_mask": gt_mask}

    if return_coor:
        if "scale1" in batch:
            points0_to_1 = points0_to_1 / batch["scale1"][:, None]
            points1 = points1 / batch["scale1"][:, None]
        supervision["points0_to_1"] = points0_to_1
        supervision["points1"] = points1

    if return_flow:
        supervision["gt_flows0"] = flows0[b_idxes, i_idxes]
        supervision["gt_flows1"] = flows1[b_idxes, j_idxes]
    return supervision


@torch.no_grad()
def create_second_stage_supervision(
    batch: Dict[str, Any],
    scales: Tuple[int, int],
    idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Dict[str, Any]:
    device = batch["image0"].device
    stride = scales[0] // scales[1]
    ww = stride ** 2
    b_idxes, i_idxes, j_idxes = idxes
    m = len(b_idxes)
    if m == 0:
        points0_to_1 = torch.empty((0, ww, 2), device=device)
        points1 = torch.empty((0, ww, 2), device=device)
        gt_mask = torch.empty((0, ww, ww), dtype=torch.bool, device=device)
        supervision = {"points0_to_1": points0_to_1,
                       "points1": points1,
                       "second_stage_gt_mask": gt_mask}
        return supervision

    n, _, h0, w0 = batch["image0"].shape
    _, _, h1, w1 = batch["image1"].shape
    h0, w0, h1, w1 = map(lambda x: x // scales[1], (h0, w0, h1, w1))
    scale0, scale1 = batch.get("scale0"), batch.get("scale1")
    scale0 = scales[1] * scale0[:, None] if scale0 is not None else scales[1]
    scale1 = scales[1] * scale1[:, None] if scale1 is not None else scales[1]

    coors0 = K.create_meshgrid(
        h0, w0, normalized_coordinates=False, device=device)
    coors1 = K.create_meshgrid(
        h1, w1, normalized_coordinates=False, device=device)
    coors0 = coors0.repeat(n, 1, 1, 1).permute(0, 3, 1, 2)
    coors1 = coors1.repeat(n, 1, 1, 1).permute(0, 3, 1, 2)
    coors0 = F.pad(coors0, [stride // 2, 0, stride // 2, 0])
    coors1 = F.pad(coors1, [stride // 2, 0, stride // 2, 0])
    coors0 = _crop_windows(coors0, stride, stride, 0)[b_idxes, i_idxes]
    coors1 = _crop_windows(coors1, stride, stride, 0)[b_idxes, j_idxes]
    idxes0 = w0 * coors0[:, :, 1] + coors0[:, :, 0]
    idxes1 = w1 * coors1[:, :, 1] + coors1[:, :, 0]
    coors0 = coors0 + 0.5
    coors1 = coors1 + 0.5
    points0 = scale0 * coors0
    points1 = scale1 * coors1
    points0_to_1 = _warp_point(
        points0.reshape(1, -1, 2), batch["depth0"], batch["K0"], batch["K1"], batch["T0_to_1"]).reshape(-1, ww, 2)
    points1_to_0 = _warp_point(
        points1.reshape(1, -1, 2), batch["depth1"], batch["K1"], batch["K0"], batch["T1_to_0"]).reshape(-1, ww, 2)
    flows0 = coors0_to_1 = points0_to_1 / scale1
    flows1 = coors1_to_0 = points1_to_0 / scale0

    coors0_to_1 = (coors0_to_1 - 0.5).round().long()
    coors1_to_0 = (coors1_to_0 - 0.5).round().long()
    _mask_out_of_bound(coors0_to_1, h1, w1)
    _mask_out_of_bound(coors1_to_0, h0, w0)
    idxes0_to_1 = w1 * coors0_to_1[:, :, 1] + coors0_to_1[:, :, 0]
    idxes1_to_0 = w0 * coors1_to_0[:, :, 1] + coors1_to_0[:, :, 0]
    gt_mask = ((idxes0_to_1[:, :, None] == idxes1[:, None, :]) &
               (idxes0[:, :, None] == idxes1_to_0[:, None, :]))
    if "scale1" in batch:
        points0_to_1 = points0_to_1 / batch["scale1"][:, None]
        points1 = points1 / batch["scale1"][:, None]
    supervision = {"points0_to_1": points0_to_1,
                   "points1": points1,
                   "second_stage_gt_mask": gt_mask}
    return supervision
