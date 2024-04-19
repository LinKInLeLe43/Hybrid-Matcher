from typing import Any, Dict, Optional, Tuple

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


@torch.no_grad()
def create_cls_supervision(  # TODO: add scale for keys
    batch: Dict[str, Any],
    scale: int,
    use_offset: bool = False,
    return_with_gt_mask: bool = True,
    use_flow: bool = False
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

    point0 = K.create_meshgrid(
        h0, w0, normalized_coordinates=False, device=device)
    point1 = K.create_meshgrid(
        h1, w1, normalized_coordinates=False, device=device)
    if use_offset:
        point0 += 0.5
        point1 += 0.5
    point0 = scale0 * point0.reshape(1, -1, 2).repeat(n, 1, 1)
    point1 = scale1 * point1.reshape(1, -1, 2).repeat(n, 1, 1)
    if mask0 is not None:
        point0[~mask0], point1[~mask1] = 0.0, 0.0

    point0_to_1 = _warp_point(
        point0, batch["depth0"], batch["K0"], batch["K1"], batch["T0_to_1"])
    point1_to_0 = _warp_point(
        point1, batch["depth1"], batch["K1"], batch["K0"], batch["T1_to_0"])
    coor0_to_1, coor1_to_0 = point0_to_1 / scale1, point1_to_0 / scale0
    if use_offset:
        coor0_to_1 -= 0.5
        coor0_to_1 -= 0.5
    grid0_to_1 = coor0_to_1.round().long()
    grid1_to_0 = coor1_to_0.round().long()
    _mask_out_of_bound(grid0_to_1, h1, w1)
    _mask_out_of_bound(grid1_to_0, h0, w0)
    idx0_to_1 = w1 * grid0_to_1[:, :, 1] + grid0_to_1[:, :, 0]
    idx1_to_0 = w0 * grid1_to_0[:, :, 1] + grid1_to_0[:, :, 0]

    biprojection = torch.stack([idx1_to_0[b, idx1]
                                for b, idx1 in enumerate(idx0_to_1)])
    mask = biprojection == torch.arange(l0, device=device)
    mask[:, 0] = False
    b_idxes, i_idxes = mask.nonzero(as_tuple=True)
    j_idxes = idx0_to_1[b_idxes, i_idxes]
    if len(b_idxes) != 0:
        gt_idxes = b_idxes, i_idxes, j_idxes
    else:
        gt_idxes = 3 * (torch.tensor([0], device=device),)  # TODO: check whether need argument
    supervision = {"point0_to_1": point0_to_1,
                   "point1": point1,
                   "gt_idxes": gt_idxes}
    if return_with_gt_mask:
        gt_mask = torch.zeros((n, l0, l1), dtype=torch.bool, device=device)
        gt_mask[b_idxes, i_idxes, j_idxes] = True
        supervision["gt_mask"] = gt_mask
    if use_flow:
        supervision["gt_coor0"] = coor0_to_1[b_idxes, i_idxes]
        supervision["gt_coor1"] = coor1_to_0[b_idxes, j_idxes]
    return supervision


@torch.no_grad()
def create_fine_cls_supervision(
    batch: Dict[str, Any],
    scales: Tuple[int, int],
    fine_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Dict[str, Any]:
    device = batch["image0"].device
    n, _, h0, w0 = batch["image0"].shape
    _, _, h1, w1 = batch["image1"].shape
    ((coarse_h0, _), (coarse_w0, fine_w0),
     (coarse_h1, _), (coarse_w1, fine_w1)) = map(
        lambda x: (x // scales[0], x // scales[1]), (h0, w0, h1, w1))
    stride = scales[0] // scales[1]
    m, ww = len(fine_idxes[0]), stride ** 2

    coarse_mask0 = coarse_mask1 = None
    if "mask0" in batch:
        coarse_mask0, coarse_mask1 = batch.pop("mask0"), batch.pop("mask1")
        mask0 = coarse_mask0.reshape(1, n, coarse_h0, coarse_w0).float()
        batch["mask0"] = F.interpolate(
            mask0, scale_factor=stride).reshape(n, -1).bool()
        mask1 = coarse_mask1.reshape(1, n, coarse_h1, coarse_w1).float()
        batch["mask1"] = F.interpolate(
            mask1, scale_factor=stride).reshape(n, -1).bool()
    supervision = create_cls_supervision(
        batch, scales[1], use_offset=True, return_with_gt_mask=False)
    if "mask0" in batch:
        batch["mask0"], batch["mask1"] = coarse_mask0, coarse_mask1

    b_idxes, i_idxes, j_idxes = supervision.pop("gt_idxes")
    subi_idxes = (stride * (i_idxes // fine_w0 % stride) +
                  i_idxes % fine_w0 % stride)
    subj_idxes = (stride * (j_idxes // fine_w1 % stride) +
                  j_idxes % fine_w1 % stride)
    i_idxes = ((fine_w0 // stride) * (i_idxes // fine_w0 // stride) +
               i_idxes % fine_w0 // stride)
    j_idxes = ((fine_w1 // stride) * (j_idxes // fine_w1 // stride) +
               j_idxes % fine_w1 // stride)

    k0 = coarse_h0 * coarse_w0 * coarse_h1 * coarse_w1
    k1 = coarse_h1 * coarse_w1
    identifiers = k0 * fine_idxes[0] + k1 * fine_idxes[1] + fine_idxes[2]
    identifiers = {v: i for i, v in enumerate(identifiers)}
    m_idxes = k0 * b_idxes + k1 * i_idxes + j_idxes
    m_idxes = torch.tensor(
        [identifiers.get(v, -1) for v in m_idxes], device=device)
    valid_mask = m_idxes != -1
    gt_idxes = (m_idxes[valid_mask], subi_idxes[valid_mask],
                subj_idxes[valid_mask])
    gt_mask = torch.zeros((m, ww, ww), dtype=torch.bool, device=device)
    gt_mask[gt_idxes] = True
    supervision["fine_gt_mask"] = gt_mask
    return supervision


@torch.no_grad()
def create_fine_reg_supervision(
    point0_to_1: torch.Tensor,
    point1: torch.Tensor,
    fine_idxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fine_scale: int,
    window_size: int,
    scale1: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    b_idxes, i_idxes, j_idxes = fine_idxes

    gt_biases = point0_to_1[b_idxes, i_idxes] - point1[b_idxes, j_idxes]
    gt_biases /= fine_scale * (window_size // 2)
    if scale1 is not None:
        gt_biases /= scale1[b_idxes]
    supervision = {"fine_gt_biases": gt_biases}
    return supervision
