from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import kornia as K
from kornia import geometry
import numpy as np
from numpy import linalg
import torch
from torch.nn import functional as F


def _warp_point(
    points0: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T0_to_1: torch.Tensor,
    use_bilinear: bool = False,
    return_mask: bool = False,
    consistent_depth_ratio: float = 0.2
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    n, h, w = depth0.shape

    image_2d_points0 = points0
    image_2d_norm_points0 = (2 * (image_2d_points0 + 0.5) /
                             points0.new_tensor([w, h]) - 1)
    image_depth0 = F.grid_sample(
        depth0[:, None], image_2d_norm_points0[:, :, None],
        mode="nearest")[:, 0]
    if use_bilinear:
        _depth0 = torch.where(depth0 > 0.0, depth0, float("nan"))
        _image_depth0 = F.grid_sample(
            _depth0[:, None], image_2d_norm_points0[:, :, None],
            mode="bilinear")[:, 0]
        image_depth0 = torch.where(
            ~_image_depth0.isnan(), _image_depth0, image_depth0)
    image_3d_points0 = torch.cat([image_depth0 * image_2d_points0,
                                  image_depth0], dim=2)

    camera_point0 = K0.inverse() @ image_3d_points0.transpose(1, 2)
    camera_point1 = T0_to_1[:, :3, :3] @ camera_point0 + T0_to_1[:, :3, 3:]

    image_3d_points1 = (K1 @ camera_point1).transpose(1, 2)
    image_2d_points1 = (image_3d_points1[:, :, :2] /
                        (image_3d_points1[:, :, 2:] + 1e-4))
    points1 = image_2d_points1

    if return_mask:
        image_2d_norm_points1 = (2 * (image_2d_points1 + 0.5) /
                                 points0.new_tensor([w, h]) - 1)
        image_depth1 = F.grid_sample(
            depth1[:, None], image_2d_norm_points1[:, :, None],
            mode="nearest")[:, 0]
        if use_bilinear:
            _depth1 = torch.where(depth1 > 0.0, depth1, float("nan"))
            _image_depth1 = F.grid_sample(
                _depth1[:, None], image_2d_norm_points1[:, :, None],
                mode="bilinear")[:, 0]
            image_depth1 = torch.where(
                ~_image_depth1.isnan(), _image_depth1, image_depth1)

        depth0_mask = image_depth0[:, :, 0] != 0.0
        depth1_mask = image_depth1[:, :, 0] != 0.0
        consistent_mask = ((image_depth1[:, :, 0] - camera_point1[:, 2, :]) /
                           image_depth1[:, :, 0]).abs() < consistent_depth_ratio
        mask = depth0_mask & depth1_mask & consistent_mask
        return points1, mask
    else:
        return points1


def _compute_end_point_errors(
    b_idxes: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T0_to_1: torch.Tensor,
    coarse_scale: int,
    scale1: Union[int, torch.Tensor],
    coarse_w1: int,
    j_idxes: torch.Tensor,
    inliers_per_batch: np.array,
    consistent_depth_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, h, w = depth0.shape
    device = points0.device

    end_point_errors_per_batch = np.empty(n, dtype=object)
    coarse_precisions = []
    coarse_3x3_precisions = []
    inlier_end_point_errors_per_batch = np.empty(n, dtype=object)
    inlier_coarse_precisions = []
    inlier_coarse_3x3_precisions = []
    for b in range(n):
        b_mask = b_idxes == b
        b_scale1 = scale1 if isinstance(scale1, int) else scale1[b]
        b_coarse_scale1 = coarse_scale * b_scale1
        gt_points1, mask = _warp_point(
            points0[b_mask][None], depth0[[b]], depth1[[b]], K0[[b]], K1[[b]],
            T0_to_1[[b]], return_mask=True,
            consistent_depth_ratio=consistent_depth_ratio)
        gt_points1, mask = gt_points1[0], mask[0]

        end_point_errors = ((points1[b_mask][mask] - gt_points1[mask]) /
                            b_scale1).norm(dim=1)
        end_point_errors_per_batch[b] = end_point_errors.cpu().numpy()

        gt_coarse_3x3_grid1 = (
            (gt_points1[mask] / b_coarse_scale1)[:, None].round().long() +
            K.create_meshgrid(3, 3, device=device).reshape(-1, 2))
        gt_j_3x3_idxes = (coarse_w1 * gt_coarse_3x3_grid1[:, :, 1] +
                          gt_coarse_3x3_grid1[:, :, 0])
        coarse_3x3_results = (j_idxes[b_mask][mask, None] ==
                              gt_j_3x3_idxes).float()
        coarse_precision = 0.0
        coarse_3x3_precision = 0.0
        if len(coarse_3x3_results) != 0:
            coarse_precision = coarse_3x3_results[:, 3 * 3 // 2].mean().item()
            coarse_3x3_precision = coarse_3x3_results.sum(dim=1).mean().item()
        coarse_precisions.append(coarse_precision)
        coarse_3x3_precisions.append(coarse_3x3_precision)

        inlier_end_point_errors = end_point_errors.new_tensor([])
        inlier_coarse_precision = 0.0
        inlier_coarse_3x3_precision = 0.0
        if len(inliers_per_batch[b]) != 0:
            inlier = torch.from_numpy(inliers_per_batch[b]).to(device)[mask]
            if inlier.any():
                inlier_end_point_errors = end_point_errors[inlier]
                inlier_coarse_precision = (
                    coarse_3x3_results[inlier, 3 * 3 // 2].mean().item())
                inlier_coarse_3x3_precision = (
                    coarse_3x3_results[inlier].sum(dim=1).mean().item())
        inlier_end_point_errors_per_batch[b] = (
            inlier_end_point_errors.cpu().numpy())
        inlier_coarse_precisions.append(inlier_coarse_precision)
        inlier_coarse_3x3_precisions.append(inlier_coarse_3x3_precision)
    coarse_precisions = np.array(coarse_precisions)
    inlier_coarse_precisions = np.array(inlier_coarse_precisions)
    coarse_3x3_precisions = np.array(coarse_3x3_precisions)
    inlier_coarse_3x3_precisions = np.array(inlier_coarse_3x3_precisions)
    return (end_point_errors_per_batch, coarse_precisions,
            coarse_3x3_precisions, inlier_end_point_errors_per_batch,
            inlier_coarse_precisions, inlier_coarse_3x3_precisions)



def _compute_epipolar_errors(
    b_idxes: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor
) -> np.ndarray:
    n = len(K0)
    E = geometry.cross_product_matrix(t) @ R
    errors_per_batch = np.empty(n, dtype=object)
    for b in range(n):
        mask = b_idxes == b
        b_points0 = geometry.normalize_points_with_intrinsics(
            points0[mask], K0[b])
        b_points1 = geometry.normalize_points_with_intrinsics(
            points1[mask], K1[b])
        errors = geometry.symmetrical_epipolar_distance(
            b_points0, b_points1, E[[b]])[0].cpu().numpy()
        errors_per_batch[b] = errors
    return errors_per_batch


def _estimate_pose(
    points0: torch.Tensor,
    points1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    prob: float = 0.99999,
    threshold: float = 0.5
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(points0) < 5:
        return None

    points0 = geometry.normalize_points_with_intrinsics(
        points0, K0).cpu().numpy()
    points1 = geometry.normalize_points_with_intrinsics(
        points1, K1).cpu().numpy()
    scale = torch.tensor([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]]).mean().item()
    threshold /= scale
    E, mask = cv2.findEssentialMat(
        points0, points1, np.eye(3), method=cv2.RANSAC, prob=prob,
        threshold=threshold)
    if E is None:
        return None

    best_inliers_count = 0
    out = None
    for E_ in np.split(E, len(E) // 3):
        n, R, t, _ = cv2.recoverPose(
            E_, points0, points1, np.eye(3), 1e9, mask=mask)
        if n > best_inliers_count:
            out = R, t[:, 0], mask.ravel() != 0
            best_inliers_count = n
    return out


def _compute_relative_pose_error(
    est_R: np.ndarray,
    gt_R: np.ndarray,
    est_t: np.ndarray,
    gt_t: np.ndarray
) -> Tuple[np.float64, np.float64]:
    cos_R_error = (((est_R.T @ gt_R).trace() - 1) / 2).clip(min=-1.0, max=1.0)
    R_error = np.rad2deg(np.arccos(cos_R_error))

    norm = linalg.norm(est_t) * linalg.norm(gt_t)
    cos_t_error = (est_t @ gt_t / norm).clip(min=-1.0, max=1.0)
    t_error = np.rad2deg(np.arccos(cos_t_error))
    t_error = np.minimum(t_error, 180 - t_error)
    return R_error, t_error


def _compute_pose_errors(
    b_idxes: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    prob: float = 0.99999,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(K0)
    R_errors, t_errors, inliers_per_batch = [], [], np.empty(n, dtype=object)
    for b in range(n):
        mask = b_idxes == b
        R_error, t_error, inliers = np.inf, np.inf, np.array([], dtype=np.bool_)
        out = _estimate_pose(
            points0[mask], points1[mask], K0[b], K1[b], prob=prob,
            threshold=threshold)
        if out is not None:
            est_R, est_t, inliers = out
            gt_R, gt_t = R[b].cpu().numpy(), t[b].cpu().numpy()
            R_error, t_error = _compute_relative_pose_error(
                est_R, gt_R, est_t, gt_t)
        R_errors.append(R_error)
        t_errors.append(t_error)
        inliers_per_batch[b] = inliers
    R_errors, t_errors = np.array(R_errors), np.array(t_errors)
    return R_errors, t_errors, inliers_per_batch


def compute_error(
    batch: Dict[str, Any],
    result: Dict[str, Any],
    coarse_scale: int,
) -> Dict[str, Any]:
    identifiers = ["#".join(paths)
                   for paths in zip(batch["name0"], batch["name1"])]
    identifiers = np.array(identifiers, dtype=object)
    epipolar_errors_per_batch = _compute_epipolar_errors(
        result["idxes"][0], result["points0"], result["points1"],
        batch["K0"], batch["K1"], batch["T0_to_1"][:, :3, :3],
        batch["T0_to_1"][:, :3, 3])
    R_errors, t_errors, inliers_per_batch = _compute_pose_errors(
        result["idxes"][0], result["points0"], result["points1"],
        batch["K0"], batch["K1"], batch["T0_to_1"][:, :3, :3],
        batch["T0_to_1"][:, :3, 3])
    error = {"identifiers": identifiers,
             "epipolar_errors_per_batch": epipolar_errors_per_batch,
             "R_errors": R_errors,
             "t_errors": t_errors,
             "inliers_per_batch": inliers_per_batch}
    if True:
        if coarse_scale is None:
            raise ValueError("")
        scale1 = batch["scale1"][:, None] if "scale1" in batch else 1
        coarse_w1 = batch["image1"].shape[3] // coarse_scale
        (error["end_point_errors_per_batch"],
         error["coarse_precisions"],
         error["coarse_3x3_precisions"],
         error["inlier_end_point_errors_per_batch"],
         error["inlier_coarse_precisions"],
         error["inlier_coarse_3x3_precisions"]
         ) = _compute_end_point_errors(
            result["idxes"][0], result["points0"], result["points1"],
            batch["depth0"], batch["depth1"], batch["K0"], batch["K1"],
            batch["T0_to_1"], coarse_scale, scale1, coarse_w1,
            result["idxes"][2], inliers_per_batch)
    return error


def _compute_precision(
    errors_per_batch: np.ndarray,
    thresholds: List[float]
) -> np.ndarray:
    precisions_per_threshold = []
    for threshold in thresholds:
        precisions = []
        for b in range(len(errors_per_batch)):
            mask = errors_per_batch[b] < threshold
            precisions.append(mask.mean() if len(mask) != 0 else 0.0)
        precisions_per_threshold.append(np.mean(precisions))
    precisions_per_threshold = np.array(precisions_per_threshold)
    return precisions_per_threshold


def _compute_aucs(errors: np.ndarray, thresholds: List[int]) -> np.ndarray:
    errors = np.sort(np.append(errors, 0))
    recalls = np.linspace(0, 1, num=len(errors))
    aucs = []
    for threshold in thresholds:
        idx = np.searchsorted(errors, threshold)
        y = np.append(recalls[:idx], recalls[idx - 1])
        x = np.append(errors[:idx] / threshold, 1)
        aucs.append(np.trapz(y, x=x))
    aucs = np.array(aucs)
    return aucs


def compute_metric(
    error: Dict[str, Any],
    epipolar_threshold: float,
    pose_thresholds: List[int]
) -> Dict[str, Any]:
    idxes = np.unique(error["identifiers"], return_index=True)[1]
    epipolar_precision = _compute_precision(
        error["epipolar_errors_per_batch"][idxes], [epipolar_threshold])
    pose_aucs = _compute_aucs(
        np.maximum(error["R_errors"][idxes], error["t_errors"][idxes]),
        pose_thresholds)
    metric = {"epipolar_precision": epipolar_precision,
              "pose_aucs": pose_aucs}
    if True:
        metric["end_point_precisions"] = _compute_precision(
            error["end_point_errors_per_batch"][idxes], [1.0, 3.0, 5.0])
        metric["inlier_end_point_precisions"] = _compute_precision(
            error["inlier_end_point_errors_per_batch"][idxes],
            [1.0, 3.0, 5.0])
        metric["coarse_precision"] = error["coarse_precisions"][idxes].mean()
        metric["coarse_3x3_precision"] = (
            error["coarse_3x3_precisions"][idxes].mean())
        metric["inlier_coarse_precision"] = (
            error["inlier_coarse_precisions"][idxes].mean())
        metric["inlier_coarse_3x3_precision"] = (
            error["inlier_coarse_3x3_precisions"][idxes].mean())
    return metric
