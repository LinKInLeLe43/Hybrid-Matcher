import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import kornia as K
import numpy as np
import poselib
import torch
import torch.nn.functional as F
from kornia.geometry import (
    cross_product_matrix,
    normalize_points_with_intrinsics,
    symmetrical_epipolar_distance,
)
from numpy import ndarray
from torch import Tensor


def _warp_point(
    points0: Tensor,
    depth0: Tensor,
    depth1: Tensor,
    K0: Tensor,
    K1: Tensor,
    T0_to_1: Tensor,
    use_bilinear: bool = False,
    return_mask: bool = False,
    consistent_depth_ratio: float = 0.2,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    n, h, w = depth0.shape

    image_2d_points0 = points0
    image_2d_norm_points0 = (
        2 * (image_2d_points0 + 0.5) / points0.new_tensor([w, h]) - 1
    )
    image_depth0 = F.grid_sample(
        depth0[:, None], image_2d_norm_points0[:, :, None], mode="nearest"
    )[:, 0]
    if use_bilinear:
        _depth0 = torch.where(depth0 > 0.0, depth0, float("nan"))
        _image_depth0 = F.grid_sample(
            _depth0[:, None], image_2d_norm_points0[:, :, None], mode="bilinear"
        )[:, 0]
        image_depth0 = torch.where(
            ~_image_depth0.isnan(), _image_depth0, image_depth0
        )
    image_3d_points0 = torch.cat(
        [image_depth0 * image_2d_points0, image_depth0], dim=2
    )

    camera_point0 = K0.inverse() @ image_3d_points0.transpose(1, 2)
    camera_point1 = T0_to_1[:, :3, :3] @ camera_point0 + T0_to_1[:, :3, 3:]

    image_3d_points1 = (K1 @ camera_point1).transpose(1, 2)
    image_2d_points1 = image_3d_points1[:, :, :2] / (
        image_3d_points1[:, :, 2:] + 1e-4
    )
    points1 = image_2d_points1

    if return_mask:
        image_2d_norm_points1 = (
            2 * (image_2d_points1 + 0.5) / points0.new_tensor([w, h]) - 1
        )
        image_depth1 = F.grid_sample(
            depth1[:, None], image_2d_norm_points1[:, :, None], mode="nearest"
        )[:, 0]
        if use_bilinear:
            _depth1 = torch.where(depth1 > 0.0, depth1, float("nan"))
            _image_depth1 = F.grid_sample(
                _depth1[:, None],
                image_2d_norm_points1[:, :, None],
                mode="bilinear",
            )[:, 0]
            image_depth1 = torch.where(
                ~_image_depth1.isnan(), _image_depth1, image_depth1
            )

        depth0_mask = image_depth0[:, :, 0] != 0.0
        depth1_mask = image_depth1[:, :, 0] != 0.0
        consistent_mask = (
            (image_depth1[:, :, 0] - camera_point1[:, 2, :])
            / image_depth1[:, :, 0]
        ).abs() < consistent_depth_ratio
        mask = depth0_mask & depth1_mask & consistent_mask
        return points1, mask
    else:
        return points1


def _compute_end_point_errors(
    b_idxes: Tensor,
    fine_points0: Tensor,
    fine_points1: Tensor,
    depth0: Tensor,
    depth1: Tensor,
    K0: Tensor,
    K1: Tensor,
    T0_to_1: Tensor,
    coarse_scale: int,
    scale1: Union[int, Tensor],
    coarse_w1: int,
    j_idxes: Tensor,
    inliers_per_batch: ndarray,
    consistent_depth_ratio: float = 0.2,
    coarse_points0: Optional[Tensor] = None,
) -> Tuple[
    ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
]:
    n, h, w = depth0.shape
    device = fine_points0.device

    end_point_errors_per_batch = np.empty(n, dtype=object)
    inlier_end_point_errors_per_batch = np.empty(n, dtype=object)
    true_coarse_counts = []
    inlier_true_coarse_counts = []
    coarse_precisions = []
    inlier_coarse_precisions = []
    coarse_3x3_precisions = []
    inlier_coarse_3x3_precisions = []
    for b in range(n):
        b_mask = b_idxes == b
        b_scale1 = scale1 if isinstance(scale1, int) else scale1[b]

        gt_fine_points1, mask = _warp_point(
            fine_points0[b_mask][None],
            depth0[[b]],
            depth1[[b]],
            K0[[b]],
            K1[[b]],
            T0_to_1[[b]],
            return_mask=True,
            consistent_depth_ratio=consistent_depth_ratio,
        )
        gt_fine_points1, mask = gt_fine_points1[0], mask[0]

        gt_coarse_points1 = gt_fine_points1
        if coarse_points0 is not None:
            gt_coarse_points1, mask = _warp_point(
                coarse_points0[b_mask][None],
                depth0[[b]],
                depth1[[b]],
                K0[[b]],
                K1[[b]],
                T0_to_1[[b]],
                return_mask=True,
                consistent_depth_ratio=consistent_depth_ratio,
            )
            gt_coarse_points1, mask = gt_coarse_points1[0], mask[0]

        end_point_errors = (
            (fine_points1[b_mask][mask] - gt_fine_points1[mask]) / b_scale1
        ).norm(dim=1)
        end_point_errors_per_batch[b] = end_point_errors.cpu().numpy()

        gt_coarse_grid1 = gt_coarse_points1[mask] / coarse_scale / b_scale1
        gt_coarse_3x3_grid1 = gt_coarse_grid1[
            :, None
        ].round().long() + K.create_meshgrid(3, 3, device=device).reshape(-1, 2)
        gt_j_3x3_idxes = (
            coarse_w1 * gt_coarse_3x3_grid1[:, :, 1]
            + gt_coarse_3x3_grid1[:, :, 0]
        )
        coarse_3x3_results = (
            j_idxes[b_mask][mask, None] == gt_j_3x3_idxes
        ).float()
        true_coarse_count = 0.0
        coarse_precision = 0.0
        coarse_3x3_precision = 0.0
        if len(coarse_3x3_results) != 0:
            true_coarse_count = coarse_3x3_results[:, 3 * 3 // 2].sum().item()
            coarse_precision = coarse_3x3_results[:, 3 * 3 // 2].mean().item()
            coarse_3x3_precision = coarse_3x3_results.sum(dim=1).mean().item()
        true_coarse_counts.append(true_coarse_count)
        coarse_precisions.append(coarse_precision)
        coarse_3x3_precisions.append(coarse_3x3_precision)

        inlier_end_point_errors = end_point_errors.new_tensor([])
        inlier_true_coarse_count = 0.0
        inlier_coarse_precision = 0.0
        inlier_coarse_3x3_precision = 0.0
        if len(inliers_per_batch[b]) != 0:
            inlier = torch.from_numpy(inliers_per_batch[b]).to(device)[mask]
            if inlier.any():
                inlier_end_point_errors = end_point_errors[inlier]
                inlier_true_coarse_count = (
                    coarse_3x3_results[inlier, 3 * 3 // 2].sum().item()
                )
                inlier_coarse_precision = (
                    coarse_3x3_results[inlier, 3 * 3 // 2].mean().item()
                )
                inlier_coarse_3x3_precision = (
                    coarse_3x3_results[inlier].sum(dim=1).mean().item()
                )
        inlier_end_point_errors_per_batch[b] = (
            inlier_end_point_errors.cpu().numpy()
        )
        inlier_true_coarse_counts.append(inlier_true_coarse_count)
        inlier_coarse_precisions.append(inlier_coarse_precision)
        inlier_coarse_3x3_precisions.append(inlier_coarse_3x3_precision)
    true_coarse_counts = np.array(true_coarse_counts)
    inlier_true_coarse_counts = np.array(inlier_true_coarse_counts)
    coarse_precisions = np.array(coarse_precisions)
    inlier_coarse_precisions = np.array(inlier_coarse_precisions)
    coarse_3x3_precisions = np.array(coarse_3x3_precisions)
    inlier_coarse_3x3_precisions = np.array(inlier_coarse_3x3_precisions)
    return (
        end_point_errors_per_batch,
        inlier_end_point_errors_per_batch,
        true_coarse_counts,
        inlier_true_coarse_counts,
        coarse_precisions,
        inlier_coarse_precisions,
        coarse_3x3_precisions,
        inlier_coarse_3x3_precisions,
    )


def compute_sym_epi_errors(
    b_indices: Tensor,
    points0: Tensor,
    points1: Tensor,
    K0: Tensor,
    K1: Tensor,
    T0_to_1: Tensor,
) -> ndarray:
    E = cross_product_matrix(T0_to_1[:, :3, 3]) @ T0_to_1[:, :3, :3]
    errors_per_batch = np.empty(len(K0), dtype=object)
    for b in range(len(K0)):
        mask = b_indices == b
        b_points0 = normalize_points_with_intrinsics(points0[mask], K0[b])
        b_points1 = normalize_points_with_intrinsics(points1[mask], K1[b])
        b_errors = (
            symmetrical_epipolar_distance(
                b_points0[None], b_points1[None], E[[b]]
            )[0]
            .cpu()
            .numpy()
        )
        errors_per_batch[b] = b_errors
    return errors_per_batch


def _estimate_rel_pose_opencv(
    b_points0: Tensor,
    b_points1: Tensor,
    b_K0: Tensor,
    b_K1: Tensor,
    method: str = "RANSAC",
    confidence: float = 0.99999,
    threshold: float = 0.5,
) -> Optional[Tuple[ndarray, ndarray, ndarray]]:
    if len(b_points0) < 5:
        return None

    b_points0 = normalize_points_with_intrinsics(b_points0, b_K0).cpu().numpy()
    b_points1 = normalize_points_with_intrinsics(b_points1, b_K1).cpu().numpy()
    scale = (
        Tensor([b_K0[0, 0], b_K1[1, 1], b_K0[0, 0], b_K1[1, 1]]).mean().item()
    )
    threshold /= scale
    method = getattr(cv2, method)
    E, mask = cv2.findEssentialMat(
        b_points0,
        b_points1,
        np.eye(3),
        method=method,
        prob=confidence,
        threshold=threshold,
    )
    if E is None:
        return None

    best_num_inliers = 0
    out = None
    for E_ in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            E_, b_points0, b_points1, np.eye(3), 1e9, mask=mask
        )
        if n > best_num_inliers:
            best_num_inliers = n
            out = R, t[:, 0], mask.ravel() != 0
    return out


def _estimate_rel_pose_poselib(
    b_points0: Tensor,
    b_points1: Tensor,
    b_K0: Tensor,
    b_K1: Tensor,
    threshold: float = 2.0,
) -> Optional[Tuple[ndarray, ndarray, ndarray]]:
    b_points0, b_points1 = b_points0.cpu().numpy(), b_points1.cpu().numpy()
    camera0 = {
        "model": "PINHOLE",
        "width": int(b_K0[0, 2] * 2),
        "height": int(b_K0[1, 2] * 2),
        "params": [b_K0[0, 0], b_K0[1, 1], b_K0[0, 2], b_K0[1, 2]],
    }
    camera1 = {
        "model": "PINHOLE",
        "width": int(b_K1[0, 2] * 2),
        "height": int(b_K1[1, 2] * 2),
        "params": [b_K1[0, 0], b_K1[1, 1], b_K1[0, 2], b_K1[1, 2]],
    }
    M, info = poselib.estimate_relative_pose(
        b_points0,
        b_points1,
        camera0,
        camera1,
        {"max_epipolar_error": threshold},
    )
    out = None
    if M is not None:
        out = M.R, M.t, np.array(info["inliers"])
    return out


def _compute_pose_error(
    est_R: ndarray, est_t: ndarray, gt_R: ndarray, gt_t: ndarray
) -> Tuple[float, float]:
    cos_R = ((est_R.T @ gt_R).trace() - 1.0) / 2.0
    cos_R = cos_R.clip(min=-1.0, max=1.0)
    R_error = np.rad2deg(np.arccos(cos_R)).item()

    cos_t = (est_t @ gt_t) / (np.linalg.norm(est_t) * np.linalg.norm(gt_t))
    cos_t = cos_t.clip(min=-1.0, max=1.0)
    t_error = np.rad2deg(np.arccos(cos_t))
    t_error = np.minimum(t_error, 180.0 - t_error).item()
    return R_error, t_error


def compute_rel_pose_errors(
    b_indices: Tensor,
    points0: Tensor,
    points1: Tensor,
    K0: Tensor,
    K1: Tensor,
    T0_to_1: Tensor,
    estimator: str = "opencv",
    num_repeats: int = 1,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[ndarray, ndarray, ndarray]:
    estimator = getattr(
        sys.modules[__name__], "_estimate_rel_pose_" + estimator
    )
    params = params if params is not None else {}

    inliers_per_batch = np.empty(len(K0), dtype=object)
    rel_R_errors, rel_t_errors = [], []
    for b in range(len(K0)):
        mask = b_indices == b
        inliers = np.array([], dtype=bool)
        rel_R_error, rel_t_error = np.inf, np.inf

        b_points0, b_points1 = points0[mask], points1[mask]
        b_K0, b_K1 = K0[b], K1[b]

        for _ in range(num_repeats):
            shuffling = torch.from_numpy(
                np.random.permutation(np.arange(len(b_points0)))
            )
            b_points0, b_points1 = b_points0[shuffling], b_points1[shuffling]
            out = estimator(b_points0, b_points1, b_K0, b_K1, **params)
            if out is not None:
                est_rel_R, est_rel_t, inliers = out
                gt_rel_R = T0_to_1[b, :3, :3].cpu().numpy()
                gt_rel_t = T0_to_1[b, :3, 3].cpu().numpy()
                rel_R_error, rel_t_error = _compute_pose_error(
                    est_rel_R, est_rel_t, gt_rel_R, gt_rel_t
                )
            inliers_per_batch[b] = inliers
            rel_R_errors.append(rel_R_error), rel_t_errors.append(rel_t_error)
    rel_R_errors, rel_t_errors = np.array(rel_R_errors), np.array(rel_t_errors)
    return rel_R_errors, rel_t_errors, inliers_per_batch


def compute_error(
    batch: Dict[str, Any],
    result: Dict[str, Any],
    rel_pose_ransac_config: Dict[str, Any],
    advanced: bool = False,
    coarse_scale: Optional[int] = None,
) -> Dict[str, Any]:
    identifiers = [
        "#".join(paths) for paths in zip(batch["name0"], batch["name1"])
    ]
    identifiers = np.array(identifiers, dtype=object)
    sym_epi_errors_per_batch = compute_sym_epi_errors(
        result["coarse_cls_indices"][0],
        result["points0"],
        result["points1"],
        batch["K0"],
        batch["K1"],
        batch["T0_to_1"],
    )
    rel_R_errors, rel_t_errors, inliers_per_batch = compute_rel_pose_errors(
        result["coarse_cls_indices"][0],
        result["points0"],
        result["points1"],
        batch["K0"],
        batch["K1"],
        batch["T0_to_1"],
        **rel_pose_ransac_config,
    )
    error = {
        "identifiers": identifiers,
        "sym_epi_errors_per_batch": sym_epi_errors_per_batch,
        "rel_R_errors": rel_R_errors,
        "rel_t_errors": rel_t_errors,
        "inliers_per_batch": inliers_per_batch,
    }
    if advanced:
        if coarse_scale is None:
            raise ValueError("")
        scale1 = batch["scale1"][:, None] if "scale1" in batch else 1
        coarse_w1 = batch["image1"].shape[3] // coarse_scale
        (
            error["end_point_errors_per_batch"],
            error["inlier_end_point_errors_per_batch"],
            error["true_coarse_counts"],
            error["inlier_true_coarse_counts"],
            error["coarse_precisions"],
            error["inlier_coarse_precisions"],
            error["coarse_3x3_precisions"],
            error["inlier_coarse_3x3_precisions"],
        ) = _compute_end_point_errors(
            result["coarse_cls_indices"][0],
            result["points0"],
            result["points1"],
            batch["depth0"],
            batch["depth1"],
            batch["K0"],
            batch["K1"],
            batch["T0_to_1"],
            coarse_scale,
            scale1,
            coarse_w1,
            result["coarse_cls_indices"][2],
            inliers_per_batch,
            coarse_points0=result.get("coarse_points0"),
        )
    return error


def _calc_precision(
    errors_per_batch: ndarray, thresholds: List[float]
) -> List[float]:
    precisions_per_threshold = []
    for threshold in thresholds:
        precisions = []
        for errors in errors_per_batch:
            mask = errors < threshold
            precisions.append(mask.mean() if len(mask) != 0 else 0.0)
        precision = np.mean(precisions).item()
        precisions_per_threshold.append(precision)
    return precisions_per_threshold


def _calc_auc(errors: ndarray, thresholds: List[float]) -> List[float]:
    errors = np.sort(np.append(errors, 0))
    recalls = np.linspace(0, 1, num=len(errors))
    aucs = []
    for threshold in thresholds:
        index = np.searchsorted(errors, threshold)
        y = np.append(recalls[:index], recalls[index - 1])
        x = np.append(errors[:index], threshold)
        auc = (np.trapz(y, x=x) / threshold).item()
        aucs.append(auc)
    return aucs


def compute_metric(
    error: Dict[str, Any],
    metric_thresholds: Dict[str, Any],
    advanced: bool = False,
) -> Dict[str, Any]:
    unique_indices = np.unique(error["identifiers"], return_index=True)[1]
    metric = {}

    errors = error["sym_epi_errors_per_batch"][unique_indices]
    thresholds = metric_thresholds["sym_epi_prec"]
    values = _calc_precision(errors, thresholds)
    for threshold, value in zip(thresholds, values):
        metric[f"sym_epi_prec@{threshold}"] = value

    errors = np.maximum(error["rel_R_errors"], error["rel_t_errors"])
    errors = errors.reshape(len(error["identifiers"]), -1)[
        unique_indices
    ].reshape(-1)
    thresholds = metric_thresholds["rel_pose_auc"]
    values = _calc_auc(errors, thresholds)
    for threshold, value in zip(thresholds, values):
        metric[f"rel_pose_auc@{threshold}"] = value

    if advanced:
        errors = error["end_point_errors_per_batch"][unique_indices]
        thresholds = metric_thresholds["end_point_prec"]
        values = _calc_precision(errors, thresholds)
        for threshold, value in zip(thresholds, values):
            metric[f"end_point_prec@{threshold}"] = value

        errors = error["inlier_end_point_errors_per_batch"][unique_indices]
        values = _calc_precision(errors, thresholds)
        for threshold, value in zip(thresholds, values):
            metric[f"end_point_inlier_prec@{threshold}"] = value

        metric["coarse_prec@1"] = error["coarse_precisions"][
            unique_indices
        ].mean()
        metric["coarse_inlier_prec@1"] = error["inlier_coarse_precisions"][
            unique_indices
        ].mean()
        metric["coarse_prec@9"] = error["coarse_3x3_precisions"][
            unique_indices
        ].mean()
        metric["coarse_inlier_prec@9"] = error["inlier_coarse_3x3_precisions"][
            unique_indices
        ].mean()
        metric["num_true_coarse"] = error["true_coarse_counts"][
            unique_indices
        ].mean()
        metric["num_true_coarse_inlier"] = error["inlier_true_coarse_counts"][
            unique_indices
        ].mean()
    return metric
