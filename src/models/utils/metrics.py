import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import poselib
import torch
from kornia.geometry import (
    cross_product_matrix,
    normalize_points_with_intrinsics,
    symmetrical_epipolar_distance,
)
from numpy import ndarray
from torch import Tensor


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
    return metric
