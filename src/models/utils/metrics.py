from typing import Any, Dict, List, Optional, Tuple

import cv2
from kornia import geometry
import numpy as np
from numpy import linalg
import torch


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
    result: Dict[str, Any]
) -> Dict[str, Any]:
    identifiers = ["#".join(paths)
                   for paths in zip(batch["name0"], batch["name1"])]
    identifiers = np.array(identifiers, dtype=object)
    epipolar_errors_per_batch = _compute_epipolar_errors(
        result["b_idxes"], result["points0"], result["points1"],
        batch["K0"], batch["K1"], batch["T0_to_1"][:, :3, :3],
        batch["T0_to_1"][:, :3, 3])
    R_errors, t_errors, inliers_per_batch = _compute_pose_errors(
        result["b_idxes"], result["points0"], result["points1"],
        batch["K0"], batch["K1"], batch["T0_to_1"][:, :3, :3],
        batch["T0_to_1"][:, :3, 3])
    error = {"identifiers": identifiers,
             "epipolar_errors_per_batch": epipolar_errors_per_batch,
             "R_errors": R_errors,
             "t_errors": t_errors,
             "inliers_per_batch": inliers_per_batch}
    return error


def _compute_precision(
    errors_per_batch: np.ndarray,
    threshold: float
) -> np.ndarray:
    precisions = []
    for b in range(len(errors_per_batch)):
        mask = errors_per_batch[b] < threshold
        precisions.append(mask.mean() if len(mask) != 0 else 0.0)
    precision = np.mean(precisions)
    return precision


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
        error["epipolar_errors_per_batch"][idxes], epipolar_threshold)
    pose_aucs = _compute_aucs(
        np.maximum(error["R_errors"][idxes], error["t_errors"][idxes]),
        pose_thresholds)
    metric = {"epipolar_precision": epipolar_precision,
              "pose_aucs": pose_aucs}
    return metric
