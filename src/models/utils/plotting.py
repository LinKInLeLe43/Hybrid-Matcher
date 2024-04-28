from typing import Any, Dict, List, Optional

from matplotlib import figure
from matplotlib import lines
from matplotlib import pyplot as plt
import numpy as np


def plot_matching_figure(
    image0: np.ndarray,
    image1: np.ndarray,
    matching_points0: np.ndarray,
    matching_points1: np.ndarray,
    colors: np.ndarray,
    key_points0: Optional[np.ndarray] = None,
    key_points1: Optional[np.ndarray] = None,
    dpi: int = 75,
    text: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Optional[figure.Figure]:
    if not len(matching_points0) == len(matching_points1) == len(colors):
        raise ValueError("")
    if (key_points0 is None) == (key_points1 is not None):
        raise ValueError("")

    figure, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(image0, cmap="gray")
    axes[1].imshow(image1, cmap="gray")
    for i in range(2):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if key_points0 is not None:
        axes[0].scatter(key_points0[:, 0], key_points0[:, 1], s=0.5, c="w")
        axes[1].scatter(key_points1[:, 0], key_points1[:, 1], s=0.5, c="w")

    n = len(matching_points0)
    if n != 0:
        figure.canvas.draw()
        axes[0].scatter(
            matching_points0[:, 0], matching_points0[:, 1], s=4, c=colors)
        axes[1].scatter(
            matching_points1[:, 0], matching_points1[:, 1], s=4, c=colors)

        # inv_figure_trans = figure.transFigure.inverted()
        # figure_points0 = axes[0].transData.transform(matching_points0)
        # figure_points1 = axes[1].transData.transform(matching_points1)
        # figure_points0 = inv_figure_trans.transform(figure_points0)
        # figure_points1 = inv_figure_trans.transform(figure_points1)
        # for i in range(n):
        #     x_coors = [figure_points0[i, 0], figure_points1[i, 0]]
        #     y_coors = [figure_points0[i, 1], figure_points1[i, 1]]
        #     line = lines.Line2D(
        #         x_coors, y_coors, lw=1, c=colors[i],
        #         transform=figure.transFigure)
        #     figure.lines.append(line)

    if text is not None:
        color = "k" if image0[:100, :200].mean() > 180 else "w"
        text = "\n".join(text)
        figure.text(
            0.01, 0.99, text, size=15, c=color, va="top", ha="left",
            transform=figure.axes[0].transAxes)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        return figure


def _create_error_colors(
    errors: np.ndarray,
    threshold: float,
    alpha: float = 1.0
) -> np.ndarray:
    errors = (errors / (2 * threshold)).clip(min=0.0, max=1.0)
    reds = (2 * errors).clip(min=0.0, max=1.0)
    greens = (2 * (1 - errors)).clip(min=0.0, max=1.0)
    blues = np.zeros_like(errors)
    alphas = alpha * np.ones_like(errors)
    colors = np.stack([reds, greens, blues, alphas], axis=1)
    return colors


def plot_evaluation_figures(
    batch: Dict[str, Any],
    result: Dict[str, Any],
    error: Dict[str, Any],
    epipolar_threshold: float
) -> List[figure.Figure]:
    figures = []
    for b in range(len(batch["image0"])):
        image0 = (255 * batch["image0"][b, 0]).round().int().cpu().numpy()
        image1 = (255 * batch["image1"][b, 0]).round().int().cpu().numpy()

        mask = result["b_idxes"] == b
        points0 = result["points0"][mask].cpu().numpy()
        points1 = result["points1"][mask].cpu().numpy()
        # if batch.get("scale0") is not None:
        #     points0 /= batch["scale0"][b].cpu().numpy()
        #     points1 /= batch["scale1"][b].cpu().numpy()

        epipolar_errors = error["epipolar_errors_per_batch"][b]
        mask = epipolar_errors < epipolar_threshold
        total_count = len(mask)
        alpha = np.interp(
            total_count, [0, 300, 1000, 2000], [1.0, 0.8, 0.4, 0.2]).item()
        colors = _create_error_colors(
            epipolar_errors, epipolar_threshold, alpha=alpha)

        correct_count = mask.sum().item()
        precision = correct_count / total_count if total_count != 0 else 0.0
        R_error, t_error = error["R_errors"][b], error["t_errors"][b]
        text = [f"#Matches {total_count}",
                f"Precision({epipolar_threshold:.2e}) "
                f"({100 * precision:.1f}%): {correct_count}/{total_count}",
                f"ΔR: {R_error:.1f}°, Δt: {t_error:.1f}°"]

        figure = plot_matching_figure(
            image0, image1, points0, points1, colors, text=text)
        figures.append(figure)
    return figures
