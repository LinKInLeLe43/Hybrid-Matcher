import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy import ndarray
from PIL import Image
from torch import Tensor, dtype
from torch.nn import Module


def make_matching_figure(
    image0: ndarray,
    image1: ndarray,
    points0: ndarray,
    points1: ndarray,
    colors: ndarray,
    text: Optional[List[str]] = None,
    dpi: int = 75,
    save_path: Optional[str] = None,
    pad_inches: float = 0.0,
) -> Optional[Figure]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    for ax, image in zip(axes, [image0, image1]):
        ax.imshow(image)
        ax.axis("off")

    if (
        points0.shape[0] != 0
        and points1.shape[0] != 0
        and points0.shape == points1.shape
    ):
        fig.canvas.draw()
        inv_fig_trans = fig.transFigure.inverted()
        fig_points0 = inv_fig_trans.transform(
            axes[0].transData.transform(points0)
        )
        fig_points1 = inv_fig_trans.transform(
            axes[1].transData.transform(points1)
        )
        fig.lines = [
            Line2D(
                (fig_points0[i, 0], fig_points1[i, 0]),
                (fig_points0[i, 1], fig_points1[i, 1]),
                linewidth=2,
                color=colors[i],
                transform=fig.transFigure,
            )
            for i in range(len(fig_points0))
        ]

        axes[0].autoscale(enable=False)
        axes[1].autoscale(enable=False)

        axes[0].scatter(points0[:, 0], points0[:, 1], 4, colors[:, :3])
        axes[1].scatter(points1[:, 0], points1[:, 1], 4, colors[:, :3])

    if text is not None:
        fig.text(
            0.01,
            0.99,
            "\n".join(text),
            color="k" if image0[:100, :200].mean() > 200 else "w",
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=15,
            transform=axes[0].transAxes,
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=pad_inches)
        plt.close()
    else:
        return fig


def gen_error_colormap(
    errors: ndarray, threshold: float, alpha: float = 1.0
) -> ndarray:
    assert alpha <= 1.0 and alpha > 0
    x = 1 - np.array(errors / (threshold * 2)).clip(min=0.0, max=1.0)
    colors = np.stack(
        [2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], axis=-1
    ).clip(min=0.0, max=1.0)
    return colors


def draw_matches(
    image0: ndarray,
    image1: ndarray,
    points0: ndarray,
    points1: ndarray,
    confidences: ndarray,
    text: Optional[List[str]] = None,
    dpi: int = 75,
    save_path: Optional[str] = None,
    pad_inches: float = 0.0,
) -> Optional[Figure]:
    threshold = 0.5
    colors = gen_error_colormap(1 - confidences, threshold, alpha=0.1)
    fig = make_matching_figure(
        image0,
        image1,
        points0,
        points1,
        colors,
        text=text or [f"#Matches: {len(points0)}"],
        save_path=save_path,
        dpi=dpi,
        pad_inches=pad_inches,
    )
    return fig


def load_image(
    path: str,
    type: str = "gray",
    size: Optional[int] = None,
    factor: int = 1,
    pad_to_square: bool = False,
    dtype: dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if type == "gray":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif type == "rgb":
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("")

    h, w = image.shape[:2]
    k = size / max(w, h) if size is not None else 1.0
    new_w, new_h = round(k * w), round(k * h)
    new_w, new_h = int(new_w // factor * factor), int(new_h // factor * factor)
    image = cv2.resize(image, (new_w, new_h))
    if type == "gray":
        image = image[:, :, None]
    scale = np.array([w / new_w, h / new_h])

    mask = None
    if pad_to_square:
        l = max(new_w, new_h)
        padded_image = np.zeros((l, l, image.shape[-1]), dtype=image.dtype)
        padded_image[:new_h, :new_w] = image
        image = padded_image
        mask = np.zeros((l, l), dtype=bool)
        mask[:new_h, :new_w] = True

    image = torch.from_numpy(image).to(dtype=dtype, device=device)
    image = image.permute(2, 0, 1) / 255.0
    scale = torch.from_numpy(scale).to(dtype=dtype, device=device)
    if mask is not None:
        mask = torch.from_numpy(mask).to(device=device)
    return image, scale, mask


def get_matcher(
    ckpt_path: str,
    size: int = 1024,
    threshold: float = 0.3,
    border_removal: int = 2,
    device: str = "cpu",
) -> Module:
    with hydra.initialize(config_path="./configs", version_base="1.3"):
        overrides = [
            "+experiment=casp_megadepth",
            f"++model.net.rope.test_size=[{max(832, size)},{max(832, size)}]",
            f"++model.net.coarse_matching.threshold={threshold}",
            f"++model.net.coarse_matching.border_removal={border_removal}",
        ]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        model = hydra.utils.instantiate(cfg.model).eval().to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")["state_dict"]
        )
        matcher = model.net
    return matcher


def main(
    matcher: Module, data_configs: Dict[str, Any], path0: str, path1: str
) -> Dict[str, Any]:
    image0, scale0, mask0 = load_image(path0, **data_configs)
    image1, scale1, mask1 = load_image(path1, **data_configs)
    batch = {
        "image0": image0[None],
        "image1": image1[None],
        "scale0": scale0[None],
        "scale1": scale1[None],
    }
    if data_configs.get("pad_to_square") is True:
        mask = torch.stack([mask0[None], mask1[None]]).float()
        for factor in [8, 16, 32]:
            batch[f"mask0_{factor}x"], batch[f"mask1_{factor}x"] = (
                F.max_pool2d(mask, factor, stride=factor).bool()
            )
    with torch.no_grad():
        result = matcher(batch)
    return result


if __name__ == "__main__":
    ransac_type = "affine"
    path0 = ""
    path1 = ""
    save_path = "casp.png"

    common_configs = {"size": 1152, "device": "cpu"}
    data_configs = {
        "type": "gray",
        "factor": 32,
        "pad_to_square": True,
        **common_configs,
    }
    model_config = {
        "threshold": 0.2,
        "border_removal": 0,
        "ckpt_path": "weights/minima.ckpt",
        **common_configs,
    }
    matcher = get_matcher(**model_config)
    result = main(matcher, data_configs, path0, path1)

    # root = "/Users/linkinlele43/Downloads/data_SAR_opt_dataset"
    # save_dir = "/".join([root, "minima"])
    # os.makedirs(save_dir, exist_ok=True)
    # with open("/".join([save_dir, "error.txt"]), "w", buffering=1) as f:
    #     for i in range(1, 10006 + 1):
    #         path0 = "/".join([root, "R", f"R{i}.jpg"])
    #         path1 = "/".join([root, "L", f"L{i}.jpg"])
    #         result = main(matcher, data_configs, path0, path1)

    #         points0 = result["points0"].cpu().numpy()
    #         points1 = result["points1"].cpu().numpy()

    #         M, inliers = cv2.estimateAffine2D(
    #             points1,
    #             points0,
    #             method=cv2.USAC_MAGSAC,
    #             ransacReprojThreshold=3.0,
    #             confidence=0.99999,
    #             maxIters=10000,
    #         )
    #         H = np.vstack([M, [0.0, 0.0, 1.0]])
    #         H_gt = np.loadtxt("/".join([root, "Res_txt", f"GT{i}.txt"]))
    #         height, width = 512, 512
    #         corners = np.array(
    #             [
    #                 [0, 0, 1],
    #                 [0, height - 1, 1],
    #                 [width - 1, 0, 1],
    #                 [width - 1, height - 1, 1],
    #             ]
    #         )
    #         gt_warped_corners = corners @ H_gt.transpose()
    #         gt_warped_corners = (
    #             gt_warped_corners[:, :2] / gt_warped_corners[:, 2:]
    #         )
    #         pred_warped_corners = corners @ H.transpose()
    #         pred_warped_corners = (
    #             pred_warped_corners[:, :2] / pred_warped_corners[:, 2:]
    #         )
    #         error = np.linalg.norm(
    #             gt_warped_corners - pred_warped_corners, axis=1
    #         ).mean()
    #         f.write(f"{error}\n")

    # root = "/Users/linkinlele43/Downloads/MatchData-lidarmap-3band/lasvegas"
    # save_dir = "/".join([root, "minima"])
    # os.makedirs(save_dir, exist_ok=True)
    # with open(f"{root}/test_gt_lasvegas.txt", "r") as f:
    #     for row in f.read().splitlines():
    #         name, _, _ = row.split()
    #         path0 = "/".join([root, "img", name + "_cropped_img.tif"])
    #         path1 = "/".join([root, "lidarmap_8bit", name + "_cropped_pc.tif"])
    #         result = main(matcher, data_configs, path0, path1)

    #         points0 = result["points0"].cpu().numpy()
    #         points1 = result["points1"].cpu().numpy()
    #         M, inliers = cv2.findHomography(
    #             points0,
    #             points1,
    #             method=cv2.USAC_MAGSAC,
    #             ransacReprojThreshold=3.0,
    #             confidence=0.99999,
    #             maxIters=10000,
    #         )
    #         H = np.vstack([M, [0.0, 0.0, 1.0]])

    #         image0 = cv2.imread(path0, cv2.IMREAD_COLOR)
    #         image1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    #         image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    #         warp = cv2.warpPerspective(image0, H, image1.shape[:2])
    #         _warp, _image1 = Image.fromarray(warp), Image.fromarray(image1)
    #         _warp.save(
    #             "/".join([save_dir, name + "_minima_warp.gif"]),
    #             save_all=True,
    #             append_images=[_image1],
    #             duration=500,
    #             loop=0,
    #         )
    #         np.savetxt(
    #             "/".join([save_dir, name + "_minima_homo.txt"]), H, fmt="%.8f", delimiter=" "
    #         )

    image0 = cv2.imread(path0, cv2.IMREAD_COLOR)
    image1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    points0 = result["points0"].cpu().numpy()
    points1 = result["points1"].cpu().numpy()
    confidences = result["scores"].cpu().numpy()
    fig = draw_matches(
        image0,
        image1,
        points0,
        points1,
        confidences,
        dpi=300,
        save_path=save_path,
        pad_inches=0.1,
    )

    if ransac_type == "affine":
        M, inliers = cv2.estimateAffine2D(
            points0,
            points1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99999,
            maxIters=10000,
        )
        H = np.vstack([M, [0.0, 0.0, 1.0]])
        mask = inliers.ravel() == 1
        fig = draw_matches(
            image0,
            image1,
            points0[mask],
            points1[mask],
            confidences[mask],
            dpi=300,
            save_path=save_path,
            pad_inches=0.1,
        )
        warp = cv2.warpPerspective(image0, H, image1.shape[:2])
        _warp, _image1 = Image.fromarray(warp), Image.fromarray(image1)
        _warp.save(
            "warp_minima.gif",
            save_all=True,
            append_images=[_image1],
            duration=500,
            loop=0,
        )
