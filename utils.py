from typing import List

import cv2
import hydra
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


def get_outdoor_hybrid_matcher(overrides: List[str], size: int, ckpt_path: str) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = overrides + [
            "experiment=new_matcher_megadepth/eval_one_stage",
            f"data.test_dataset.image_size={size}"]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model).eval()
        net.load_state_dict(torch.load(ckpt_path)["state_dict"])
    return net


def resize_im(wo, ho, imsize=None, dfactor=1, value_to_scale=max, enforce=False):
    if not isinstance(imsize, int) and len(imsize) == 2:
        # Resize to a fixed shape
        wt, ht = imsize
        scale = [wo / wt, ho / ht]
        return wt, ht, scale
    wt, ht = wo, ho

    # Resize only if the image is too big
    resize = imsize and value_to_scale(wo, ho) > imsize and imsize > 0
    if resize or enforce:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = [wo / wt, ho / ht]
    return wt, ht, scale


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]
    ), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[1], : inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def load_gray_scale_tensor_cv(
    im_path, device, imsize=None, value_to_scale=min, dfactor=1, pad2sqr=False
):
    """Image loading function applicable for LoFTR & Aspanformer."""

    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    ho, wo = im.shape
    wt, ht, scale = resize_im(
        wo,
        ho,
        imsize=imsize,
        dfactor=dfactor,
        value_to_scale=value_to_scale,
        enforce=pad2sqr,
    )
    im = cv2.resize(im, (wt, ht))
    mask = None
    if pad2sqr and (wt != ht):
        # Padding to square image
        im, mask = pad_bottom_right(im, max(wt, ht), ret_mask=True)
        mask = torch.from_numpy(mask).to(device)
    im = transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    return im, scale, mask


if __name__ == "__main__":
    path0 = ""
    path1 = ""
    size = 1152
    ckpt_path = 
    threshold = 0.2
    device = "cuda:0"

    overrides = [f"++model.net.coarse_matching.threshold={threshold}"]
    net = get_outdoor_hybrid_matcher(overrides, size=max(832, size), ckpt_path=ckpt_path).to(device)

    gray0, sc0, mask0 = load_gray_scale_tensor_cv(
        path0, device, dfactor=32, imsize=size, value_to_scale=max, pad2sqr=True)
    gray1, sc1, mask1 = load_gray_scale_tensor_cv(
        path1, device, dfactor=32, imsize=size, value_to_scale=max, pad2sqr=True)

    batch = {"image0": gray0, "image1": gray1}
    mask = torch.stack([mask0[None], mask1[None]])
    for factor in [8, 16, 32]:
        batch[f"mask0_{factor}x"], batch[f"mask1_{factor}x"] = F.max_pool2d(
            mask, factor, stride=factor
        )
    with torch.no_grad():
        result = net(batch)
