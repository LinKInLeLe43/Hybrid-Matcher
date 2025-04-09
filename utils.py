from typing import List

import hydra
import torch
from torch import nn


def get_outdoor_hybrid_matcher(overrides: List[str]) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = overrides + [
            "experiment=new_matcher_megadepth/eval_one_stage"]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model).eval()
        # net.load_state_dict(torch.load("weights/outdoor.ckpt")["state_dict"])
    return net


if __name__ == "__main__":
    net = get_outdoor_hybrid_matcher([])
    data = {
        "color0": torch.rand((1, 3, 896, 896)),
        "color1": torch.rand((1, 3, 896, 896)),
        "image0": torch.rand((1, 1, 896, 896)),
        "image1": torch.rand((1, 1, 896, 896)),
        "mask0_8x": torch.rand((1, 112, 112)).bool(),
        "mask1_8x": torch.rand((1, 112, 112)).bool(),
        "mask0_16x": torch.rand((1, 56, 56)).bool(),
        "mask1_16x": torch.rand((1, 56, 56)).bool(),
        "mask0_32x": torch.rand((1, 28, 28)).bool(),
        "mask1_32x": torch.rand((1, 28, 28)).bool()}
    with torch.no_grad():
        r = net(data)
    a = 1