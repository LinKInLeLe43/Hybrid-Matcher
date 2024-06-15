from typing import List

import hydra
import torch
from torch import nn


def get_outdoor_hybrid_matcher(overrides: List[str]) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = overrides + [
            "experiment=new_matcher_megadepth/eval_ds"]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model).eval()
        # net.load_state_dict(torch.load("weights/outdoor.ckpt")["state_dict"])
    return net


if __name__ == "__main__":
    net = get_outdoor_hybrid_matcher([]).cuda()
    data = {
        "image0": torch.rand((1, 1, 480, 640)).cuda(),
        "image1": torch.rand((1, 1, 480, 640)).cuda(),
        "mask0_8x": torch.rand((1, 60, 80)).bool().cuda(),
        "mask1_8x": torch.rand((1, 60, 80)).bool().cuda(),
        "mask0_16x": torch.rand((1, 30, 40)).bool().cuda(),
        "mask1_16x": torch.rand((1, 30, 40)).bool().cuda(),
        "mask0_32x": torch.rand((1, 15, 20)).bool().cuda(),
        "mask1_32x": torch.rand((1, 15, 20)).bool().cuda()}
    with torch.no_grad():
        r = net(data)
    a = 1