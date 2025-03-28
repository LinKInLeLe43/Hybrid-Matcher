from typing import List

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_outdoor_hybrid_matcher(overrides: List[str]) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = [
            *overrides,
            "experiment=new_matcher_megadepth/eval_one_stage",
        ]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model).eval()
        # net.load_state_dict(torch.load("weights/outdoor.ckpt")["state_dict"])
    return net


if __name__ == "__main__":
    net = get_outdoor_hybrid_matcher([])
    mask0_8x = torch.zeros((2, 60, 80)).bool()
    mask1_8x = torch.zeros((2, 60, 60)).bool()
    mask0_8x[:, :32, :48] = True
    mask1_8x[:, :36, :40] = True
    mask0_16x = F.max_pool2d(mask0_8x.float(), 2, stride=2).bool()
    mask1_16x = F.max_pool2d(mask1_8x.float(), 2, stride=2).bool()
    mask0_32x = F.max_pool2d(mask0_8x.float(), 4, stride=4).bool()
    mask1_32x = F.max_pool2d(mask1_8x.float(), 4, stride=4).bool()

    data = {
        "image0": torch.rand((2, 1, 480, 640)),
        "image1": torch.rand((2, 1, 480, 480)),
        "mask0_8x": mask0_8x,
        "mask1_8x": mask1_8x,
        "mask0_16x": mask0_16x,
        "mask1_16x": mask1_16x,
        "mask0_32x": mask0_32x,
        "mask1_32x": mask1_32x,
    }
    with torch.no_grad():
        r = net(data)
    a = 1
