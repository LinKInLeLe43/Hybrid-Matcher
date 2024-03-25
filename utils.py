from typing import List

import hydra
import torch
from torch import nn


def get_outdoor_hybrid_matcher(overrides: List[str]) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = overrides + [
            "experiment=hybrid_matcher_megadepth/eval_ds",
            "model.net.coarse_module.use_flow=false"]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model).eval()
        # net.load_state_dict(torch.load("weights/outdoor.ckpt")["state_dict"])
    return net


if __name__ == "__main__":
    net = get_outdoor_hybrid_matcher([]).cuda()
    data = {
        "image0": torch.rand((1, 1, 640, 480)).cuda(),
        "image1": torch.rand((1, 1, 640, 480)).cuda(),
        "mask0": torch.rand((1, 4800)).bool().cuda(),
        "mask1": torch.rand((1, 4800)).bool().cuda(),
        "center0_mask": torch.rand((1, 300)).bool().cuda(),
        "center1_mask": torch.rand((1, 300)).bool().cuda()}
    with torch.no_grad():
        r = net(data)
    a = 1