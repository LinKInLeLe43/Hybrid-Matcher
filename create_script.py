from typing import List

import cv2
import hydra
import numpy as np
import torch
from torch import nn


def get_outdoor_hybrid_matcher(overrides: List[str]) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = overrides + [
            "experiment=hybrid_matcher_megadepth/eval_ds",
            "model.net.coarse_module.use_flow=false"]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        module = hydra.utils.instantiate(cfg.model)
        # module.load_state_dict(torch.load("weights/outdoor.ckpt", map_location='cpu')["state_dict"])
        net = module.net.eval()
    return net


if __name__ == "__main__":
    net = get_outdoor_hybrid_matcher([]).cuda()
    path0 = '../assets/MRSDatasets/1optical-optical/pair6-1.jpg'
    path1 = '../assets/MRSDatasets/1optical-optical/pair6-2.jpg'
    image0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    image0 = cv2.resize(image0, (1024, 1024)).astype(np.float32)
    image1 = cv2.resize(image1, (1024, 1024)).astype(np.float32)
    image0 /= image0.max()
    image1 /= image1.max()
    batch = torch.cat((
        torch.from_numpy(image0)[None, None],
        torch.from_numpy(image1)[None, None]))
    with torch.no_grad():
        output = net(batch)
        traced_script_module = torch.jit.script(net)
    traced_script_module.save('EcoMatcher.pt')
