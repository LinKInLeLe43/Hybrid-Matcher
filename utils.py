from typing import List, Optional

import cv2
import hydra
import torch
import torch.nn as nn


def get_outdoor_hybrid_matcher(
    overrides: List[str], ckpt_path: Optional[str]
) -> nn.Module:
    with hydra.initialize(version_base="1.3", config_path="./configs"):
        overrides = [
            *overrides,
            "experiment=new_matcher_megadepth/eval_one_stage",
        ]
        cfg = hydra.compose(config_name="eval", overrides=overrides)
        net = hydra.utils.instantiate(cfg.model.net).eval()
        if ckpt_path is not None:
            net.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])

        for m in net.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
    return net


if __name__ == "__main__":
    size = 1024
    threshold = 0.3
    ckpt_path = "/Users/linkinlele43/Downloads/casp.ckpt"
    net = get_outdoor_hybrid_matcher(
        [
            f"++model.net.coarse_matching.threshold={threshold}",
            f"++model.net.rope.test_size=[{size},{size}]",
        ],
        ckpt_path=ckpt_path,
    )

    path0 = "/Users/linkinlele43/Downloads/assets/pair1-1.png"
    path1 = "/Users/linkinlele43/Downloads/assets/pair1-2.png"
    image0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image0 = cv2.resize(image0, (size, size)) / 255.0
    image1 = cv2.resize(image1, (size, size)) / 255.0
    input = torch.cat(
        [
            torch.from_numpy(image0)[None, None].float(),
            torch.from_numpy(image1)[None, None].float(),
        ]
    )

    with torch.no_grad():  
        traced_script_module = torch.jit.script(net)
    traced_script_module.save("CasP_cpu_1024_v6.26.pt")
    # import zipfile, humanize, pathlib

    # zf = zipfile.ZipFile("CasP_cpu_1024_v6.26.pt")
    # for i in sorted(zf.infolist(), key=lambda x: -x.file_size)[:10]:
    #     print(i.filename, humanize.naturalsize(i.file_size))
    # _net = torch.load("CasP_cpu_1024_v6.26.pt").eval()
    # print(net(input).shape)
    # print(_net(input).shape)
    # input = input.to("cuda:0")
    # _net = _net.to("cuda:0")
    # print(_net(input).shape)
