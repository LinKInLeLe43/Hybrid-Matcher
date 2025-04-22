from pathlib import Path
from typing import List, Optional

import hydra
import omegaconf
import torch.nn as nn

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def compose_configs(overrides: List[str]) -> omegaconf.DictConfig:
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIR),
        job_name="compose_configs",
    ):
        return hydra.compose(
            config_name="eval",
            overrides=[
                "experiment=new_matcher_megadepth/eval_one_stage",
                *overrides,
            ],
        )


def casp_full(
    image_size: int = 832, threshold: float = 0.3, weight: Optional[str] = None
) -> nn.Module:
    overrides = [
        f"data.test_dataset.image_size={image_size}",
        f"++model.net.coarse_matching.threshold={threshold}",
    ]
    config = compose_configs(overrides)
    model = hydra.utils.instantiate(config.model).eval()
    if weight is not None:
        model.load_state_dict(
            torch.load(weight, map_location="cpu")["state_dict"]
        )
    return model


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    net = casp_full()
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
