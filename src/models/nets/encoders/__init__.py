from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .convnextv2 import (
    convnextv2_pico,
    convnextv2_pico_modified,
    convnextv2_tiny,
)
from .dinov2.dpt import DepthAnythingV2
from .repvgg import create_RepVGG_A1, create_RepVGG_A2


class Encoder(Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "repvgg_a1":
            self.backbone = create_RepVGG_A1()
        elif name == "repvgg_a2":
            self.backbone = create_RepVGG_A2()
        elif name == "convnextv2_pico":
            self.backbone = convnextv2_pico()
        elif name == "convnextv2_pico_modified":
            self.backbone = convnextv2_pico_modified()
        elif name == "convnextv2_tiny":
            self.backbone = convnextv2_tiny()
        else:
            raise ValueError("")

        depth_anything_v2 = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        ).eval()
        depth_anything_v2.load_state_dict(
            torch.load("weights/depth_anything_v2_vits.pth")
        )
        self.vit = [depth_anything_v2.pretrained]
        self.depth_head = [depth_anything_v2.depth_head]

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406])[:, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225])[:, None, None],
            persistent=False,
        )

    def _preprocess(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        return x

    def _forward_vit(self, x: Tensor) -> Tensor:
        if self.vit[0].cls_token.device != x.device:
            self.vit[0] = self.vit[0].to(x.device)
            self.depth_head[0] = self.depth_head[0].to(x.device)

        with torch.no_grad():
            h, w = x.shape[2:]
            x = F.interpolate(
                x, size=(h // 16 * 14, w // 16 * 14), mode="bilinear"
            )
            vit_features = self.vit[0].get_intermediate_layers(
                x, [2, 5, 8, 11], return_class_token=True
            )
            out = vit_features[-1][0].unflatten(1, (h // 16, w // 16))
            return out

    def forward(
        self, x0: Tensor, x1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
        x0 = self._preprocess(x0)
        x1 = self._preprocess(x1)
        if x0.shape == x1.shape:
            x = torch.cat([x0, x1])

            x_list = self.backbone(x)
            x0_list, x1_list = map(list, zip(*[x.chunk(2) for x in x_list]))

            prompt0, prompt1 = self._forward_vit(x).chunk(2)
        else:
            x0_list, x1_list = self.backbone(x0), self.backbone(x1)

            prompt0 = self._forward_vit(x0)
            prompt1 = self._forward_vit(x1)
        return x0_list, x1_list, prompt0, prompt1
