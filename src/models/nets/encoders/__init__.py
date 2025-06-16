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

    def _preprocess_for_vit(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        return x

    def _preprocess_for_conv(self, x: Tensor) -> Tensor:
        x = 0.299 * x[:, [0]] + 0.587 * x[:, [1]] + 0.114 * x[:, [2]]
        return x

    def _forward_vit(self, x: Tensor) -> Tensor:
        if self.vit[0].cls_token.device != x.device:
            self.vit[0] = self.vit[0].to(x.device)
            self.depth_head[0] = self.depth_head[0].to(x.device)

        with torch.no_grad():
            patch_h, patch_w = x.shape[-2] // 32, x.shape[-1] // 32
            x = F.interpolate(
                x,
                size=(patch_h * 14, patch_w * 14),
                mode="bilinear",
                align_corners=True,
            )
            features = self.vit[0].get_intermediate_layers(
                x, [2, 5, 8, 11], return_class_token=True
            )
            out = features[-1][0]
            # out = self.depth_head[0](features, patch_h, patch_w)
            return out

    def forward(
        self, x0: Tensor, x1: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if x0.shape == x1.shape:
            x = torch.cat([x0, x1])
            patch_h, patch_w = x.shape[-2] // 32, x.shape[-1] // 32
            x_list = self.backbone(self._preprocess_for_conv(x))
            x0_list, x1_list = map(list, zip(*[x.chunk(2) for x in x_list]))

            x0, x1 = self._forward_vit(
                self._preprocess_for_vit(x)
            ).chunk(2)
            sim = x0 @ x1.transpose(-1, -2)
            _, indices = sim.topk(8, dim=-1)
            _x = 648
            _y = indices[0, _x]
            result = {}
            result["points0"] = 4 * torch.stack([_x % patch_w, _x // patch_w], dim=1).float()
            result["points1"] = 4 * torch.stack([_y % patch_w, _y // patch_w], dim=1).float()
            result["idxes"] = [torch.tensor([0])]
        # else:
        #     x0_list = self.backbone(self._preprocess_for_conv(x0))
        #     x1_list = self.backbone(self._preprocess_for_conv(x1))

        #     x0_list[-2] = self._forward_vit(self._preprocess_for_vit(x0))
        #     x1_list[-2] = self._forward_vit(self._preprocess_for_vit(x1))
        return result
