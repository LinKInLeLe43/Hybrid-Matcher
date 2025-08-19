from typing import List

from torch import Tensor

from .repvgg import RepVGG, RepVGGBlock


class RepVGGAdapter(RepVGG):
    def __init__(self, dim_list: List[int], num_blocks_list: List[int]) -> None:
        assert len(num_blocks_list) == len(dim_list) == 3
        num_blocks_list = [*num_blocks_list, 1]
        width_multiplier = [
            dim / base for dim, base in zip(dim_list, [64, 128, 256])
        ]
        width_multiplier = [*width_multiplier, 1.0]
        super().__init__(num_blocks_list, width_multiplier=width_multiplier)
        del self.stage0, self.stage4, self.gap, self.linear

        in_planes = self.in_planes
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=1,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se,
        )
        self.in_planes = in_planes

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stage0(x)
        for block in self.stage1:
            x = block(x)
        x_4x = x
        for block in self.stage2:
            x = block(x)
        x_8x = x
        for block in self.stage3:
            x = block(x)
        x_16x = x
        out = [x_4x, x_8x, x_16x]
        return out
