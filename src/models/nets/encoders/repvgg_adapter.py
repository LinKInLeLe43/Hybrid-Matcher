from typing import List

from torch import Tensor

from .repvgg import RepVGG, RepVGGBlock


class RepVGGAdapter(RepVGG):
    def __init__(self, dim_list: List[int], num_blocks_list: List[int]) -> None:
        assert len(num_blocks_list) == len(dim_list) == 3
        width_multiplier = [
            dim / base for dim, base in zip(dim_list, [64, 128, 256])
        ]
        num_blocks_list = [*num_blocks_list, 0]
        width_multiplier = [*width_multiplier, 1.0]
        super().__init__(num_blocks_list, width_multiplier=width_multiplier)
        del self.stage0, self.stage1, self.stage4, self.gap, self.linear

        in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=1,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se,
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        out = self.stage0(x)
        for block in self.stage1:
            out = block(out)
        x_4x = out
        for block in self.stage2:
            out = block(out)
        x_8x = out
        for block in self.stage3:
            out = block(out)
        x_16x = out
        out = [x_4x, x_8x, x_16x]
        return out
