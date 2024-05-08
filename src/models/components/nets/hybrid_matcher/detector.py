import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()

        self.linear0 = nn.Linear(depth, depth // 2)
        self.linear1 = nn.Linear(depth // 2, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.linear1(x)
        heatmap = F.log_softmax(x[:, :, 0], dim=1)
        return heatmap
