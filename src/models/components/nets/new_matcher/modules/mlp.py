import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(hidden_depth, out_depth, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            pass
        elif len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
        else:
            assert False

        x = self.linear0(x)
        x = self.gelu(x)
        x = self.linear1(x)

        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        return x
