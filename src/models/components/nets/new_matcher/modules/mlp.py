from typing import Optional, Tuple

import einops
import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True,
        kernel_size0: int = 1,
        kernel_size1: int = 1,
        padding0: Optional[int] = None,
        padding1: Optional[int] = None,
        act: nn.Module = nn.GELU()
    ) -> None:
        super().__init__()

        self.proj0 = self._create_proj(
            in_depth, hidden_depth, kernel_size0, padding0, bias)
        self.act = act
        self.proj1 = self._create_proj(
            hidden_depth, out_depth, kernel_size1, padding1, bias)

    def _create_proj(
        self,
        in_depth: int,
        out_depth: int,
        kernel_size: int,
        padding: Optional[int],
        bias: bool
    ) -> nn.Module:
        if kernel_size == 1:
            if padding is not None:
                raise ValueError("")

            proj = nn.Linear(in_depth, out_depth, bias=bias)
        else:
            padding = padding if padding is not None else kernel_size // 2
            proj = nn.Conv2d(
                in_depth, out_depth, kernel_size, padding=padding, bias=bias)
        return proj

    def _forward(
        self,
        module: nn.Module,
        x: torch.Tensor,
        flag: str,
        size: Optional[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, str]:
        if isinstance(module, nn.Linear):
            if flag == "nchw":
                x = x.permute(0, 2, 3, 1)
                flag = "nhwc"
            elif flag == "nhwc" or flag == "nlc":
                pass
            else:
                assert False
        elif isinstance(module, nn.Conv2d):
            if flag == "nchw":
                pass
            elif flag == "nhwc":
                x = x.permute(0, 3, 1, 2)
            elif flag == "nlc":
                if size is None:
                    raise ValueError("")
                x = x.transpose(1, 2).unflatten(2, size)
            else:
                assert False
            flag = "nchw"
        else:
            assert False
        out = module(x)
        return out, flag

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        if len(x.shape) == 3:
            x_flag = "nlc"
        elif len(x.shape) == 4:
            x_flag = "nchw"
        else:
            raise ValueError("")

        out, out_flag = self._forward(self.proj0, x, x_flag, size)
        out = self.act(out)
        out, out_flag = self._forward(self.proj1, out, out_flag, size)

        if out_flag != x_flag:
            out_flag = " ".join(f"{c}" for c in out_flag).replace("l", "(h w)")
            x_flag = " ".join(f"{c}" for c in x_flag).replace("l", "(h w)")
            out = einops.rearrange(out, f"{out_flag} -> {x_flag}")
        return out
