import einops
import torch
from torch.nn import functional as F


def crop_windows(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    ww = kernel_size ** 2
    output = F.unfold(x, kernel_size, padding=padding, stride=stride)
    output = einops.rearrange(output, "n (c ww) l -> n l ww c", ww=ww)
    return output
