from typing import Sequence

from torch import Tensor


def window_partition(x: Tensor, stride: int) -> Tensor:
    n, c, h, w = x.shape
    fh, fw = h // stride, w // stride
    x = (
        x.view(n, c, fh, stride, fw, stride)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(n, fh * fw, stride * stride, c)
    )
    return x


def window_unpartition(x: Tensor, size: Sequence[int], stride: int) -> Tensor:
    n, _, _, c = x.shape
    fh, fw = size
    x = (
        x.view(n, fh, fw, stride, stride, c)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(n, c, fh * stride, fw * stride)
    )
    return x
