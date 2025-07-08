from typing import List, Sequence

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


def crop_by_mask(x: Tensor, mask: Tensor) -> List[Tensor]:
    x_list = []
    for b in range(x.shape[0]):
        b_h, b_w = mask[b].sum(dim=0).amax(), mask[b].sum(dim=1).amax()
        x_list.append(x[[b], :, :b_h, :b_w])
    return x_list


def pad_by_mask(x: Tensor, mask: Tensor) -> Tensor:
    n, c, b_h, b_w = x.shape
    _, h, w = mask.shape
    assert n == 1

    out = x.new_zeros(n, c, h, w)
    out[0, :, :b_h, :b_w] = x
    return out
