from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..encoders.repvgg import create_RepVGG_A0
from ..positional_encoding import SinusoidalPositionalEncoding
from ..submodules import PyramidFuser
from ..utils import crop_to_mask, pad_to_mask, patchify, unpatchify
from .transformer import RegionSelectiveTransformerLayer, TransformerLayer


class GlobalDecoder(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        stride: int,
        factor: int,
        train_size: int,
        test_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.stride = stride

        layer = TransformerLayer(dim, num_heads, stride=stride, **kwargs)
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )
        encoding = SinusoidalPositionalEncoding(
            dim, factor, train_size, test_size=test_size
        )
        self.encodings = nn.ModuleList([encoding for _ in range(num_layers)])

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        enable_crop: bool = True,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n = len(x0)
        x0, x1 = x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1)
        if enable_crop and mask0 is not None and mask1 is not None:
            x0_list, x1_list = crop_to_mask(x0, mask0), crop_to_mask(x1, mask1)
            for b in range(n):
                b_x0, b_x1 = x0_list[b], x1_list[b]
                for layer, encoding in zip(self.layers, self.encodings):
                    encoding0, encoding1 = encoding(b_x0), encoding(b_x1)
                    b_x0, b_x1 = layer(
                        b_x0, b_x1, encoding0=encoding0, encoding1=encoding1
                    )
                x0_list[b] = pad_to_mask(b_x0, mask0[[b]])
                x1_list[b] = pad_to_mask(b_x1, mask1[[b]])
            x0, x1 = torch.cat(x0_list), torch.cat(x1_list)
        else:
            mask00 = mask11 = mask01 = None
            if mask0 is not None and mask1 is not None:
                if self.stride > 1:
                    mask0, mask1 = mask0.float(), mask1.float()
                    mask0 = F.max_pool2d(mask0, self.stride).bool()
                    mask1 = F.max_pool2d(mask1, self.stride).bool()
                mask00 = mask0.reshape(n, -1, 1) & mask0.reshape(n, 1, -1)
                mask11 = mask1.reshape(n, -1, 1) & mask1.reshape(n, 1, -1)
                mask01 = mask0.reshape(n, -1, 1) & mask1.reshape(n, 1, -1)
            for layer, encoding in zip(self.layers, self.encodings):
                encoding0, encoding1 = encoding(x0), encoding(x1)
                x0, x1 = layer(
                    x0,
                    x1,
                    encoding0=encoding0,
                    encoding1=encoding1,
                    mask00=mask00,
                    mask11=mask11,
                    mask01=mask01,
                )
        x0 = x0.permute(0, 3, 1, 2).contiguous()
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        return x0, x1


class RegionSelectiveDecoder(Module):
    def __init__(
        self,
        dim_list: List[int],
        num_heads: int,
        stride: int,
        topk: int,
        num_layers: int,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(dim_list) == 2
        self.stride = stride
        self.topk = topk

        self.fuser = PyramidFuser(dim_list)
        layer = RegionSelectiveTransformerLayer(
            dim_list[1], num_heads, stride, **kwargs
        )
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x0_list: List[Tensor],
        x1_list: List[Tensor],
        enable_unpatchify: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        assert len(x0_list) == len(x1_list) == 2
        grid_size0 = tuple(x0_list[0].shape[-2:])
        grid_size1 = tuple(x1_list[0].shape[-2:])
        scale = x0_list[0].shape[1] ** -0.5

        x0_prior = x0_list[0].flatten(start_dim=2)
        x1_prior = x1_list[0].flatten(start_dim=2)
        similarity = x0_prior.transpose(-2, -1) @ x1_prior * scale
        _, indices0_to_1 = similarity.topk(self.topk, dim=-1)
        _, indices1_to_0 = similarity.transpose(-2, -1).topk(self.topk, dim=-1)
        if not self.training:
            similarity = None

        x0, x1 = self.fuser(x0_list, x1_list)
        x0, x1 = patchify(x0, self.stride), patchify(x1, self.stride)
        for layer in self.layers:
            x0, x1 = layer(x0, x1, indices0_to_1, indices1_to_0)
        if enable_unpatchify:
            x0 = unpatchify(x0, grid_size0, self.stride)
            x1 = unpatchify(x1, grid_size1, self.stride)
        return x0, x1, indices0_to_1, indices1_to_0, similarity


class Decoder(Module):
    def __init__(
        self,
        dim_list: List[int],
        num_heads_list: List[int],
        stride_list: List[int],
        num_layers_list: List[int],
        topk_list: List[int],
        num_iters: int,
        global_patch_size: int,
        factor: int,
        train_size: int,
        test_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.stride_list = stride_list
        factor = factor * global_patch_size

        global_decoder = GlobalDecoder(
            dim_list[0],
            num_heads_list[0],
            num_layers_list[0],
            global_patch_size,
            factor,
            train_size,
            test_size=test_size,
            **kwargs,
        )
        region_selective_decoders = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            region_selective_decoders.append(
                RegionSelectiveDecoder(
                    dim_list[i : i + 2],
                    num_heads_list[i + 1],
                    stride_list[i],
                    topk_list[i],
                    num_layers_list[i + 1],
                    **kwargs,
                )
            )
        conv_maaker = create_RepVGG_A0()
        conv_net = nn.Sequential()
        for i in reversed(range(len(dim_list) - 1)):
            conv_maaker.in_planes = dim_list[i + 1]
            conv_net.append(
                nn.Sequential(
                    *conv_maaker._make_stage(dim_list[i], 4, stride_list[i])
                )
            )
        decoder = nn.ModuleDict(
            {
                "global_decoder": global_decoder,
                "region_selective_decoders": region_selective_decoders,
                "conv_net": conv_net,
            }
        )
        self.decoders = nn.ModuleList(
            [deepcopy(decoder) for _ in range(num_iters)]
        )
        self.decoders[-1].pop("conv_net")

        self.init_weights()

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x0_list: List[Tensor],
        x1_list: List[Tensor],
        enable_crop: bool = True,
        mask0: Optional[Tensor] = None,
        mask1: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        mask0_global = mask1_global = None
        if mask0 is not None and mask1 is not None:
            stride_global = torch.tensor(self.stride_list).prod().int().item()
            mask0_global, mask1_global = mask0.float(), mask1.float()
            mask0_global = F.max_pool2d(mask0_global, stride_global).bool()
            mask1_global = F.max_pool2d(mask1_global, stride_global).bool()

        n = len(self.stride_list)
        similarity_list = [[] for _ in range(n + 1)]
        for decoder in self.decoders:
            is_last = "conv_net" not in decoder

            x0, x1 = decoder["global_decoder"](
                x0_list[0],
                x1_list[0],
                enable_crop=enable_crop,
                mask0=mask0_global,
                mask1=mask1_global,
            )

            for i in range(n):
                x0, x1, indices0_to_1, indices1_to_0, similarity = decoder[
                    "region_selective_decoders"
                ][i](
                    [x0, x0_list[i + 1]],
                    [x1, x1_list[i + 1]],
                    enable_unpatchify=(not is_last)
                    or (i != n - 1)
                    or self.training,
                )
                similarity_list[i].append(similarity)

            grid_size0 = tuple(x0_list[-2].shape[-2:])
            grid_size1 = tuple(x1_list[-2].shape[-2:])
            if self.training:
                scale = x0.shape[1] ** -0.5
                x0_, x1_ = x0.flatten(start_dim=2), x1.flatten(start_dim=2)
                similarity = x0_.transpose(-2, -1) @ x1_ * scale
            else:
                similarity = None
            similarity_list[-1].append(similarity)

            if not is_last:
                x0_list[0] = x0_list[0] + decoder["conv_net"](x0)
                x1_list[0] = x1_list[0] + decoder["conv_net"](x1)

        results = {
            "x0": x0,
            "x1": x1,
            "grid_size0": grid_size0,
            "grid_size1": grid_size1,
            "indices0_to_1": indices0_to_1,
            "indices1_to_0": indices1_to_0,
            "similarity_list": similarity_list,
        }
        return results
