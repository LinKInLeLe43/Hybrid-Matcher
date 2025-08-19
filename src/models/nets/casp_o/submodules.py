from copy import deepcopy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module


class PyramidFuser(Module):
    def __init__(
        self,
        dims: Sequence[int],
        bias: bool = False,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.lateral_convs.append(nn.Conv2d(dims[0], dims[0], 1, bias=bias))
        for i in range(len(dims) - 1):
            dim0, dim1 = dims[i : i + 2]
            self.lateral_convs.append(nn.Conv2d(dim1, dim0, 1, bias=bias))
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(dim0, dim0, 3, padding=1, bias=bias),
                    nn.BatchNorm2d(dim0),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(dim0, dim1, 3, padding=1, bias=bias),
                )
            )

    def _fuse(self, x_list: List[Tensor]) -> Tensor:
        x = self.lateral_convs[0](x_list[0])
        for i in range(len(x_list) - 1):
            x = F.interpolate(
                x,
                size=x_list[i + 1].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            x = x + self.lateral_convs[i + 1](x_list[i + 1])
            x = self.fusion_blocks[i](x)
        return x

    def forward(
        self, x0_list: Sequence[Tensor], x1_list: Sequence[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        x0_list, x1_list = list(x0_list), list(x1_list)
        if x0_list[0].shape == x1_list[0].shape:
            x_list = [torch.cat(t) for t in zip(x0_list, x1_list)]
            x0, x1 = self._fuse(x_list).chunk(2)
        else:
            x0, x1 = self._fuse(x0_list), self._fuse(x1_list)
        return x0, x1


class RealNVP(Module):
    def __init__(self, dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.register_buffer("loc", torch.zeros(2), persistent=False)
        self.register_buffer("cov", torch.eye(2), persistent=False)
        self.register_buffer(
            "mask",
            torch.tensor([[0.0, 1.0], [1.0, 0.0]] * num_layers),
            persistent=False,
        )

        scale_net, trans_net = self._get_nets(dim)
        self.scale_nets = nn.ModuleList(
            [deepcopy(scale_net) for _ in range(2 * num_layers)]
        )
        self.trans_nets = nn.ModuleList(
            [deepcopy(trans_net) for _ in range(2 * num_layers)]
        )

    def _get_nets(self, dim: int) -> Tuple[Module, Module]:
        scale_net = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Tanh(),
        )
        trans_net = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        return scale_net, trans_net

    @property
    def prior(self) -> Distribution:
        prior = MultivariateNormal(self.loc, self.cov)
        return prior

    def backward_p(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, log_det = x, x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.mask))):
            z_ = z * self.mask[i]
            scale = self.scale_nets[i](z_) * (1 - self.mask[i])
            trans = self.trans_nets[i](z_) * (1 - self.mask[i])
            z = (z - trans) * (-scale).exp() * (1 - self.mask[i]) + z_
            log_det = log_det - scale.sum(dim=1)
        return z, log_det

    def log_prob(self, x: Tensor) -> Tensor:
        z, log_det = self.backward_p(x)
        log_prob = self.prior.log_prob(z) + log_det
        return log_prob
