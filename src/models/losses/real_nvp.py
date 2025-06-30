from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module


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
