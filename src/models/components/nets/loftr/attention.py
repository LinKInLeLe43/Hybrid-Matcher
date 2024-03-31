from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FullAttention(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qk = torch.einsum("nlhd,nshd->nlsh", q, k)
        if q_mask is not None:
            mask = ~(q_mask[:, :, None, None] & kv_mask[:, None, :, None])
            qk.masked_fill_(mask, -1e9)

        qk = torch.softmax(qk / q.shape[3] ** 0.5, dim=2)
        qk = self.dropout(qk)
        out = torch.einsum("nlsh,nshd->nlhd", qk, v)
        return out


class LinearAttention(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.feature_map = lambda x: F.elu(x) + 1.0
        self.eps = eps

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, k = self.feature_map(q), self.feature_map(k)
        if q_mask is not None:
            q = q_mask[:, :, None, None] * q
        if kv_mask is not None:
            k, v = kv_mask[:, :, None, None] * k, kv_mask[:, :, None, None] * v

        kv = torch.einsum("nshd,nshv->nhdv", k, v / v.shape[1])
        div = 1.0 / (torch.einsum("nlhd,nshd->nlh", q, k) + self.eps)
        out = v.shape[1] * torch.einsum("nlhd,nhdv,nlh->nlhv", q, kv, div)
        return out
