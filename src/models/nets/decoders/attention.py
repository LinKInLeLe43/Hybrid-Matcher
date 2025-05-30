from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False


class Attention(nn.Module):
    def __init__(self, try_sdpa: bool = False, try_flash: bool = False) -> None:
        super().__init__()
        self.enable_sdpa = try_sdpa and FLASH_AVAILABLE
        self.enable_flash = try_flash and self.enable_sdpa

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mask = None
        if q_mask is not None and kv_mask is not None:
            mask = q_mask[..., :, None] & kv_mask[..., None, :]

        if self.enable_sdpa:
            if self.enable_flash:
                args = [x.contiguous().half() for x in [q, k, v]]
                with sdp_kernel(
                    enable_math=False, enable_flash=True,
                    enable_mem_efficient=False):
                    out = F.scaled_dot_product_attention(
                        *args, attn_mask=mask).to(q.dtype)
            else:
                args = [x.contiguous() for x in [q, k, v]]
                out = F.scaled_dot_product_attention(*args, attn_mask=mask)
        else:
            sc = q.shape[-1]
            similarity = torch.einsum("...ld,...sd->...ls", q, k) / sc ** 0.5
            if mask is not None:
                similarity.masked_fill_(~mask, -1e9)

            out = torch.einsum(
                "...ls,...sc->...lc", F.softmax(similarity, dim=-1), v)
        return out
