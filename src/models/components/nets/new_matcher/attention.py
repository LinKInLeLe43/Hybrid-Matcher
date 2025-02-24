from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self, allow_sdp: bool = False, force_flash: bool = False
    ) -> None:
        super().__init__()
        self.enable_sdp = allow_sdp and FLASH_AVAILABLE
        self.force_flash = force_flash

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = None
        if q_mask is not None and kv_mask is not None:
            mask = q_mask[..., :, None] & kv_mask[..., None, :]

        if self.enable_sdp:
            if self.force_flash:
                args = [x.contiguous().half() for x in [q, k, v]]
                with sdp_kernel(
                    enable_math=False,
                    enable_flash=True,
                    enable_mem_efficient=False,
                ):
                    out = F.scaled_dot_product_attention(
                        *args, attn_mask=mask
                    ).to(q.dtype)
            else:
                args = [x.contiguous() for x in [q, k, v]]
                out = F.scaled_dot_product_attention(*args, attn_mask=mask)
        else:
            scale = q.shape[-1] ** -0.5
            similarity = scale * torch.einsum("...ld,...sd->...ls", q, k)
            if mask is not None:
                similarity.masked_fill_(~mask, -1e9)

            out = torch.einsum(
                "...ls,...sc->...lc", F.softmax(similarity, dim=-1), v
            )
        return out
