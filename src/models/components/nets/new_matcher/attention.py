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
            args = [x.contiguous() for x in [q, k, v]]
            if self.force_flash:
                # FlashAttention does not support mask and FP32 precision
                if mask is not None:
                    raise ValueError()

                with sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    out = F.scaled_dot_product_attention(*args)
            else:
                out = F.scaled_dot_product_attention(*args, attn_mask=mask)
        else:
            scale = q.shape[-1] ** -0.5
            similarity = scale * torch.einsum("...ld,...sd->...ls", q, k)
            if mask is not None:
                similarity.masked_fill_(~mask, -float("inf"))

            out = torch.einsum(
                "...ls,...sc->...lc", F.softmax(similarity, dim=-1), v
            )
            if mask is not None:
                out.nan_to_num_()
        return out
