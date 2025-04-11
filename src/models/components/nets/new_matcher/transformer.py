import copy
from typing import List, Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.attention = attention
        self.nchw = False

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False))
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc = self.heads_count

        if x_mask is not None and source_mask is not None:
            x_mask, source_mask = x_mask[:, None], source_mask[:, None]

        q = einops.rearrange(self.q_proj(x), "n l (fc sc) -> n fc l sc", fc=fc)
        k = einops.rearrange(
            self.k_proj(source), "n s (fc sc) -> n fc s sc", fc=fc)
        v = einops.rearrange(
            self.v_proj(source), "n s (fc sc) -> n fc s sc", fc=fc)
        out = self.attention(q, k, v, q_mask=x_mask, kv_mask=source_mask)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)

        out = torch.cat([x, out], dim=2)
        out = self.mlp(out)
        out = self.norm2(out)

        out += x
        return out


class ConvTransformerEncoder(nn.Module):
    def __init__(
        self,
        scale: int,
        depth: int,
        heads_count: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.scale = scale
        self.heads_count = heads_count
        self.attention = attention
        self.nchw = False

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Conv2d(2 * depth, 2 * depth, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * depth, depth, 3, padding=1, bias=False))
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        size: Tuple[int, int],
        x_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        sh, sw, fc = self.scale, self.scale, self.heads_count
        fh, fw = size[0] // sh, size[1] // sw

        if x_mask is not None and source_mask is not None:
            x_mask, source_mask = x_mask[:, None], source_mask[:, None]

        q = einops.rearrange(self.q_proj(x), "n l (fc sc) -> n fc l sc", fc=fc)
        k = einops.rearrange(
            self.k_proj(source), "n s (fc sc) -> n fc s sc", fc=fc)
        v = einops.rearrange(
            self.v_proj(source), "n s (fc sc) -> n fc s sc", fc=fc)
        out = self.attention(q, k, v, q_mask=x_mask, kv_mask=source_mask)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)

        out = torch.cat([x, out], dim=2)
        out = einops.rearrange(
            out, "(n fh fw) (sh sw) c -> n c (fh sh) (fw sw)", fh=fh, sh=sh,
            fw=fw, sw=sw)
        out = self.mlp(out)
        out = einops.rearrange(
            out, "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c", sh=sh, sw=sw)
        out = self.norm2(out)

        out += x
        return out


class AggregatedEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        scale: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.scale = scale
        self.attention = attention
        self.nchw = True

        if scale == 1:
            self.down_q = nn.Identity()
            self.down_kv = nn.Identity()
        else:
            self.down_q = nn.Conv2d(
                depth, depth, scale, stride=scale, groups=depth, bias=False)
            self.down_kv = nn.MaxPool2d(scale, stride=scale)

        self.q_proj = nn.Linear(depth, depth, bias=False)
        self.k_proj = nn.Linear(depth, depth, bias=False)
        self.v_proj = nn.Linear(depth, depth, bias=False)

        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm1 = nn.LayerNorm(depth)

        self.mlp = nn.Sequential(
            nn.Linear(2 * depth, 2 * depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * depth, depth, bias=False))
        self.norm2 = nn.LayerNorm(depth)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        rope: Optional[nn.Module] = None,
        x_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        s, fc = self.scale, self.heads_count

        if x_mask is not None and source_mask is not None:
            x_mask, source_mask = x_mask[:, None], source_mask[:, None]

        q = self.down_q(x).permute(0, 2, 3, 1)
        kv = self.down_kv(source).permute(0, 2, 3, 1)
        q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)

        if rope is not None:
            q, k = rope.rel_pe(q), rope.rel_pe(k)

        q = einops.rearrange(q, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        k = einops.rearrange(k, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        v = einops.rearrange(v, "n h w (fc sc) -> n fc (h w) sc", fc=fc)
        out = self.attention(q, k, v, q_mask=x_mask, kv_mask=source_mask)
        out = einops.rearrange(out, " n fc l sc -> n l (fc sc)")

        out = self.merge(out)
        out = self.norm1(out)
        out = out.transpose(1, 2).unflatten(2, (x.shape[2] // s, x.shape[3] // s))
        if s != 1:
            out = F.interpolate(out, scale_factor=s, mode="bilinear")

        out = torch.cat([x, out], dim=1)
        out = out.permute(0, 2, 3, 1)
        out = self.mlp(out)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        out += x
        return out


class LoFTR(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        types: List[str]
    ) -> None:
        super().__init__()
        self.types = types
        self.nchw = encoder.nchw

        self.layers = nn.ModuleList([copy.deepcopy(encoder) for _ in types])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        rope,
        size0: Optional[Tuple[int, int]] = None,
        size1: Optional[Tuple[int, int]] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self.nchw:
        #     if size0 is None or size1 is None:
        #         raise ValueError("")

        #     feature0 = feature0.transpose(1, 2).unflatten(2, size0).contiguous()
        #     feature1 = feature1.transpose(1, 2).unflatten(2, size1).contiguous()

        for layer, type in zip(self.layers, self.types):
            if type == "self":
                feature0 = layer(
                    feature0, feature0, rope, x_mask=mask0, source_mask=mask0)
                feature1 = layer(
                    feature1, feature1, rope, x_mask=mask1, source_mask=mask1)
            elif type == "cross":
                feature0 = layer(
                    feature0, feature1, x_mask=mask0, source_mask=mask1)
                feature1 = layer(
                    feature1, feature0, x_mask=mask1, source_mask=mask0)
            else:
                raise ValueError("")

        # if self.nchw:
        #     feature0 = feature0.flatten(start_dim=2).transpose(1, 2)
        #     feature1 = feature1.flatten(start_dim=2).transpose(1, 2)
        return feature0, feature1


class FusedSelectiveTransformer(nn.Module):
    def __init__(
        self,
        scale: int,
        depths: Tuple[int, int],
        encoder: nn.Module,
        layer_count: int
    ) -> None:
        super().__init__()
        self.scale = scale

        self.x_up = nn.Conv2d(depths[0], depths[1], 1, bias=False)
        self.y_up = nn.Conv2d(depths[1], depths[1], 1, bias=False)
        self.down = nn.Sequential(
            nn.Conv2d(depths[1], depths[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(depths[1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depths[1], depths[0], 3, padding=1, bias=False))

        self.layers = nn.ModuleList([copy.deepcopy(encoder)
                                     for _ in range(layer_count)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: torch.Tensor,
        y1: torch.Tensor,
        idxes0_to_1: torch.Tensor,
        idxes1_to_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        s = sh = sw = self.scale
        n, _, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape
        fh0, fw0, fh1, fw1 = h0 // sh, w0 // sw, h1 // sh, w1 // sw

        x, y = self.x_up(torch.cat([x0, x1])), self.y_up(torch.cat([y0, y1]))
        x += F.interpolate(y, scale_factor=s, mode="bilinear")
        x0, x1 = einops.rearrange(
            self.down(x),
            "n c (fh sh) (fw sw) -> (n fh fw) (sh sw) c", sh=sh, sw=sw).chunk(2)

        idxes1_to_0 = idxes1_to_0.transpose(1, 2)
        range = torch.arange(n, device=x0.device)[:, None, None]
        _idxes0_to_1 = (idxes0_to_1 + fh1 * fw1 * range).flatten(end_dim=1)
        _idxes1_to_0 = (idxes1_to_0 + fh0 * fw0 * range).flatten(end_dim=1)

        for layer in self.layers:
            x0 = layer(
                x0, x1[_idxes0_to_1].flatten(start_dim=1, end_dim=2), (h0, w0))
            x1 = layer(
                x1, x0[_idxes1_to_0].flatten(start_dim=1, end_dim=2), (h1, w1))

        out0 = einops.rearrange(
            x0, "(n fh fw) (sh sw) c -> n (fh sh fw sw) c", fh=fh0, sh=sh,
            fw=fw0, sw=sw)
        out1 = einops.rearrange(
            x1, "(n fh fw) (sh sw) c -> n (fh sh fw sw) c", fh=fh1, sh=sh,
            fw=fw1, sw=sw)
        selective0 = einops.repeat(
            x0[_idxes1_to_0], "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            fh=fh1, sh=sh, fw=fw1, sw=sw)
        selective1 = einops.repeat(
            x1[_idxes0_to_1], "(n fh fw) k ss c -> n (fh sh fw sw) (k ss) c",
            fh=fh0, sh=sh, fw=fw0, sw=sw)
        idxes0_to_1 = (w1 * s * (idxes0_to_1 // (w1 // s)) +
                       s * (idxes0_to_1 % (w1 // s)))[..., None]
        idxes1_to_0 = (w0 * s * (idxes1_to_0 // (w0 // s)) +
                       s * (idxes1_to_0 % (w0 // s)))[..., None]
        idxes0_to_1 = idxes0_to_1 + idxes0_to_1.new_tensor([0, 1, w1, w1 + 1])
        idxes1_to_0 = idxes1_to_0 + idxes1_to_0.new_tensor([0, 1, w0, w0 + 1])
        idxes0_to_1 = einops.repeat(
            idxes0_to_1, "n (fh fw) k ss -> n (fh sh fw sw) (k ss)",
            fh=fh0, sh=sh, fw=fw0, sw=sw)
        idxes1_to_0 = einops.repeat(
            idxes1_to_0, "n (fh fw) k ss -> n (k ss) (fh sh fw sw)",
            fh=fh1, sh=sh, fw=fw1, sw=sw)
        return out0, out1, selective0, selective1, idxes0_to_1, idxes1_to_0
