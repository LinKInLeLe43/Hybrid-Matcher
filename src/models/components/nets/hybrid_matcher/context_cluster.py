import copy
from typing import List, Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.linear1 = nn.Linear(hidden_depth, out_depth, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            pass
        elif len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
        else:
            raise ValueError("")

        x = self.linear0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout(x)

        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Mlp3x3(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.conv = nn.Conv2d(hidden_depth, out_depth, 3, padding=1, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Size] = None
    ) -> torch.Tensor:
        if len(x.shape) == 3:
            if size is None:
                raise ValueError("")
            x = x.unflatten(1, size)
            flatten = True
        elif len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
            flatten = False
        else:
            raise ValueError("")

        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.dropout(x)

        if flatten:
            x = x.flatten(start_dim=2).transpose(1, 2)
        return x


class LocallyEnhancedFeedForward(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(in_depth, hidden_depth, 1, bias=bias)
        self.conv1 = nn.Conv2d(
            hidden_depth, hidden_depth, 3, padding=1, groups=hidden_depth,
            bias=bias)
        self.conv2 = nn.Conv2d(hidden_depth, out_depth, 1, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

        self.norm0 = nn.BatchNorm2d(hidden_depth)
        self.norm1 = nn.BatchNorm2d(hidden_depth)
        self.norm2 = nn.BatchNorm2d(out_depth)

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Size] = None
    ) -> torch.Tensor:
        if len(x.shape) == 3:
            if size is None:
                raise ValueError("")
            x = x.transpose(1, 2).unflatten(2, size).contiguous()
            flatten = True
        elif len(x.shape) == 4:
            flatten = False
        else:
            raise ValueError("")

        x = self.conv0(x)
        x = self.norm0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        if flatten:
            x = x.flatten(start_dim=2).transpose(1, 2)
        return x


class LocalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        center_size: int,
        fold_size: int,
        bias: bool = True,
        use_efficient: bool = True
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.center_size = center_size
        self.fold_size = fold_size
        self.use_efficient = use_efficient

        self.proj = nn.Conv2d(in_depth, 2 * hidden_depth, 1, bias=bias)
        self.center_proposal = nn.AdaptiveAvgPool2d(center_size)
        self.merge = nn.Conv2d(hidden_depth, in_depth, 1, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc, fh, fw = self.heads_count, self.fold_size, self.fold_size
        n, c, h, w = x.shape
        sh, sw = h // fh, w // fw
        m, l, s = n * fc * fh * fw, sh * sw, self.center_size ** 2
        device = x.device

        x = self.proj(x)
        x = einops.rearrange(
            x, "n c (fh sh) (fw sw) -> (n fh fw) c sh sw", fh=fh, fw=fw)
        if mask is not None and fh == 1 and fw == 1:
            center = []
            for b_x, b_mask in zip(x, mask):
                b_h = b_mask.sum(dim=0).amax().item()
                b_w = b_mask.sum(dim=1).amax().item()
                center.append(self.center_proposal(b_x[:, :b_h, :b_w]))
            center = torch.stack(center)
        else:
            center = self.center_proposal(x)
        x = einops.rearrange(
            x, "(n fh fw) (fc sc) sh sw -> (n fc fh fw) (sh sw) sc", fc=fc,
            fh=fh, fw=fw)
        center = einops.rearrange(
            center, "(n fh fw) (fc sc) sh sw -> (n fc fh fw) (sh sw) sc", fc=fc,
            fh=fh, fw=fw)
        x_point, x_value = x.chunk(2, dim=2)
        center_point, center_value = center.chunk(2, dim=2)

        norm_x_point = F.normalize(x_point, dim=2)
        norm_center_point = F.normalize(center_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x_point, norm_center_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.repeat(
                mask, "n (fh sh) (fw sw) -> (n fc fh fw) (sh sw) s", fc=fc,
                fh=fh, fw=fw, s=s)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.use_efficient:
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, x_value, center_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, x_value, center_value))

            cat_ones = torch.ones_like(x_value[:, [0]])
            cat_x_value = torch.cat([x_value, cat_ones], dim=1)
            cat_ones = torch.ones_like(center_value[:, [0]])
            cat_center_value = torch.cat([center_value, cat_ones], dim=1)
            aggregated = cat_center_value.index_add_(
                0, max_sim_idxes, max_sim_values[:, None] * cat_x_value)
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = (max_sim_values[:, None] *
                          aggregated.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        else:
            mask = torch.zeros_like(similarities)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarities = (mask * similarities)[..., None]

            aggregated = (center_value +
                          (similarities * x_value[:, :, None, :]).sum(dim=1))
            aggregated /= 1 + similarities.sum(dim=1)
            dispatched = (similarities * aggregated[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw) (sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        dispatched = self.merge(dispatched)
        return dispatched


class GlobalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True,
        use_efficient: bool = True
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.use_efficient = use_efficient

        self.proj0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.proj1 = nn.Linear(in_depth, 2 * hidden_depth, bias=bias)
        self.merge = nn.Linear(hidden_depth, in_depth, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x0: torch.Tensor,
        center1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc = self.heads_count
        n, l, c = x0.shape
        _, s, _ = center1.shape
        m = n * fc
        device = x0.device

        x0_point, center1 = self.proj0(x0), self.proj1(center1)
        x0_point = einops.rearrange(
            x0_point, "n l (fc sc) -> (n fc) l sc", fc=fc)
        center1 = einops.rearrange(center1, "n s (fc sc) -> (n fc) s sc", fc=fc)
        center1_point, center1_value = center1.chunk(2, dim=2)

        norm_x0_point = F.normalize(x0_point, dim=2)
        norm_center1_point = F.normalize(center1_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x0_point, norm_center1_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.repeat(mask, "n l s -> (n fc) l s", fc=fc)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.use_efficient:
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, center1_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, center1_value))

            dispatched = (max_sim_values[:, None] *
                          center1_value.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched, "(n fc l) sc -> n l (fc sc)", fc=fc, l=l)
            dispatched = self.merge(dispatched)
        else:
            raise NotImplementedError("")
        return dispatched


class BidirectionalGlobalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        downsample_size: int,
        anchor_size: int,
        bias: bool = True,
        use_efficient: bool = True
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.downsample_size = downsample_size
        self.anchor_size = anchor_size
        self.use_efficient = use_efficient

        self.down_qk = nn.Conv2d(
            in_depth, in_depth, downsample_size, stride=downsample_size,
            groups=in_depth, bias=bias)
        self.down_v = nn.MaxPool2d(downsample_size, stride=downsample_size)
        self.norm = nn.LayerNorm(in_depth)
        self.proj_qk = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.proj_v = nn.Linear(in_depth, hidden_depth, bias=bias)

        self.proj_point = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.anchor_proposal = nn.AdaptiveAvgPool2d(anchor_size)
        self.down_alpha = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))
        self.down_beta = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.merge0 = nn.Linear(hidden_depth, in_depth, bias=bias)
        self.merge1 = nn.Linear(hidden_depth, in_depth, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        self_down_mask: Optional[torch.Tensor] = None,
        cross_down_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        fc, r, s = self.heads_count, self.downsample_size, self.anchor_size ** 2
        fn_n, c, h, w = x.shape
        sh, sc, m = h // r, c // fc, fn_n * fc
        device = x.device

        down_x_qk, down_x_v = self.down_qk(x), self.down_v(x)
        down_x_qk = down_x_qk.permute(0, 2, 3, 1)
        down_x_v = down_x_v.permute(0, 2, 3, 1)
        down_x_qk, down_x_v = self.norm(down_x_qk), self.norm(down_x_v)
        down_x_qk, down_x_v = self.proj_qk(down_x_qk), self.proj_v(down_x_v)
        down_x = torch.cat([down_x_qk, down_x_v], dim=3).permute(0, 3, 1, 2)
        if self_down_mask is not None:
            anchor = []
            for b_down_x, b_down_mask in zip(down_x, self_down_mask):
                b_h = b_down_mask.sum(dim=0).amax().item()
                b_w = b_down_mask.sum(dim=1).amax().item()
                anchor.append(self.anchor_proposal(b_down_x[:, :b_h, :b_w]))
            anchor = torch.stack(anchor)
        else:
            anchor = self.anchor_proposal(down_x)
        down_x_qk, down_x_v = einops.rearrange(
            down_x, "(fn n) (fp fc sc) sh sw -> fp (fn n fc) (sh sw) sc", fn=2,
            fp=2, fc=fc)
        anchor_qk, anchor_v = einops.rearrange(
            anchor, "(fn n) (fp fc sc) sh sw -> fp (fn n fc) (sh sw) sc", fn=2,
            fp=2, fc=fc)

        # ---------- Attention ----------
        down_x0_qk, down_x1_qk = down_x_qk.chunk(2)
        down_x0_v, down_x1_v = down_x_v.chunk(2)
        if True:  # TODO: use flash attention
            sim = torch.einsum("nld,nsd->nls", down_x0_qk, down_x1_qk)
            if cross_down_mask is not None:
                sim.masked_fill_(~cross_down_mask, float("-inf"))
            sim /= sc ** 0.5
            att01 = F.softmax(sim, dim=2).nan_to_num()
            att10 = F.softmax(sim.transpose(1, 2), dim=2).nan_to_num()
            att_message0 = torch.einsum("nls,nsd->nld", att01, down_x1_v)
            att_message1 = torch.einsum("nsl,nld->nsd", att10, down_x0_v)
        att_message = torch.cat([att_message0, att_message1])
        att_message = einops.rearrange(
            att_message, "(fn n fc) (sh sw) sc -> (fn n) sh sw (fc sc)",
            fn=2, fc=fc, sh=sh)
        att_message = self.merge0(att_message).permute(0, 3, 1, 2)
        att_message = F.interpolate(
            att_message, scale_factor=r, mode="bilinear")

        # ---------- Self Cluster Aggregating ----------
        norm_down_x_qk = F.normalize(down_x_qk, dim=2)
        norm_anchor_v = F.normalize(anchor_qk, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_down_x_qk, norm_anchor_v)
        similarities = self.down_alpha * similarities + self.down_beta
        if self_down_mask is not None:
            mask = einops.repeat(
                self_down_mask, "(fn n) sh sw -> (fn n fc) (sh sw) s", fn=2,
                fc=fc, s=s)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.use_efficient:
            down_x = torch.cat([down_x_qk, down_x_v], dim=2)
            anchor = torch.cat([anchor_qk, anchor_v], dim=2)
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            (max_sim_values, max_sim_idxes, down_x, anchor) = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, down_x, anchor))

            cat_ones = torch.ones_like(down_x[:, [0]])
            cat_down_x_value = torch.cat([down_x, cat_ones], dim=1)
            cat_ones = torch.ones_like(anchor[:, [0]])
            cat_anchor_value = torch.cat([anchor, cat_ones], dim=1)
            aggregated = cat_anchor_value.index_add_(
                0, max_sim_idxes, max_sim_values[:, None] * cat_down_x_value)
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
        else:
            raise NotImplementedError("")

        # ---------- Cross Cluster Dispatching ----------
        x_point = self.proj_point(x.permute(0, 2, 3, 1))
        x_point = einops.rearrange(
            x_point, "(fn n) h w (fc sc) -> (fn n fc) (h w) sc", fn=2, fc=fc)
        aggregated0, aggregated1 = aggregated.chunk(2)
        aggregated = torch.cat([aggregated1, aggregated0])
        aggregated_point, aggregated_value = einops.rearrange(
            aggregated, "(fn n fc s) (fp sc) -> fp (fn n fc) s sc", fn=2, fc=fc,
            fp=2, s=s)

        norm_x_point = F.normalize(x_point, dim=2)
        norm_aggregated_point = F.normalize(aggregated_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x_point, norm_aggregated_point)
        similarities = self.alpha * similarities + self.beta
        if self_mask is not None:
            mask = einops.repeat(
                self_mask, "(fn n) h w -> (fn n fc) (h w) s", fn=2, fc=fc, s=s)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.use_efficient:
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, aggregated_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, aggregated_value))

            dispatched = (max_sim_values[:, None] *
                          aggregated_value.index_select(0, max_sim_idxes))
        else:
            raise NotImplementedError("")
        coc_message = einops.rearrange(
            dispatched, "(fn n fc h w) sc -> (fn n) h w (fc sc)", fn=2, fc=fc,
            h=h, w=w)
        coc_message = self.merge1(coc_message).permute(0, 3, 1, 2)

        message = torch.cat([att_message, coc_message], dim=1)
        return message


class LocalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        center_size: int,
        fold_size: int,
        bias: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: Optional[float] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.use_layer_scale = use_layer_scale

        if use_layer_scale:
            if layer_scale_value is None:
                raise ValueError("")
            self.layer_scale = nn.Parameter(
                layer_scale_value * torch.ones((in_depth,)))

        self.cluster = LocalCluster(
            in_depth, hidden_depth, heads_count, center_size, fold_size,
            bias=bias)
        self.norm0 = nn.GroupNorm(1, in_depth)

        self.mlp = Mlp(
            2 * in_depth, 2 * in_depth, in_depth, bias=bias, dropout=dropout)
        self.norm1 = nn.GroupNorm(1, in_depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        new_x = self.cluster(x, mask=mask)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x, new_x], dim=1)
        new_x = self.mlp(new_x)
        new_x = self.norm1(new_x)

        if self.use_layer_scale:
            new_x *= self.layer_scale[:, None, None]
        new_x += x
        return new_x


class GlobalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True,
        use_flow: bool = False,
        flow_depth: Optional[int] = None,
        use_layer_scale: bool = False,
        layer_scale_value: Optional[float] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.use_flow = use_flow
        self.use_layer_scale = use_layer_scale

        out_depth = in_depth
        if use_flow:
            if flow_depth is None:
                raise ValueError("")
            out_depth += flow_depth
        if use_layer_scale:
            if layer_scale_value is None:
                raise ValueError("")
            self.layer_scale = nn.Parameter(
                layer_scale_value * torch.ones((out_depth,)))

        self.cluster = GlobalCluster(
            in_depth, hidden_depth, heads_count, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = Mlp3x3(
            in_depth + out_depth, in_depth + out_depth, out_depth, bias=bias,
            dropout=dropout)
        self.norm1 = nn.LayerNorm(out_depth)

    def forward(
        self,
        x0: torch.Tensor,
        center1: torch.Tensor,
        size0: torch.Size,
        flow0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_flow0 = None
        if self.use_flow:
            if flow0 is None:
                raise ValueError("")
            c0, c1 = x0.shape[2], flow0.shape[2]

        new_x0 = self.cluster(x0, center1, mask=mask)
        new_x0 = self.norm0(new_x0)

        if self.use_flow:
            x0 = torch.cat([x0, flow0], dim=2)
        new_x0 = torch.cat([x0, new_x0], dim=2)
        new_x0 = self.mlp3x3(new_x0, size=size0)
        new_x0 = self.norm1(new_x0)

        if self.use_layer_scale:
            new_x0 *= self.layer_scale
        new_x0 += x0
        if self.use_flow:
            new_x0, new_flow0 = new_x0.split([c0, c1], dim=2)
        return new_x0, new_flow0


class BidirectionalGlobalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True,
        use_flow: bool = False,
        flow_depth: Optional[int] = None,
        use_layer_scale: bool = False,
        layer_scale_value: Optional[float] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.use_flow = use_flow
        self.use_layer_scale = use_layer_scale

        out_depth = in_depth
        if use_flow:
            if flow_depth is None:
                raise ValueError("")
            out_depth += flow_depth
        if use_layer_scale:
            if layer_scale_value is None:
                raise ValueError("")
            self.layer_scale = nn.Parameter(
                layer_scale_value * torch.ones((out_depth,)))

        self.cluster = BidirectionalGlobalCluster(
            in_depth, hidden_depth, heads_count, 4, 8, bias=bias)
        self.norm0 = nn.GroupNorm(1, 2 * in_depth)

        self.mlp3x3 = Mlp3x3(
            2 * in_depth + out_depth, in_depth + out_depth, out_depth,
            bias=bias, dropout=dropout)
        self.norm1 = nn.GroupNorm(1, out_depth)

    def forward(
        self,
        x: torch.Tensor,
        size: torch.Size,
        flow: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        self_down_mask: Optional[torch.Tensor] = None,
        cross_down_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_flow = None
        if self.use_flow:
            if flow is None:
                raise ValueError("")
            c0, c1 = x.shape[2], flow.shape[2]

        new_x = self.cluster(
            x, self_mask=self_mask, self_down_mask=self_down_mask,
            cross_down_mask=cross_down_mask)
        new_x = self.norm0(new_x)

        if self.use_flow:
            x = torch.cat([x, flow], dim=1)
        new_x = torch.cat([x, new_x], dim=1)
        new_x = self.mlp3x3(new_x, size=size)
        new_x = self.norm1(new_x)

        if self.use_layer_scale:
            new_x *= self.layer_scale
        new_x += x
        if self.use_flow:
            new_x, new_flow = new_x.split([c0, c1], dim=2)
        return new_x, new_flow


class LocalCoC(nn.Module):
    def __init__(
        self,
        blocks_counts: Tuple[int, int, int],
        layer_depths: Tuple[int, int, int],
        hidden_depths: Tuple[int, int, int],
        heads_counts: Tuple[int, int, int],
        center_sizes: Tuple[int, int, int],
        fold_sizes: Tuple[int, int, int],
        bias: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: Optional[float] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        layers = []
        for i in range(3):
            layer = nn.ModuleList()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i], hidden_depths[i], heads_counts[i],
                    center_sizes[i], fold_sizes[i], bias=bias,
                    use_layer_scale=use_layer_scale,
                    layer_scale_value=layer_scale_value, dropout=dropout)
                layer.append(block)
            layers.append(layer)
        self.layer0, self.layer1, self.layer2 = layers

        # self.point_reducer0 = nn.Conv2d(
        #     layer_depths[0], layer_depths[1], 3, stride=2, padding=1)
        # self.point_reducer1 = nn.Conv2d(
        #     layer_depths[1], layer_depths[2], 3, stride=2, padding=1)

        # TODO: check FPN design
        # self.layer1_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[0]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False))
        # self.layer0_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[0]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[0], layer_depths[0], 3, padding=1, bias=False))
        # self.layer2_up = nn.Conv2d(
        #     layer_depths[2], layer_depths[2], 1, bias=False)
        # self.layer1_up = nn.Conv2d(
        #     layer_depths[1], layer_depths[2], 1, bias=False)
        # self.layer1_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[2], layer_depths[2], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[2]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[2], layer_depths[1], 3, padding=1, bias=False))
        # self.layer0_up = nn.Conv2d(
        #     layer_depths[0], layer_depths[1], 1, bias=False)
        # self.layer0_out = nn.Sequential(
        #     nn.Conv2d(
        #         layer_depths[1], layer_depths[1], 3, padding=1, bias=False),
        #     nn.BatchNorm2d(layer_depths[1]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(
        #         layer_depths[1], layer_depths[0], 3, padding=1, bias=False))

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = x
        mask0, r0 = mask, mask.shape[2] // x0.shape[3]
        if r0 != 1:
            mask0 = F.max_pool2d(mask0.float(), r0, stride=r0).bool()
        for block in self.layer0:
            x0 = block(x0, mask=mask0)

        # x1 = self.point_reducer0(x0)
        # mask1, r1 = mask, mask.shape[2] // x1.shape[3]
        # if r1 != 1:
        #     mask1 = F.max_pool2d(mask1.float(), r1, stride=r1).bool()
        # for block in self.layer1:
        #     x1 = block(x1, mask=mask1)
        #
        # x2 = self.point_reducer1(x1)
        # mask2, r2 = mask, mask.shape[2] // x2.shape[3]
        # if r2 != 1:
        #     mask2 = F.max_pool2d(mask2.float(), r2, stride=r2).bool()
        # for block in self.layer2:
        #     x2 = block(x2, mask=mask2)
        return x0

        # x1 = x1 + F.interpolate(
        #     x2, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x1 = self.layer1_out(x1)
        # x0 = x0 + F.interpolate(
        #     x1, scale_factor=2.0, mode="bilinear", align_corners=True)
        # x0 = self.layer0_out(x0)
        # return x2, x0
        # new_x2 = self.layer2_up(x2)
        # new_x1 = self.layer1_up(x1)
        # new_x1 += F.interpolate(
        #     new_x2, scale_factor=2.0, mode="bilinear", align_corners=True)
        # new_x1 = self.layer1_out(new_x1)
        # new_x0 = self.layer0_up(x0)
        # new_x0 += F.interpolate(
        #     new_x1, scale_factor=2.0, mode="bilinear", align_corners=True)
        # new_x0 = self.layer0_out(new_x0)
        # return new_x2, new_x0


class MergeBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.mlp = Mlp(2 * depth, 2 * depth, depth, bias=bias, dropout=dropout)
        self.norm = nn.GroupNorm(1, depth)

    def forward(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        size: torch.Size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x = x.transpose(1, 2).unflatten(2, (size[0], size[1]))
        new_center = center.transpose(1, 2).unflatten(
            2, (size[0] // 4, size[1] // 4))
        up_center = F.interpolate(
            new_center, scale_factor=4.0, mode="bilinear", align_corners=True)
        new_x = torch.cat([new_x, up_center], dim=1)

        new_x = self.mlp(new_x)
        new_x = self.norm(new_x)
        new_center = F.interpolate(
            new_x, scale_factor=0.25, mode="bilinear", align_corners=True)
        new_x = new_x.flatten(start_dim=2).transpose(1, 2)
        new_center = new_center.flatten(start_dim=2).transpose(1, 2)
        new_x += x
        new_center += center
        return new_x, new_center


class GlobalCoC(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        types: List[str],
        bias: bool = True,
        use_flow: bool = False,
        flow_depth: Optional[int] = None,
        use_layer_scale: bool = False,
        layer_scale_value: Optional[float] = None,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.types = types
        self.use_flow = use_flow

        if use_flow:
            if flow_depth is None:
                raise ValueError("")
            self.flow_proj = nn.Linear(in_depth, flow_depth)

        # merge_block = MergeBlock(in_depth, bias=bias, dropout=dropout)
        # self.merge_blocks = nn.ModuleList(
        #     [copy.deepcopy(merge_block) for _ in types])
        #
        # global_block = GlobalClusterBlock(
        #     in_depth, hidden_depth, heads_count, bias=bias, use_flow=use_flow,
        #     flow_depth=flow_depth, use_layer_scale=use_layer_scale,
        #     layer_scale_value=layer_scale_value, dropout=dropout)
        # self.global_blocks = nn.ModuleList(
        #     [copy.deepcopy(global_block) for _ in types])

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1, bias=bias,
        #     use_layer_scale=use_layer_scale,
        #     layer_scale_value=layer_scale_value, dropout=dropout)
        # self.local_blocks = nn.ModuleList(
        #     [copy.deepcopy(local_block) for _ in types])

        bi_global_block = BidirectionalGlobalClusterBlock(
            in_depth, hidden_depth, heads_count, bias=bias, use_flow=use_flow,
            flow_depth=flow_depth, use_layer_scale=use_layer_scale,
            layer_scale_value=layer_scale_value, dropout=dropout)
        self.bi_global_blocks = nn.ModuleList(
            [copy.deepcopy(bi_global_block) for _ in types])

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        size0: torch.Size,
        size1: torch.Size,
        pos0: Optional[torch.Tensor] = None,
        pos1: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        flow = None
        if self.use_flow:
            if pos0 is None or pos1 is None:
                raise ValueError("")
            flow0, flow1 = self.flow_proj(pos0), self.flow_proj(pos1)
            flow = torch.cat([flow0, flow1])

        self_mask = self_down_mask = cross_down_mask = None
        if mask0 is not None:
            down_mask0 = F.max_pool2d(mask0.float(), 4, stride=4).bool()
            down_mask1 = F.max_pool2d(mask1.float(), 4, stride=4).bool()
            self_mask = torch.cat([mask0, mask1])
            self_down_mask = torch.cat([down_mask0, down_mask1])
            cross_down_mask = (down_mask0.flatten(start_dim=1)[:, :, None] &
                               down_mask1.flatten(start_dim=1)[:, None, :])
            # mask11 = (mask1.flatten(start_dim=1)[:, :, None] &
            #           mask_1.flatten(start_dim=1)[:, None, :])
            # mask01 = (mask0.flatten(start_dim=1)[:, :, None] &
            #           mask_1.flatten(start_dim=1)[:, None, :])
            # mask10 = (mask1.flatten(start_dim=1)[:, :, None] &
            #           mask_0.flatten(start_dim=1)[:, None, :])

        # for merge_block, global_block, type in zip(
        #     self.merge_blocks, self.global_blocks, self.types):
            # x0, center0 = merge_block(x0, center0, size0)
            # x1, center1 = merge_block(x1, center1, size1)
        x = torch.cat([x0, x1])
        for bi_global_block, type in zip(self.bi_global_blocks, self.types):
            if type == "self":
                # x0 = global_block(x0, center0, mask=mask00)
                # x1 = global_block(x1, center1, mask=mask11)
                pass
            elif type == "cross":
                x, flow = bi_global_block(
                    x, size0, flow=flow, self_mask=self_mask,
                    self_down_mask=self_down_mask, cross_down_mask=cross_down_mask)
                # x0, flow0 = global_block(
                #     x0, center1, size0, flow0=flow0, mask=mask01)
                # x1, flow1 = global_block(
                #     x1, center0, size1, flow0=flow1, mask=mask10)
                # x0 = x0.transpose(1, 2).unflatten(2, (size0[0], size0[1]))
                # x1 = x1.transpose(1, 2).unflatten(2, (size1[0], size1[1]))
                # x0 = local_block(x0, mask=x0_mask)
                # x1 = local_block(x1, mask=x1_mask)
                # x0 = x0.flatten(start_dim=2).transpose(1, 2)
                # x1 = x1.flatten(start_dim=2).transpose(1, 2)
            else:
                raise ValueError("")
        x0, x1 = x.chunk(2)
        flow0 = flow1 = None
        if flow is not None:
            flow0, flow1 = flow.chunk(2)
        return x0, x1, flow0, flow1
