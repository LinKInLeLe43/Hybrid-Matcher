import copy
from typing import List, Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .modules.attention import Attention
from .positional_encoding import RoPESinePositionalEncoding

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
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.dropout(x)
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
        m, s = n * fc * fh * fw, self.center_size ** 2
        device = x.device

        x = self.proj(x)
        x = einops.rearrange(
            x, "n (fc sc) (fh sh) (fw sw) -> (n fc fh fw) sc sh sw", fc=fc,
            fh=fh, fw=fw)
        center = self.center_proposal(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        center = center.flatten(start_dim=2).transpose(1, 2)
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

        self.point_proj = nn.Linear(in_depth, 2 * hidden_depth, bias=bias)
        self.anchor_proj = nn.Linear(in_depth, 2 * hidden_depth, bias=bias)
        self.point_merge = nn.Linear(hidden_depth, in_depth, bias=bias)
        self.anchor_merge = nn.Linear(hidden_depth, in_depth, bias=bias)

        self.attention = Attention()

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        point0: torch.Tensor,
        point1: torch.Tensor,
        anchor0: torch.Tensor,
        anchor1: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fc = self.heads_count
        n, l, c = anchor0.shape
        _, s, _ = anchor1.shape
        m = n * fc
        device = anchor0.device

        anchor = torch.cat([anchor0, anchor1], dim=1)
        anchor = self.anchor_proj(anchor)
        anchor = einops.rearrange(anchor, "n k (fc sc) -> (n fc) k sc", fc=fc)
        anchor0, anchor1 = anchor.split([l, s], dim=1)
        anchor0_sim, anchor0_value = anchor0.chunk(2, dim=2)
        anchor1_sim, anchor1_value = anchor1.chunk(2, dim=2)

        norm_anchor0_sim = F.normalize(anchor0_sim, dim=2)
        norm_anchor1_sim = F.normalize(anchor1_sim, dim=2)
        similarity = torch.einsum(
            "mlc,msc->mls", norm_anchor0_sim, norm_anchor1_sim)
        similarity = self.alpha * similarity + self.beta
        if anchor_mask is not None:
            anchor_mask = einops.repeat(
                anchor_mask, "n l s -> (n fc) l s", fc=fc)
            similarity.masked_fill_(~anchor_mask, float("-inf"))
        similarity.sigmoid_()
        max_sim_value1_to_0, max_sim_idx1_to_0 = similarity.max(dim=2)
        max_sim_value0_to_1, max_sim_idx0_to_1 = similarity.max(dim=1)

        if self.use_efficient:
            max_sim_value = torch.cat([max_sim_value1_to_0,
                                       max_sim_value0_to_1])
            range = torch.arange(m, device=device)[:, None]
            max_sim_idx = torch.cat([max_sim_idx1_to_0 + s * range,
                                     max_sim_idx0_to_1 + s * m + l * range])
            anchor_value = torch.cat([anchor1_value, anchor0_value])
            max_sim_value, max_sim_idx, anchor_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_value, max_sim_idx, anchor_value))

            anchor_message = (max_sim_value[:, None] *
                              anchor_value.index_select(0, max_sim_idx))
            anchor_message = einops.rearrange(
                anchor_message, "(n fc k) sc -> n k (fc sc)", n=n, fc=fc)
            anchor_message = self.anchor_merge(anchor_message)

            point = torch.cat([point0, point1], dim=1)
            point = self.point_proj(point)
            point = einops.rearrange(
                point, "n k ww (fc sc) -> (n fc k) ww sc", fc=fc)
            point_q, _ = point.chunk(2, dim=2)
            point_k, point_v = (torch.cat([point[m * l:], point[:m * l]])
                                .index_select(0, max_sim_idx).chunk(2, dim=2))
            point_message = (max_sim_value[:, None, None] *
                             self.attention(point_q, point_k, point_v))
            point_message = einops.rearrange(
                point_message, "(n fc k) ww sc -> n k ww (fc sc)", n=n, fc=fc)
            point_message = self.point_merge(point_message)
        else:
            raise NotImplementedError("")
        anchor_message1_to_0, anchor_message0_to_1 = anchor_message.split(
            [l, s], dim=1)
        point_message1_to_0, point_message0_to_1 = point_message.split(
            [l, s], dim=1)
        return (point_message1_to_0, point_message0_to_1,
                anchor_message1_to_0, anchor_message0_to_1)


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
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = Mlp(
            2 * in_depth, 2 * in_depth, in_depth, bias=bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        new_x = self.cluster(x, mask=mask)
        new_x = new_x.permute(0, 2, 3, 1)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x.permute(0, 2, 3, 1), new_x], dim=3)
        new_x = self.mlp(new_x)
        new_x = self.norm1(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()

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
        use_flow: bool = False,
        flow_depth: Optional[int] = None,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.use_flow = use_flow

        out_depth = in_depth
        if use_flow:
            if flow_depth is None:
                raise ValueError("")
            out_depth += flow_depth

        self.cluster = GlobalCluster(
            in_depth, hidden_depth, heads_count, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = Mlp(
            in_depth + out_depth, in_depth + out_depth, out_depth, bias=bias)
        self.norm1 = nn.LayerNorm(out_depth)

    def forward(
        self,
        point0: torch.Tensor,
        point1: torch.Tensor,
        anchor0: torch.Tensor,
        anchor1: torch.Tensor,
        size0: torch.Size,
        size1: torch.Size,
        flow0: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_flow0 = None
        if self.use_flow:
            if flow0 is None:
                raise ValueError("")
            c0, c1 = point0.shape[2], flow0.shape[2]

        f_point0 = einops.rearrange(
            point0, "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c", sh=4, sw=4)
        f_point1 = einops.rearrange(
            point1, "n c (fh sh) (fw sw) -> n (fh fw) (sh sw) c", sh=4, sw=4)
        f_anchor0 = anchor0.flatten(start_dim=2).transpose(1, 2)
        f_anchor1 = anchor1.flatten(start_dim=2).transpose(1, 2)
        new_point0, new_point1, new_anchor0, new_anchor1 = self.cluster(
            f_point0, f_point1, f_anchor0, f_anchor1, anchor_mask=anchor_mask)
        new_point0 = einops.rearrange(
            new_point0, "n (fh fw) (sh sw) c -> n c (fh sh) (fw sw)", sh=4, fh=size0[0] // 4)
        new_point1 = einops.rearrange(
            new_point1, "n (fh fw) (sh sw) c -> n c (fh sh) (fw sw)", sh=4, fh=size1[0] // 4)
        new_anchor0 = new_anchor0.transpose(1, 2).unflatten(2, (size0[0] // 4, size0[1] // 4))
        new_anchor1 = new_anchor1.transpose(1, 2).unflatten(2, (size1[0] // 4, size1[1] // 4))
        new_anchor0 = F.interpolate(new_anchor0, scale_factor=4.0, mode="bilinear")
        new_anchor1 = F.interpolate(new_anchor1, scale_factor=4.0, mode="bilinear")
        message0 = new_point0 + new_anchor0
        message1 = new_point1 + new_anchor1
        message0 = message0.permute(0, 2, 3, 1)
        message1 = message1.permute(0, 2, 3, 1)
        message0 = self.norm0(message0)
        message1 = self.norm0(message1)

        if self.use_flow:
            point0 = torch.cat([point0, flow0], dim=2)
        message0 = torch.cat([point0.permute(0, 2, 3, 1), message0], dim=3)
        message1 = torch.cat([point1.permute(0, 2, 3, 1), message1], dim=3)
        message0 = self.mlp(message0)
        message1 = self.mlp(message1)
        message0 = self.norm1(message0)
        message1 = self.norm1(message1)
        message0 = message0.permute(0, 3, 1, 2).contiguous()
        message1 = message1.permute(0, 3, 1, 2).contiguous()

        message0 += point0
        message1 += point1
        if self.use_flow:
            new_point0, new_flow0 = new_point0.split([c0, c1], dim=2)
        return message0, message1


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
            layer = nn.Sequential()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i], hidden_depths[i], heads_counts[i],
                    center_sizes[i], fold_sizes[i], bias=bias,
                    use_layer_scale=use_layer_scale,
                    layer_scale_value=layer_scale_value, dropout=dropout)
                layer.append(block)
            layers.append(layer)
        self.layer0, self.layer1, self.layer2 = layers

        self.point_reducer0 = nn.Conv2d(
            layer_depths[0], layer_depths[1], 3, stride=2, padding=1)
        self.point_reducer1 = nn.Conv2d(
            layer_depths[1], layer_depths[2], 3, stride=2, padding=1)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.layer0(x)
        x1 = self.point_reducer0(x0)
        x1 = self.layer1(x1)
        x2 = self.point_reducer1(x1)
        x2 = self.layer2(x2)
        return x2, x0

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
        scale: int,
        depth: int,
        head_count: int,
        bias: bool = True,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head_count = head_count

        self.pooling = nn.Conv2d(
            depth, depth, scale, stride=scale, groups=depth, bias=False)
        self.pe = RoPESinePositionalEncoding(depth)
        self.proj = nn.Linear(depth, 3 * depth, bias=False)
        self.attention = Attention()
        self.merge = nn.Linear(depth, depth, bias=False)
        self.norm0 = nn.LayerNorm(depth)

        self.mlp = Mlp(
            3 * depth, 3 * depth, 2 * depth, bias=bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(2 * depth)

    def forward(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        size: torch.Size,
        center_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fc = self.head_count

        down_x = self.pooling(x).permute(0, 2, 3, 1)
        q, k, v = self.proj(down_x).chunk(3, dim=3)
        q, k = self.pe(q), self.pe(k)
        q, k, v = map(
            lambda x: einops.rearrange(x, "n h w (fc sc) -> n fc (h w) sc", fc=fc),
            (q, k, v))
        mask = None
        if center_mask is not None:
            mask = center_mask.flatten(start_dim=1)
        message = self.attention(q, k, v, q_mask=mask, kv_mask=mask)
        message = einops.rearrange(
            message, "n fc (h w) sc -> n h w (fc sc)", h=size[0] // self.scale)
        message = self.merge(message)
        message = self.norm0(message)

        message = torch.cat([down_x, center.permute(0, 2, 3, 1), message], dim=3)
        message = self.mlp(message)
        message = self.norm1(message)
        new_x, new_center = message.permute(0, 3, 1, 2).chunk(2, dim=1)
        new_x = F.interpolate(
            new_x, scale_factor=self.scale, mode="bilinear")
        new_x = x + new_x
        new_center = center + new_center
        return new_x, new_center


class GlobalCoC(nn.Module):
    def __init__(
        self,
        scale: int,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        types: List[str],
        use_flow: bool = False,
        flow_depth: Optional[int] = None,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.types = types
        self.use_flow = use_flow

        if use_flow:
            if flow_depth is None:
                raise ValueError("")
            self.flow_proj = nn.Linear(in_depth, flow_depth)

        merge_block = MergeBlock(scale, in_depth, heads_count, bias=bias)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in types])

        global_block = GlobalClusterBlock(
            in_depth, hidden_depth, heads_count, use_flow=use_flow,
            flow_depth=flow_depth, bias=bias)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in types])

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1, bias=bias,
        #     use_layer_scale=use_layer_scale,
        #     layer_scale_value=layer_scale_value, dropout=dropout)
        # self.local_blocks = nn.ModuleList(
        #     [copy.deepcopy(local_block) for _ in types])

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
        center0: torch.Tensor,
        center1: torch.Tensor,
        size0: torch.Size,
        size1: torch.Size,
        pos0: Optional[torch.Tensor] = None,
        pos1: Optional[torch.Tensor] = None,
        x0_mask: Optional[torch.Tensor] = None,
        x1_mask: Optional[torch.Tensor] = None,
        center0_mask: Optional[torch.Tensor] = None,
        center1_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        flow0 = flow1 = None
        if self.use_flow:
            if pos0 is None or pos1 is None:
                raise ValueError("")
            flow0, flow1 = self.flow_proj(pos0), self.flow_proj(pos1)

        mask00 = mask11 = mask01 = mask10 = mask = None
        if x0_mask is not None:
            mask00 = (x0_mask.flatten(start_dim=1)[:, :, None] &
                      center0_mask.flatten(start_dim=1)[:, None, :])
            mask11 = (x1_mask.flatten(start_dim=1)[:, :, None] &
                      center1_mask.flatten(start_dim=1)[:, None, :])
            mask01 = (x0_mask.flatten(start_dim=1)[:, :, None] &
                      center1_mask.flatten(start_dim=1)[:, None, :])
            mask10 = (x1_mask.flatten(start_dim=1)[:, :, None] &
                      center0_mask.flatten(start_dim=1)[:, None, :])
            mask = (center0_mask.flatten(start_dim=1)[:, :, None] &
                    center1_mask.flatten(start_dim=1)[:, None, :])

        for merge_block, global_block, type in zip(
            self.merge_blocks, self.global_blocks, self.types):
            x0, center0 = merge_block(x0, center0, size0, center_mask=center0_mask)
            x1, center1 = merge_block(x1, center1, size1, center_mask=center1_mask)
            if type == "self":
                # x0 = global_block(x0, center0, mask=mask00)
                # x1 = global_block(x1, center1, mask=mask11)
                pass
            elif type == "cross":
                x0, x1 = global_block(
                    x0, x1, center0, center1, size0, size1, flow0=flow0,
                    anchor_mask=mask)
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
        return x0, x1, flow0, flow1
