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

        merge_block = MergeBlock(in_depth, bias=bias, dropout=dropout)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in types])

        global_block = GlobalClusterBlock(
            in_depth, hidden_depth, heads_count, bias=bias, use_flow=use_flow,
            flow_depth=flow_depth, use_layer_scale=use_layer_scale,
            layer_scale_value=layer_scale_value, dropout=dropout)
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
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        flow0 = flow1 = None
        if self.use_flow:
            if pos0 is None or pos1 is None:
                raise ValueError("")
            flow0, flow1 = self.flow_proj(pos0), self.flow_proj(pos1)

        mask00 = mask11 = mask01 = mask10 = None
        if mask0 is not None:
            r = int((x0.shape[1] // center0.shape[1]) ** 0.5)
            mask_0 = F.max_pool2d(mask0.float(), r, stride=r).bool()
            mask_1 = F.max_pool2d(mask1.float(), r, stride=r).bool()
            mask0_, mask1_, mask_0, mask_1 = map(
                lambda x: x.flatten(start_dim=1),
                (mask0, mask1, mask_0, mask_1))
            mask00 = mask0_[:, :, None] & mask_0[:, None, :]
            mask11 = mask1_[:, :, None] & mask_1[:, None, :]
            mask01 = mask0_[:, :, None] & mask_1[:, None, :]
            mask10 = mask1_[:, :, None] & mask_0[:, None, :]

        for merge_block, global_block, type in zip(
            self.merge_blocks, self.global_blocks, self.types):
            x0, center0 = merge_block(x0, center0, size0)
            x1, center1 = merge_block(x1, center1, size1)
            if type == "self":
                # x0 = global_block(x0, center0, mask=mask00)
                # x1 = global_block(x1, center1, mask=mask11)
                pass
            elif type == "cross":
                x0, flow0 = global_block(
                    x0, center1, size0, flow0=flow0, mask=mask01)
                x1, flow1 = global_block(
                    x1, center0, size1, flow0=flow1, mask=mask10)
                # x0 = x0.transpose(1, 2).unflatten(2, (size0[0], size0[1]))
                # x1 = x1.transpose(1, 2).unflatten(2, (size1[0], size1[1]))
                # x0 = local_block(x0, mask=x0_mask)
                # x1 = local_block(x1, mask=x1_mask)
                # x0 = x0.flatten(start_dim=2).transpose(1, 2)
                # x1 = x1.flatten(start_dim=2).transpose(1, 2)
            else:
                raise ValueError("")
        return x0, x1, flow0, flow1
