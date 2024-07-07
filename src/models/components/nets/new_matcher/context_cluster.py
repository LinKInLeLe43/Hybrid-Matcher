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
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth)
        self.linear1 = nn.Linear(hidden_depth, out_depth)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)

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
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_depth, hidden_depth)
        self.conv = nn.Conv2d(hidden_depth, out_depth, 3, padding=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Size] = None
    ) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)

        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)

        flatten = False
        if len(x.shape) == 3:
            if size is None:
                raise ValueError("")
            flatten = True
            x = x.transpose(1, 2).unflatten(2, (size[0], size[1]))
        else:
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
        fold_size: int
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.center_size = center_size
        self.fold_size = fold_size

        self.proj = nn.Conv2d(in_depth, 2 * hidden_depth, 1)
        self.center_proposal = nn.AdaptiveAvgPool2d(center_size)
        self.merge = nn.Conv2d(hidden_depth, in_depth, 1)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = x.device
        n = x.shape[0]
        fc, fh, fw = self.heads_count, self.fold_size, self.fold_size
        m, s = n * fc * fh * fw, self.center_size * self.center_size
        w = x.shape[3] // fw

        x = self.proj(x)
        x = einops.rearrange(
            x, "n (fc c) (fh h) (fw w) -> (n fc fh fw) c h w", fc=fc, fh=fh,
            fw=fw)
        center = self.center_proposal(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        center = center.flatten(start_dim=2).transpose(1, 2)
        x_point, x_value = x.chunk(2, dim=2)
        center_point, center_value = center.chunk(2, dim=2)

        norm_x_point = F.normalize(x_point, dim=2)
        norm_center_point = F.normalize(center_point, dim=2)
        similarities = torch.einsum(
            "nlc,nsc->nls", norm_x_point, norm_center_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.repeat(
                mask, "n l -> (n fc fh fw) l s", fc=fc, fh=fh, fw=fw, s=s)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()

        # max_sim_idxes = similarities.argmax(dim=2, keepdim=True)
        # mask = torch.zeros_like(similarities).scatter_(2, max_sim_idxes, 1.0)
        # similarities = (mask * similarities)[..., None]
        #
        # new_center = (center_value +
        #               (similarities * x_value[:, :, None, :]).sum(dim=1))
        # new_center /= (1 + similarities.sum(dim=1))
        # new_x = (similarities * new_center[:, None, :, :]).sum(dim=2)
        # new_x = einops.rearrange(
        #     new_x, "(n fc fh fw) (h w) c -> n (fc c) (fh h) (fw w)", fc=fc,
        #     fh=fh, fw=fw, w=w)
        # new_x = self.merge(new_x)
        max_sim_values, max_sim_idxes = similarities.max(dim=2)
        max_sim_idxes = (max_sim_idxes +
                         s * torch.arange(m, device=device)[:, None])
        max_sim_values, max_sim_idxes, x_value, center_value = map(
            lambda x: x.flatten(end_dim=1),
            (max_sim_values, max_sim_idxes, x_value, center_value))

        center_ones = torch.ones((center_value.shape[0], 1), device=device)
        center_value = torch.cat([center_value, center_ones], dim=1)
        x_ones = torch.ones((x_value.shape[0], 1), device=device)
        x_value = torch.cat([x_value, x_ones], dim=1)
        new_center = center_value.index_add_(
            0, max_sim_idxes, max_sim_values[:, None] * x_value)
        new_center = new_center[:, :-1] / new_center[:, -1:]
        new_x = (max_sim_values[:, None] *
                 new_center.index_select(0, max_sim_idxes))
        new_x = einops.rearrange(
            new_x, "(n fc fh fw h w) c -> n (fc c) (fh h) (fw w)", n=n, fc=fc,
            fh=fh, fw=fw, w=w)
        new_x = self.merge(new_x)
        return new_x


class GlobalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int
    ) -> None:
        super().__init__()
        self.heads_count = heads_count

        self.proj0 = nn.Linear(in_depth, hidden_depth)
        self.proj1 = nn.Linear(in_depth, 2 * hidden_depth)
        self.merge = nn.Linear(hidden_depth, in_depth)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = x0.device
        n, s, _ = x1.shape
        fc = self.heads_count
        m = n * fc

        x0_point, x1 = self.proj0(x0), self.proj1(x1)
        x0_point = einops.rearrange(x0_point, "n l (fc c) -> (n fc) l c", fc=fc)
        x1 = einops.rearrange(x1, "n s (fc c) -> (n fc) s c", fc=fc)
        x1_point, x1_value = x1.chunk(2, dim=2)

        norm_x0_point = F.normalize(x0_point, dim=2)
        norm_x1_point = F.normalize(x1_point, dim=2)
        similarities = torch.einsum(
            "nlc,nsc->nls", norm_x0_point, norm_x1_point)
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = einops.repeat(mask, "n l s -> (n fc) l s", fc=fc)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()

        max_sim_values, max_sim_idxes = similarities.max(dim=2)
        max_sim_idxes = (max_sim_idxes +
                         s * torch.arange(m, device=device)[:, None])
        max_sim_values, max_sim_idxes, x1_value = map(
            lambda x: x.flatten(end_dim=1),
            (max_sim_values, max_sim_idxes, x1_value))

        new_x0 = (max_sim_values[:, None] *
                  x1_value.index_select(0, max_sim_idxes))
        new_x0 = einops.rearrange(
            new_x0, "(n fc l) c -> n l (fc c)", n=n, fc=fc)
        new_x0 = self.merge(new_x0)
        return new_x0


class LocalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        center_size: int,
        fold_size: int,
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
            in_depth, hidden_depth, heads_count, center_size, fold_size)
        self.norm0 = nn.GroupNorm(1, in_depth)

        self.mlp = Mlp(2 * in_depth, 2 * in_depth, in_depth, dropout=dropout)
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

        self.cluster = GlobalCluster(in_depth, hidden_depth, heads_count)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = Mlp3x3(
            in_depth + out_depth, 2 * in_depth, out_depth, dropout=dropout)
        self.norm1 = nn.LayerNorm(out_depth)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        size0: torch.Size,
        flow0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_flow0 = None
        if self.use_flow:
            if flow0 is None:
                raise ValueError("")
            c0, c1 = x0.shape[2], flow0.shape[2]

        new_x0 = self.cluster(x0, x1, mask=mask)
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
                    center_sizes[i], fold_sizes[i],
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
    def __init__(self, depth: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.mlp = Mlp(2 * depth, 2 * depth, depth, dropout=dropout)
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

        merge_block = MergeBlock(in_depth, dropout=dropout)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in types])

        global_block = GlobalClusterBlock(
            in_depth, hidden_depth, heads_count, use_flow=use_flow,
            flow_depth=flow_depth, use_layer_scale=use_layer_scale,
            layer_scale_value=layer_scale_value, dropout=dropout)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in types])

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1,
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

        mask00 = mask11 = mask01 = mask10 = None
        if x0_mask is not None:
            mask00 = x0_mask[:, :, None] & center0_mask[:, None, :]
            mask11 = x1_mask[:, :, None] & center1_mask[:, None, :]
            mask01 = x0_mask[:, :, None] & center1_mask[:, None, :]
            mask10 = x1_mask[:, :, None] & center0_mask[:, None, :]

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
