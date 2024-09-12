import copy
from typing import List, Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter


class Mlp(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.linear1 = nn.Linear(hidden_depth, out_depth, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear0(x)
        x = self.gelu(x)
        x = self.linear1(x)
        return x


class Mlp3x3(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        out_depth: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_depth, hidden_depth, bias=bias)
        self.conv = nn.Conv2d(hidden_depth, out_depth, 3, padding=1, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.gelu(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
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
        type: str = "original"
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.center_size = center_size
        self.fold_size = fold_size
        self.type = type

        self.proj = nn.Linear(in_depth, 2 * hidden_depth, bias=bias)
        self.center_proposal = nn.AdaptiveAvgPool2d(center_size)
        self.merge = nn.Linear(hidden_depth, in_depth, bias=bias)

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

        x = self.proj(x.permute(0, 2, 3, 1))
        x = einops.rearrange(
            x, "n (fh sh) (fw sw) (fc sc) -> (n fc fh fw) sc sh sw", fc=fc,
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

        if self.type == "flattened_index":
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
                "(n fc fh fw sh sw) sc -> n (fh sh) (fw sw) (fc sc)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        elif self.type == "torch_scatter":
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, x_value, center_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, x_value, center_value))
            _max_sim_idxes, sorted_idxes = max_sim_idxes.sort()

            cat_ones = torch.ones_like(x_value[:, [0]])
            cat_x_value = torch.cat([x_value, cat_ones], dim=1)
            cat_ones = torch.ones_like(center_value[:, [0]])
            cat_center_value = torch.cat([center_value, cat_ones], dim=1)
            max_sim_idxes_csr = torch._convert_indices_from_coo_to_csr(
                _max_sim_idxes, size=m * s)
            aggregated = cat_center_value + torch_scatter.segment_csr(
                (max_sim_values[:, None] * cat_x_value)[sorted_idxes],
                max_sim_idxes_csr)
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = (max_sim_values[:, None] *
                          aggregated.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fh sh) (fw sw) (fc sc)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        elif self.type == "original":
            mask = torch.zeros_like(similarities)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarities = (mask * similarities)[..., None]

            aggregated = (center_value +
                          (similarities * x_value[:, :, None, :]).sum(dim=1))
            aggregated /= 1 + similarities.sum(dim=1)
            dispatched = (similarities * aggregated[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw) (sh sw) sc -> n (fh sh) (fw sw) (fc sc)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        else:
            raise NotImplementedError("")
        dispatched = self.merge(dispatched)
        return dispatched


class GlobalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True,
        type: str = "flattened_index"
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.type = type

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
        n, c, h0, w0 = x0.shape
        _, _, h1, w1 = center1.shape
        m, l, s = n * fc, h0 * w0, h1 * w1
        device = x0.device

        x0_point = self.proj0(x0.permute(0, 2, 3, 1))
        center1 = self.proj1(center1.permute(0, 2, 3, 1))
        x0_point = einops.rearrange(
            x0_point, "n h w (fc sc) -> (n fc) (h w) sc", fc=fc)
        center1 = einops.rearrange(
            center1, "n h w (fc sc) -> (n fc) (h w) sc", fc=fc)
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

        if self.type == "flattened_index":
            max_sim_values, max_sim_idxes = similarities.max(dim=2)
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, center1_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, center1_value))

            dispatched = (max_sim_values[:, None] *
                          center1_value.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched, "(n fc h w) sc -> n h w (fc sc)", fc=fc, h=h0, w=w0)
        elif self.type == "torch_scatter":
            csr_idxes = s * torch.arange(l + 1, device=device)[None]
            max_sim_values, max_sim_idxes = torch_scatter.segment_max_csr(
                similarities.flatten(start_dim=1), csr_idxes)

            range = torch.arange(m, device=device)[:, None]
            dispatched = (max_sim_values[:, :, None] *
                          center1_value[range, max_sim_idxes % s])
            dispatched = einops.rearrange(
                dispatched, "(n fc) (h w) sc -> n h w (fc sc)", fc=fc, h=h0,
                w=w0)
        elif self.type == "original":
            max_sim_idxes = similarities.argmax(dim=2)
            mask = torch.zeros_like(similarities)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarities = (mask * similarities)[..., None]

            dispatched = (similarities *
                          center1_value[:, None, :, :]).sum(dim=2)
            dispatched = einops.rearrange(
                dispatched, "(n fc) (h w) sc -> n h w (fc sc)", fc=fc, h=h0,
                w=w0)
        else:
            raise NotImplementedError("")
        dispatched = self.merge(dispatched)
        return dispatched


class LocalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        center_size: int,
        fold_size: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.cluster = LocalCluster(
            in_depth, hidden_depth, heads_count, center_size, fold_size,
            bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = Mlp(2 * in_depth, 2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        new_x = self.cluster(x, mask=mask)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x.permute(0, 2, 3, 1), new_x], dim=3)
        new_x = self.mlp(new_x)
        new_x = self.norm1(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()

        new_x += x
        return new_x


class GlobalClusterBlock(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.cluster = GlobalCluster(
            in_depth, hidden_depth, heads_count, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = Mlp3x3(2 * in_depth, 2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self,
        x0: torch.Tensor,
        center1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        new_x0 = self.cluster(x0, center1, mask=mask)
        new_x0 = self.norm0(new_x0)

        new_x0 = torch.cat([x0.permute(0, 2, 3, 1), new_x0], dim=3)
        new_x0 = self.mlp3x3(new_x0)
        new_x0 = self.norm1(new_x0)
        new_x0 = new_x0.permute(0, 3, 1, 2).contiguous()

        new_x0 += x0
        return new_x0


class LocalCoC(nn.Module):
    def __init__(
        self,
        scales: List[int],
        blocks_counts: List[int],
        layer_depths: List[int],
        hidden_depths: List[int],
        heads_counts: List[int],
        center_sizes: List[int],
        fold_sizes: List[int],
        bias: bool = True
    ) -> None:
        super().__init__()
        self.scales = scales

        initial_depth = layer_depths[0]
        self.point_reducers, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(scales)):
            if scales[i] > 1:
                point_reducer = nn.Conv2d(
                    initial_depth, layer_depths[i], scales[i] + 1,
                    stride=scales[i], padding=1)
            else:
                point_reducer = nn.Identity()
            self.point_reducers.append(point_reducer)

            layer = nn.Sequential()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i], hidden_depths[i], heads_counts[i],
                    center_sizes[i], fold_sizes[i], bias=bias)
                layer.append(block)
            self.layers.append(layer)

            initial_depth = layer_depths[i]

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
        outs = []
        for point_reducer, layer in zip(self.point_reducers, self.layers):
            x = point_reducer(x)
            x = layer(x)
            outs.append(x)
        return outs[0], outs[-1]

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
        bias: bool = True
    ) -> None:
        super().__init__()
        self.scale = scale

        self.mlp = Mlp(2 * depth, 2 * depth, depth, bias=bias)
        self.norm = nn.LayerNorm(depth)
        self.pooling = nn.MaxPool2d(scale, stride=scale)

    def forward(
        self,
        x: torch.Tensor,
        center: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up_center = F.interpolate(
            center, scale_factor=self.scale, mode="bilinear")
        new_x = torch.cat([x, up_center], dim=1)
        new_x = new_x.permute(0, 2, 3, 1)
        new_x = self.mlp(new_x)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()
        new_center = self.pooling(new_x)
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
        layer_count: int,
        attention_block: nn.Module,
        bias: bool = True
    ) -> None:
        super().__init__()

        merge_block = MergeBlock(scale, in_depth, bias=bias)
        self.merge_blocks = nn.ModuleList(
            [copy.deepcopy(merge_block) for _ in range(layer_count)])

        global_block = GlobalClusterBlock(
            in_depth, hidden_depth, heads_count, bias=bias)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in range(layer_count)])

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1, bias=bias)
        # self.local_blocks = nn.ModuleList(
        #     [copy.deepcopy(local_block) for _ in types])

        self.self_blocks = nn.ModuleList([copy.deepcopy(attention_block)
                                          for _ in range(layer_count)])
        self.cross_blocks = nn.ModuleList([copy.deepcopy(attention_block)
                                           for _ in range(layer_count)])

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
        y0: torch.Tensor,
        y1: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        x0_mask: Optional[torch.Tensor] = None,
        x1_mask: Optional[torch.Tensor] = None,
        y0_mask: Optional[torch.Tensor] = None,
        y1_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask00 = mask11 = mask01 = mask10 = None
        if x0_mask is not None:
            x0_mask = x0_mask.flatten(start_dim=1)
            x1_mask = x1_mask.flatten(start_dim=1)
            y0_mask = y0_mask.flatten(start_dim=1)
            y1_mask = y1_mask.flatten(start_dim=1)
            mask00 = x0_mask[:, :, None] & y0_mask[:, None, :]
            mask11 = x1_mask[:, :, None] & y1_mask[:, None, :]
            mask01 = x0_mask[:, :, None] & y1_mask[:, None, :]
            mask10 = x1_mask[:, :, None] & y0_mask[:, None, :]

        for merge_block, global_block, self_block, cross_block in zip(
            self.merge_blocks, self.global_blocks, self.self_blocks, self.cross_blocks):
            x0, y0 = merge_block(x0, y0)
            x1, y1 = merge_block(x1, y1)
            # x0 = global_block(x0, center0, mask=mask00)
            # x1 = global_block(x1, center1, mask=mask11)
            x0 = global_block(x0, y1, mask=mask01)
            x1 = global_block(x1, y0, mask=mask10)
            x0 = self_block(
                x0, x0, rope=rope, x_mask=y0_mask, source_mask=y0_mask)
            x1 = self_block(
                x1, x1, rope=rope, x_mask=y1_mask, source_mask=y1_mask)
            x0 = cross_block(x0, x1, x_mask=y0_mask, source_mask=y1_mask)
            x1 = cross_block(x1, x0, x_mask=y1_mask, source_mask=y0_mask)
        return x0, x1
