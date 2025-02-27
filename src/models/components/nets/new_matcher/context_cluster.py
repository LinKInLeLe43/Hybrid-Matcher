from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear0 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.linear1 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.linear1(self.gelu(self.linear0(feat)))
        return out


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.conv = nn.Conv2d(hidden_dim, out_dim, 3, padding=1, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.gelu(self.linear(feat)).permute(0, 3, 1, 2)
        out = self.conv(out).permute(0, 2, 3, 1)
        return out


# TODO:
# - Change fold_size anchor_size to fold_size_per_side anchor_size_per_side


class LocalCluster(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        num_heads: int,
        anchor_size: int,
        fold_size: int,
        bias: bool = True,
        type: str = "original",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.anchor_size = anchor_size
        self.fold_size = fold_size
        self.type = type

        self.proj = nn.Linear(feat_dim, 2 * hidden_dim, bias=bias)
        self.anchor_proposal = nn.AdaptiveMaxPool2d(anchor_size)
        self.merge = nn.Linear(hidden_dim, feat_dim, bias=bias)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kwargs = {
            "fc": self.num_heads,
            "fh": self.fold_size,
            "fw": self.fold_size,
            "sh": feat.shape[2] // self.fold_size,
            "sw": feat.shape[3] // self.fold_size,
        }

        feat = self.proj(feat.permute(0, 2, 3, 1))
        feat = rearrange(
            feat,
            "n (fh sh) (fw sw) (fc sc) -> (n fc fh fw) sc sh sw",
            **kwargs,
        )
        anchor = self.anchor_proposal(feat)
        anchor = anchor.flatten(start_dim=-2).transpose(1, 2)
        anchor_sim, anchor_val = anchor.chunk(2, dim=-1)
        feat = feat.flatten(start_dim=-2).transpose(1, 2)
        feat_sim, feat_val = feat.chunk(2, dim=-1)

        feat_sim = F.normalize(feat_sim, dim=-1)
        anchor_sim = F.normalize(anchor_sim, dim=-1)
        sim = torch.einsum("...lc,...sc->...ls", feat_sim, anchor_sim)
        sim = self.alpha * sim + self.beta
        if mask is not None:
            mask = repeat(
                mask, "n (fh sh) (fw sw) -> (n fc fh fw) (sh sw)", **kwargs
            )
            sim.masked_fill_(~mask[..., None], -float("inf"))
        sim = sim.sigmoid()
        max_sim_values, max_sim_idxes = sim.max(dim=2)

        if self.type == "flattened_index":
            range = torch.arange(feat_sim.shape[0], device=feat.device)
            max_sim_idxes = max_sim_idxes + self.fold_size**2 * range[:, None]
            max_sim_values, max_sim_idxes, feat_val, anchor_val = (
                x.flatten(end_dim=1)
                for x in (max_sim_values, max_sim_idxes, feat_val, anchor_val)
            )

            cat_ones = torch.ones_like(feat_val[:, [0]])
            cat_x_value = torch.cat([feat_val, cat_ones], dim=1)
            cat_ones = torch.ones_like(anchor_val[:, [0]])
            cat_anchor_value = torch.cat([anchor_val, cat_ones], dim=1)
            aggregated = cat_anchor_value.index_add_(
                0, max_sim_idxes, max_sim_values[:, None] * cat_x_value
            )
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = max_sim_values[:, None] * aggregated.index_select(
                0, max_sim_idxes
            )
            dispatched = rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fh sh) (fw sw) (fc sc)",
                **kwargs,
            )
        elif self.type == "original":
            mask = torch.zeros_like(sim)
            mask = mask.scatter(2, max_sim_idxes[:, :, None], 1.0)
            sim = (mask * sim)[..., None]

            aggregated = anchor_val + (sim * feat_val[:, :, None]).sum(dim=1)
            aggregated = aggregated / (1 + sim.sum(dim=1))
            dispatched = (sim * aggregated[:, None]).sum(dim=2)
            dispatched = rearrange(
                dispatched,
                "(n fc fh fw) (sh sw) sc -> n (fh sh) (fw sw) (fc sc)",
                **kwargs,
            )
        else:
            raise NotImplementedError("")
        out = self.merge(dispatched)
        return out


class GlobalCluster(nn.Module):
    def __init__(
        self,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        bias: bool = True,
        type: str = "flattened_index",
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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fc = self.heads_count
        n, c, h0, w0 = x0.shape
        _, _, h1, w1 = center1.shape
        m, l, s = n * fc, h0 * w0, h1 * w1
        device = x0.device

        x0_point = self.proj0(x0.permute(0, 2, 3, 1))
        center1 = self.proj1(center1.permute(0, 2, 3, 1))
        x0_point = rearrange(
            x0_point, "n h w (fc sc) -> (n fc) (h w) sc", fc=fc
        )
        center1 = rearrange(center1, "n h w (fc sc) -> (n fc) (h w) sc", fc=fc)
        center1_point, center1_value = center1.chunk(2, dim=2)

        norm_x0_point = F.normalize(x0_point, dim=2)
        norm_center1_point = F.normalize(center1_point, dim=2)
        similarities = torch.einsum(
            "mlc,msc->mls", norm_x0_point, norm_center1_point
        )
        similarities = self.alpha * similarities + self.beta
        if mask is not None:
            mask = repeat(mask, "n l s -> (n fc) l s", fc=fc)
            similarities.masked_fill_(~mask, float("-inf"))
        similarities.sigmoid_()

        if self.type == "flattened_index":
            max_sim_values, max_sim_idxes = similarities.max(dim=2)
            max_sim_idxes = (
                max_sim_idxes + s * torch.arange(m, device=device)[:, None]
            )
            max_sim_values, max_sim_idxes, center1_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, center1_value),
            )

            dispatched = max_sim_values[:, None] * center1_value.index_select(
                0, max_sim_idxes
            )
            dispatched = rearrange(
                dispatched, "(n fc h w) sc -> n h w (fc sc)", fc=fc, h=h0, w=w0
            )
        elif self.type == "torch_scatter":
            csr_idxes = s * torch.arange(l + 1, device=device)[None]
            max_sim_values, max_sim_idxes = torch_scatter.segment_max_csr(
                similarities.flatten(start_dim=1), csr_idxes
            )

            range = torch.arange(m, device=device)[:, None]
            dispatched = (
                max_sim_values[:, :, None]
                * center1_value[range, max_sim_idxes % s]
            )
            dispatched = rearrange(
                dispatched,
                "(n fc) (h w) sc -> n h w (fc sc)",
                fc=fc,
                h=h0,
                w=w0,
            )
        elif self.type == "original":
            max_sim_idxes = similarities.argmax(dim=2)
            mask = torch.zeros_like(similarities)
            mask.scatter_(2, max_sim_idxes[:, :, None], 1.0)
            similarities = (mask * similarities)[..., None]

            dispatched = (similarities * center1_value[:, None, :, :]).sum(
                dim=2
            )
            dispatched = rearrange(
                dispatched,
                "(n fc) (h w) sc -> n h w (fc sc)",
                fc=fc,
                h=h0,
                w=w0,
            )
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
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.cluster = LocalCluster(
            in_depth,
            hidden_depth,
            heads_count,
            center_size,
            fold_size,
            bias=bias,
        )
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp = MLP(2 * in_depth, 2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
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
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.cluster = GlobalCluster(
            in_depth, hidden_depth, heads_count, bias=bias
        )
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = ConvMLP(2 * in_depth, 2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(in_depth)

    def forward(
        self,
        x0: torch.Tensor,
        center1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        initial_depth: int,
        scales: List[int],
        blocks_counts: List[int],
        layer_depths: List[int],
        hidden_depths: List[int],
        heads_counts: List[int],
        center_sizes: List[int],
        fold_sizes: List[int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.scales = scales

        self.point_reducers, self.layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(scales)):
            if scales[i] > 1:
                point_reducer = nn.Conv2d(
                    initial_depth,
                    layer_depths[i],
                    scales[i] + 1,
                    stride=scales[i],
                    padding=1,
                )
            else:
                point_reducer = nn.Identity()
            self.point_reducers.append(point_reducer)

            layer = nn.ModuleList()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i],
                    hidden_depths[i],
                    heads_counts[i],
                    center_sizes[i],
                    fold_sizes[i],
                    bias=bias,
                )
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
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if masks is None:
            masks = len(self.layers) * [None]

        outs = []
        for point_reducer, layer, mask in zip(
            self.point_reducers, self.layers, masks
        ):
            x = point_reducer(x)
            for block in layer:
                x = block(x, mask=mask)
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
    def __init__(self, scale: int, depth: int, bias: bool = True) -> None:
        super().__init__()
        self.scale = scale

        self.mlp = MLP(2 * depth, 2 * depth, depth, bias=bias)
        self.norm = nn.LayerNorm(depth)
        self.pooling = nn.MaxPool2d(scale, stride=scale)

    def forward(
        self, x: torch.Tensor, center: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up_center = F.interpolate(
            center, scale_factor=self.scale, mode="bilinear"
        )
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
        bias: bool = True,
    ) -> None:
        super().__init__()

        merge_block = MergeBlock(scale, in_depth, bias=bias)
        self.merge_blocks = nn.ModuleList(
            [deepcopy(merge_block) for _ in range(layer_count)]
        )

        global_block = GlobalClusterBlock(
            in_depth, hidden_depth, heads_count, bias=bias
        )
        self.global_blocks = nn.ModuleList(
            [deepcopy(global_block) for _ in range(layer_count)]
        )

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1, bias=bias)
        # self.local_blocks = nn.ModuleList(
        #     [copy.deepcopy(local_block) for _ in types])

        self.self_blocks = nn.ModuleList(
            [deepcopy(attention_block) for _ in range(layer_count)]
        )
        self.cross_blocks = nn.ModuleList(
            [deepcopy(attention_block) for _ in range(layer_count)]
        )

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
        y1_mask: Optional[torch.Tensor] = None,
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
            self.merge_blocks,
            self.global_blocks,
            self.self_blocks,
            self.cross_blocks,
        ):
            x0, y0 = merge_block(x0, y0)
            x1, y1 = merge_block(x1, y1)
            # x0 = global_block(x0, center0, mask=mask00)
            # x1 = global_block(x1, center1, mask=mask11)
            x0 = global_block(x0, y1, mask=mask01)
            x1 = global_block(x1, y0, mask=mask10)
            x0 = self_block(x0, x0, rope=rope, mask0=y0_mask, mask1=y0_mask)
            x1 = self_block(x1, x1, rope=rope, mask0=y1_mask, mask1=y1_mask)
            x0 = cross_block(x0, x1, mask0=y0_mask, mask1=y1_mask)
            x1 = cross_block(x1, x0, mask0=y1_mask, mask1=y0_mask)
        return x0, x1
