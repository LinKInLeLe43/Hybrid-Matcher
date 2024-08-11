import copy
from typing import Optional, Tuple

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
        type: str = "flattened_index"
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.center_size = center_size
        self.fold_size = fold_size
        self.type = type

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
                "(n fc fh fw sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
                fh=fh, fw=fw, sh=sh, sw=sw)
        elif self.type == "segment_csr":
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, x_value, center_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, x_value, center_value))

            max_sim_idxes, sorted_idxes = max_sim_idxes.sort()

            cat_ones = torch.ones_like(x_value[:, [0]])
            cat_x_value = torch.cat([x_value, cat_ones], dim=1)
            cat_ones = torch.ones_like(center_value[:, [0]])
            cat_center_value = torch.cat([center_value, cat_ones], dim=1)
            max_sim_idxes_csr = torch._convert_indices_from_coo_to_csr(
                max_sim_idxes, size=m * s)
            aggregated = cat_center_value + torch_scatter.segment_csr(
                (max_sim_values[:, None] * cat_x_value)[sorted_idxes],
                max_sim_idxes_csr)
            aggregated = aggregated[:, :-1] / aggregated[:, -1:]
            dispatched = (max_sim_values[:, None] *
                          aggregated.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched,
                "(n fc fh fw sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
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
                "(n fc fh fw) (sh sw) sc -> n (fc sc) (fh sh) (fw sw)", fc=fc,
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
        m, s = n * fc, h1 * w1
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
        max_sim_values, max_sim_idxes = similarities.max(dim=2)

        if self.type == "flattened_index":
            max_sim_idxes = (max_sim_idxes +
                             s * torch.arange(m, device=device)[:, None])
            max_sim_values, max_sim_idxes, center1_value = map(
                lambda x: x.flatten(end_dim=1),
                (max_sim_values, max_sim_idxes, center1_value))

            dispatched = (max_sim_values[:, None] *
                          center1_value.index_select(0, max_sim_idxes))
            dispatched = einops.rearrange(
                dispatched, "(n fc h w) sc -> n h w (fc sc)", fc=fc, h=h0, w=w0)
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
        bias: bool = True
    ) -> None:
        super().__init__()

        self.cluster = LocalCluster(
            in_depth, hidden_depth, heads_count, center_size, fold_size,
            bias=bias)
        self.norm0 = nn.GroupNorm(1, in_depth)

        self.mlp = Mlp(2 * in_depth, 2 * in_depth, in_depth, bias=bias)
        self.norm1 = nn.GroupNorm(1, in_depth)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        new_x = self.cluster(x, mask=mask)
        new_x = self.norm0(new_x)

        new_x = torch.cat([x, new_x], dim=1).permute(0, 2, 3, 1)
        new_x = self.mlp(new_x).permute(0, 3, 1, 2).contiguous()
        new_x = self.norm1(new_x)

        new_x += x
        return new_x


class GlobalClusterBlock(nn.Module):
    def __init__(
        self,
        scale: int,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        center_upsamle: bool = False,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.scale = scale
        self.center_upsample = center_upsamle

        self.cluster = GlobalCluster(
            in_depth, hidden_depth, heads_count, bias=bias)
        self.norm0 = nn.LayerNorm(in_depth)

        self.mlp3x3 = Mlp3x3(
            3 * in_depth, 2 * in_depth, 2 * in_depth, bias=bias)
        self.norm1 = nn.LayerNorm(2 * in_depth)

    def forward(
        self,
        x0: torch.Tensor,
        center0: torch.Tensor,
        center1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x0 = self.cluster(x0, center1, mask=mask)
        new_x0 = self.norm0(new_x0)

        up_center0 = F.interpolate(
            center0, scale_factor=self.scale, mode="bilinear",
            align_corners=True)
        x0 = torch.cat([x0, up_center0], dim=1)
        new_x0 = torch.cat([x0.permute(0, 2, 3, 1), new_x0], dim=3)
        new_x0 = self.mlp3x3(new_x0)
        new_x0 = self.norm1(new_x0)
        new_x0 = new_x0.permute(0, 3, 1, 2).contiguous()

        new_x0 += x0
        new_x0, new_center0 = new_x0.chunk(2, dim=1)
        if not self.center_upsample:
            new_center0 = F.avg_pool2d(new_center0, self.scale)
        return new_x0, new_center0


class LocalCoC(nn.Module):
    def __init__(
        self,
        blocks_counts: Tuple[int, int, int],
        layer_depths: Tuple[int, int, int],
        hidden_depths: Tuple[int, int, int],
        heads_counts: Tuple[int, int, int],
        center_sizes: Tuple[int, int, int],
        fold_sizes: Tuple[int, int, int],
        bias: bool = True
    ) -> None:
        super().__init__()

        layers = []
        for i in range(3):
            layer = nn.Sequential()
            for _ in range(blocks_counts[i]):
                block = LocalClusterBlock(
                    layer_depths[i], hidden_depths[i], heads_counts[i],
                    center_sizes[i], fold_sizes[i], bias=bias)
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
        return x0, x2

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
        center_upsample: bool = False,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.center_upsample = center_upsample
        self.scale = scale

        self.mlp = Mlp(2 * depth, 2 * depth, depth, bias=bias)
        self.norm = nn.GroupNorm(1, depth)

    def forward(
        self,
        x: torch.Tensor,
        center: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up_center = F.interpolate(
            center, scale_factor=self.scale, mode="bilinear",
            align_corners=True)
        new_x = torch.cat([x, up_center], dim=1).permute(0, 2, 3, 1)
        new_x = self.mlp(new_x)
        new_x = new_x.permute(0, 3, 1, 2).contiguous()
        new_x = self.norm(new_x)
        if self.center_upsample:
            new_center = new_x + up_center
            new_x = new_x + x
        else:
            new_center = F.interpolate(
                new_x, scale_factor=1 / self.scale, mode="bilinear",
                align_corners=True)
            new_x += x
            new_center += center
        return new_x, new_center


class AttentionBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        heads_count: int,
        attention: nn.Module
    ) -> None:
        super().__init__()
        self.heads_count = heads_count
        self.attention = attention

        self.down_q = nn.Conv2d(depth, depth, 4, stride=4, groups=depth, bias=False)
        self.down_kv = nn.MaxPool2d(4, stride=4)

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
        q = self.down_q(x).flatten(start_dim=2).transpose(1, 2)
        kv = self.down_kv(source).flatten(start_dim=2).transpose(1, 2)

        q = self.q_proj(q).unflatten(2, (self.heads_count, -1))
        k = self.k_proj(kv).unflatten(2, (self.heads_count, -1))
        v = self.v_proj(kv).unflatten(2, (self.heads_count, -1))
        out = self.attention(
            q, k, v, q_mask=x_mask, kv_mask=source_mask).flatten(start_dim=2)

        out = self.merge(out)
        out = self.norm1(out)
        out = out.transpose(1, 2).unflatten(2, (x.shape[2] // 4, x.shape[3] // 4))
        out = F.interpolate(out, scale_factor=4.0, mode="bilinear")

        out = torch.cat([x, out], dim=1)
        out = out.permute(0, 2, 3, 1)
        out = self.mlp(out)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        out += x
        return out


class GlobalCoC(nn.Module):
    def __init__(
        self,
        scale: int,
        in_depth: int,
        hidden_depth: int,
        heads_count: int,
        layer_count: int,
        attention: Optional[nn.Module] = None,
        use_flow: bool = False,
        use_matchability: bool = False,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.scale = scale
        self.use_flow = use_flow
        self.use_matchability = use_matchability

        # merge_block = MergeBlock(scale, in_depth, bias=bias)
        # self.merge_blocks = nn.ModuleList(
        #     [copy.deepcopy(merge_block) for _ in range(layer_count)])

        global_block = GlobalClusterBlock(
            scale, in_depth, hidden_depth, heads_count, bias=bias)
        self.global_blocks = nn.ModuleList(
            [copy.deepcopy(global_block) for _ in range(layer_count)])
        self.global_blocks[-1].center_upsample = True

        # local_block = LocalClusterBlock(
        #     in_depth, hidden_depth, heads_count, 8, 1, bias=bias)
        # self.local_blocks = nn.ModuleList(
        #     [copy.deepcopy(local_block) for _ in types])

        if attention is not None:
            attention_block = AttentionBlock(in_depth, heads_count, attention)
            self.self_blocks = nn.ModuleList([copy.deepcopy(attention_block)
                                              for _ in range(layer_count)])
            self.cross_blocks = nn.ModuleList([copy.deepcopy(attention_block)
                                               for _ in range(layer_count)])

        matchability_decoder = None
        if use_matchability:
            matchability_decoder = Mlp(in_depth, in_depth // 2, 1, bias=bias)
        self.matchability_decoders = nn.ModuleList(
            [copy.deepcopy(matchability_decoder) for _ in range(layer_count)])

        # TODO: check weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x0_8x: torch.Tensor,
        x1_8x: torch.Tensor,
        x0_32x: torch.Tensor,
        x1_32x: torch.Tensor,
        mask0_8x: Optional[torch.Tensor] = None,
        mask1_8x: Optional[torch.Tensor] = None,
        mask0_32x: Optional[torch.Tensor] = None,
        mask1_32x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        m0_8x = m1_8x = None
        if self.use_matchability:
            m0_8x, m1_8x = [], []

        mask00 = mask11 = mask01 = mask10 = None
        if mask0_8x is not None:
            mask0_8x = mask0_8x.flatten(start_dim=1)
            mask1_8x = mask1_8x.flatten(start_dim=1)
            mask0_32x = mask0_32x.flatten(start_dim=1)
            mask1_32x = mask1_32x.flatten(start_dim=1)
            mask00 = mask0_8x[:, :, None] & mask0_32x[:, None, :]
            mask11 = mask1_8x[:, :, None] & mask1_32x[:, None, :]
            mask01 = mask0_8x[:, :, None] & mask1_32x[:, None, :]
            mask10 = mask1_8x[:, :, None] & mask0_32x[:, None, :]

        for global_block, matchability_decoder in zip(
            self.global_blocks, self.matchability_decoders):
            # x0 = global_block(x0, center0, mask=mask00)
            # x1 = global_block(x1, center1, mask=mask11)
            (x0_8x, x0_32x), (x1_8x, x1_32x) = (
                global_block(x0_8x, x0_32x, x1_32x, mask=mask01),
                global_block(x1_8x, x1_32x, x0_32x, mask=mask10))
            # x0_8x, x0_32x = merge_block(x0_8x, x0_32x)
            # x1_8x, x1_32x = merge_block(x1_8x, x1_32x)
            # x0_8x = self_block(
            #     x0_8x, x0_8x, x_mask=mask0_32x, source_mask=mask0_32x)
            # x1_8x = self_block(
            #     x1_8x, x1_8x, x_mask=mask1_32x, source_mask=mask1_32x)
            # x0_8x = cross_block(
            #     x0_8x, x1_8x, x_mask=mask0_32x, source_mask=mask1_32x)
            # x1_8x = cross_block(
            #     x1_8x, x0_8x, x_mask=mask1_32x, source_mask=mask0_32x)

            if self.use_matchability:
                m0_8x.append(matchability_decoder(
                    x0_8x.flatten(start_dim=2).transpose(1, 2)).sigmoid())
                m1_8x.append(matchability_decoder(
                    x1_8x.flatten(start_dim=2).transpose(1, 2)).sigmoid())

        flow0_8x, flow1_8x = (x0_32x, x1_32x) if self.use_flow else (None, None)

        if self.use_matchability:
            m0_8x = torch.cat(m0_8x, dim=2).mean(dim=2)
            m1_8x = torch.cat(m1_8x, dim=2).mean(dim=2)
        return x0_8x, x1_8x, flow0_8x, flow1_8x, m0_8x, m1_8x
