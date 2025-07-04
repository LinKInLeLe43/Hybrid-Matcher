from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Sequential

# from .submodules import FFN


# class FineMatching(Module):
#     def __init__(
#         self, dim: int, num_bins: int, window_size: int, bias: bool = False
#     ) -> None:
#         super().__init__()
#         self.window_size = window_size
#         self.register_buffer(
#             "local_coords",
#             torch.arange(0, num_bins + 1) / num_bins - 0.5,
#             persistent=False,
#         )

#         self.query_ffn = FFN(dim, dim, dim, bias=bias)
#         self.ref_ffn = FFN(dim, dim, dim, bias=bias)
#         self.merge_ffn = FFN(dim * 2, dim * 2, dim * 2, bias=bias)
#         self.x_head = FFN(dim * 2, dim * 2, (num_bins + 1) + 1, bias=bias)
#         self.y_head = FFN(dim * 2, dim * 2, (num_bins + 1) + 1, bias=bias)

#     def _decode_pred(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         coord = (x[..., :-1].softmax(dim=-1) * self.local_coords).sum(
#             dim=-1, keepdim=True
#         )
#         sigma = x[..., -1:].sigmoid()
#         return coord, sigma

#     def forward(
#         self, x0_list: Sequence[Tensor], x1_list: Sequence[Tensor]
#     ) -> Dict[str, Tensor]:
#         assert len(x0_list) == 2 and len(x1_list) == 2
#         x0, x1 = None, None

#         query = self.query_ffn(torch.cat([x0, x1]))
#         ref = self.ref_ffn(torch.cat([x1, x0]))
#         feat = self.merge_ffn(torch.cat([query, ref], dim=-1))

#         pred_mu, pred_sigma = self._decode_pred()
#         if self.training:
#             out = {"pred_mu": pred_sigma, "pred_sigma": pred_sigma}


#         offsets0_to_1, offsets1_to_0 = (pred_mu * self.window_size).chunk(2)
#         scores0_to_1, scores1_to_0 = 1.0 - pred_sigma.mean(dim=-1).chunk(2)
#         offsets0 = offsets1_to_0(scores1_to_0 > scores0_to_1, 0.0)
#         offsets1 = offsets0_to_1(scores0_to_1 > scores1_to_0, 0.0)
#         out.update(fine_reg_biases0=offsets0, fine_reg_biases1=offsets1)
#         return out


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Conv1d_BN_Act(Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv1d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm1d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        self.add_module("a", nn.GELU())


def soft_argmax(x, temperature=1.0):
    L = x.shape[1]
    assert L % 2  # L is odd to ensure symmetry
    idx = torch.arange(0, L, 1, device=x.device).repeat(x.shape[0], 1)
    scale_x = x / temperature
    out = F.softmax(scale_x, dim=1) * idx
    out = out.sum(dim=1, keepdim=True)

    return out


class FineMatching(Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 256
        self.local_resolution = 8
        self.coord_length = 16

        # network
        self.fine_conv = nn.Sequential(
            self._make_layer(BasicBlock, dim // 2, dim // 2, stride=1),
            conv1x1(dim // 2, dim),
            nn.BatchNorm2d(dim),
        )

        self.query_encoder = nn.Sequential(
            Conv1d_BN_Act(dim, dim), Conv1d_BN_Act(dim, dim)
        )

        self.reference_encoder = nn.Sequential(
            Conv1d_BN_Act(dim, dim), Conv1d_BN_Act(dim, dim)
        )

        self.merge_qr = nn.Sequential(
            Conv1d_BN_Act(dim * 2, dim * 2), Conv1d_BN_Act(dim * 2, dim * 2)
        )

        self.x_head = nn.Sequential(
            Conv1d_BN_Act(dim * 2, dim * 2),
            nn.Conv1d(dim * 2, self.coord_length + 2, 1),
        )

        self.y_head = nn.Sequential(
            Conv1d_BN_Act(dim * 2, dim * 2),
            nn.Conv1d(dim * 2, self.coord_length + 2, 1),
        )

    def _make_layer(self, block, in_dim, out_dim, stride=1):
        layer1 = block(in_dim, out_dim, stride=stride)
        layer2 = block(out_dim, out_dim, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(
        self,
        feat_f0: Tensor,
        feat_f1: Tensor,
        feat_c0: Tensor,
        feat_c1: Tensor,
        indices_list: Sequence[Tensor],
    ) -> Dict[str, Tensor]:
        b_indices, i_indices, j_indices = indices_list
        if b_indices.shape[0] == 0:
            out = {
                "fine_reg_biases0": feat_c0.new_empty(0, 2),
                "fine_reg_biases1": feat_c1.new_empty(0, 2),
            }
            return out

        feat_f0, feat_f1 = self.fine_conv(feat_f0), self.fine_conv(feat_f1)
        feat0 = (feat_f0 + feat_c0).flatten(start_dim=-2)[
            None, b_indices, :, i_indices
        ]
        feat1 = (feat_f1 + feat_c1).flatten(start_dim=-2)[
            None, b_indices, :, j_indices
        ]

        q = self.query_encoder(
            torch.cat([feat0, feat1], dim=1).transpose(-1, -2)
        )
        r = self.reference_encoder(
            torch.cat([feat1, feat0], dim=1).transpose(-1, -2)
        )
        out = self.merge_qr(torch.cat([q, r], dim=1))

        x = self.x_head(out).permute(0, 2, 1).contiguous()
        y = self.y_head(out).permute(0, 2, 1).contiguous()

        x01, x10 = x.chunk(2, dim=1)
        x01 = x01.reshape(-1, self.coord_length + 2)
        x10 = x10.reshape(-1, self.coord_length + 2)
        x_out = torch.cat([x01, x10])

        y01, y10 = y.chunk(2, dim=1)
        y01 = y01.reshape(-1, self.coord_length + 2)
        y10 = y10.reshape(-1, self.coord_length + 2)
        y_out = torch.cat([y01, y10])

        x_cls = x_out[:, : self.coord_length + 1]
        coord_x = (
            soft_argmax(x_cls) / self.coord_length - 0.5
        )  # range [-0.5, +0.5]
        x_sigma = x_out[:, -1:].sigmoid()

        y_cls = y_out[:, : self.coord_length + 1]
        coord_y = soft_argmax(y_cls) / self.coord_length - 0.5
        y_sigma = y_out[:, -1:].sigmoid()

        mu = torch.cat([coord_x, coord_y], dim=1)
        sigma = torch.cat([x_sigma, y_sigma], dim=1)
        out = {"pred_mu": mu, "pred_sigma": sigma}

        with torch.no_grad():
            offsets0_to_1, offsets1_to_0 = (mu * self.local_resolution).chunk(2)
            scores0_to_1, scores1_to_0 = (1.0 - sigma.mean(dim=-1)).chunk(2)
            offsets1_to_0[scores1_to_0 < scores0_to_1] = 0.0
            offsets0_to_1[scores0_to_1 < scores1_to_0] = 0.0
            out["fine_reg_biases0"] = offsets1_to_0
            out["fine_reg_biases1"] = offsets0_to_1
        return out

    #     if data.get("target_uv", None) is not None:
    #         gt_uv = data["target_uv"]
    #         mask = data["target_uv_weight"].clone()

    #         if mask.sum() == 0:
    #             mask[0] = True
    #         mask_coord = coord[mask]
    #         mask_gt_uv = gt_uv[mask]
    #         mask_sigma = sigma[mask]

    #         mask_sigma = torch.clamp(mask_sigma, 1e-6, 1 - 1e-6)
    #         bar_mu = (mask_coord - mask_gt_uv) / mask_sigma

    #         log_phi = self.flow.log_prob(bar_mu).unsqueeze(-1)
    #         nf_loss = torch.log(mask_sigma) - log_phi

    #         data.update(
    #             {
    #                 "pred_coord": coord,
    #                 "pred_score": 1.0 - torch.mean(sigma, dim=-1).flatten(),
    #                 "mask_coord": mask_coord,
    #                 "mask_sigma": mask_sigma,
    #                 "nf_loss": nf_loss,
    #             }
    #         )
    #     else:
    #         data.update(
    #             {
    #                 "pred_coord": coord,
    #                 "pred_score": 1.0 - torch.mean(sigma, dim=-1).flatten(),
    #             }
    #         )

    #     self.final_matching_selection(data)

    #     return data["pred_coord"], data["pred_score"]

    # @torch.no_grad()
    # def final_matching_selection(self, data):
    #     offset = data["pred_coord"] * self.local_resolution

    #     if self.bi_directional_refine:
    #         fine_offset01, fine_offset10 = torch.clamp(
    #             offset, -self.local_resolution / 2, self.local_resolution / 2
    #         ).chunk(2)
    #     else:
    #         fine_offset01 = torch.clamp(
    #             offset, -self.local_resolution / 2, self.local_resolution / 2
    #         )

    #     h0, w0 = data["hw0_i"]
    #     h1, w1 = data["hw1_i"]
    #     scale0 = data["scale0"][data["b_ids"]] if "scale0" in data else 1.0
    #     scale1 = data["scale1"][data["b_ids"]] if "scale1" in data else 1.0
    #     scale0_w = scale0[:, 0] if "scale0" in data else 1.0
    #     scale0_h = scale0[:, 1] if "scale0" in data else 1.0
    #     scale1_w = scale1[:, 0] if "scale1" in data else 1.0
    #     scale1_h = scale1[:, 1] if "scale1" in data else 1.0

    #     # Filter by mconf and border
    #     mkpts0_f = data["mkpts0_c"]
    #     mkpts1_f = data["mkpts1_c"] + fine_offset01 * scale1
    #     mask = (
    #         (data["mconf"] > self.mconf_thr)
    #         & (mkpts0_f[:, 0] >= self.border_rm)
    #         & (mkpts0_f[:, 0] <= w0 * scale0_w - self.border_rm)
    #         & (mkpts0_f[:, 1] >= self.border_rm)
    #         & (mkpts0_f[:, 1] <= h0 * scale0_h - self.border_rm)
    #         & (mkpts1_f[:, 0] >= self.border_rm)
    #         & (mkpts1_f[:, 0] <= w1 * scale1_w - self.border_rm)
    #         & (mkpts1_f[:, 1] >= self.border_rm)
    #         & (mkpts1_f[:, 1] <= h1 * scale1_h - self.border_rm)
    #     )
    #     if self.bi_directional_refine:
    #         mkpts0_f_ = data["mkpts0_c"] + fine_offset10 * scale0
    #         mkpts1_f_ = data["mkpts1_c"]
    #         mask_ = (
    #             (data["mconf"] > self.mconf_thr)
    #             & (mkpts0_f_[:, 0] >= self.border_rm)
    #             & (mkpts0_f_[:, 0] <= w0 * scale0_w - self.border_rm)
    #             & (mkpts0_f_[:, 1] >= self.border_rm)
    #             & (mkpts0_f_[:, 1] <= h0 * scale0_h - self.border_rm)
    #             & (mkpts1_f_[:, 0] >= self.border_rm)
    #             & (mkpts1_f_[:, 0] <= w1 * scale1_w - self.border_rm)
    #             & (mkpts1_f_[:, 1] >= self.border_rm)
    #             & (mkpts1_f_[:, 1] <= h1 * scale1_h - self.border_rm)
    #         )

    #     if self.bi_directional_refine:
    #         mkpts0_f = torch.cat([mkpts0_f, mkpts0_f_])
    #         mkpts1_f = torch.cat([mkpts1_f, mkpts1_f_])
    #         mask = torch.cat([mask, mask_])
    #         data["mconf"] = torch.cat([data["mconf"], data["mconf"]])
    #         data["b_ids"] = torch.cat([data["b_ids"], data["b_ids"]])

    #     # Filter by sigma
    #     if self.bi_directional_refine and self.sigma_selection:
    #         # Retain the more confident matching pair with a smaller sigma (more significant) in the bi-directional matching pairs
    #         pred_score01, pred_score10 = data["pred_score"].chunk(2)
    #         pred_score_mask = pred_score01 > pred_score10
    #         pred_score_mask = torch.cat([pred_score_mask, ~pred_score_mask])
    #         pred_score_mask &= data["pred_score"] > self.sigma_thr
    #         mask &= pred_score_mask

    #     data.update(
    #         {
    #             # "gt_mask": data["mconf"] == 0,
    #             "m_bids": data["b_ids"][mask],
    #             "mkpts0_f": mkpts0_f[mask],
    #             "mkpts1_f": mkpts1_f[mask],
    #             "mconf": data["mconf"][mask],
    #         }
    #     )
