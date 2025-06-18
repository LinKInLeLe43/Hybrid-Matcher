import torch
from kornia import create_meshgrid
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F


def create_basic_block(in_dim: int, out_dim: int) -> Module:
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_dim, affine=False),
        nn.ReLU(inplace=True),
    )
    return block


class TinyConvRefiner(Module):
    def __init__(self) -> None:
        super().__init__()
        match_dim = 256
        self.matcher = nn.Sequential(
            create_basic_block(128 + 128 + 2, match_dim),
            create_basic_block(match_dim, match_dim),
            create_basic_block(match_dim, match_dim),
            create_basic_block(match_dim, match_dim),
            nn.Conv2d(match_dim, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x0: Tensor, x1: Tensor, indices0_to_1: Tensor) -> Tensor:
        n, _, h0, w0 = x0.shape
        _, _, h1, w1 = x1.shape

        grid = create_meshgrid(
            h1,
            w1,
            normalized_coordinates=False,
            device=x0.device,
            dtype=x0.dtype,
        )
        grid = (
            (2 * (grid + 0.5) / grid.new_tensor([w1, h1]) - 1)
            .flatten(start_dim=1, end_dim=2)
            .expand(n, -1, -1)
        )
        indices0_to_1 = indices0_to_1[:, :, None].expand(-1, -1, 2)
        flow = grid.gather(1, indices0_to_1).view(n, h0, w0, 2)
        warped_x1 = F.grid_sample(x1, flow, mode="bilinear")
        delta = (
            self.matcher(
                torch.cat([x0, warped_x1, flow.permute(0, 3, 1, 2)], dim=1)
            )
            .flatten(start_dim=2)
            .transpose(-1, -2)
        )
        return delta


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def local_correlation(feature0, feature1, local_radius, flow):
#     r = local_radius
#     K = (2 * r + 1) ** 2
#     B, c, h, w = feature0.size()
#     corr = torch.empty(
#         (B, K, h, w), device=feature0.device, dtype=feature0.dtype
#     )
#     coords = flow.permute(0, 2, 3, 1)
#     local_window = torch.meshgrid(
#         (
#             torch.linspace(
#                 -2 * local_radius / h,
#                 2 * local_radius / h,
#                 2 * r + 1,
#                 device=feature0.device,
#             ),
#             torch.linspace(
#                 -2 * local_radius / w,
#                 2 * local_radius / w,
#                 2 * r + 1,
#                 device=feature0.device,
#             ),
#         ),
#         indexing="ij",
#     )
#     local_window = (
#         torch.stack((local_window[1], local_window[0]), dim=-1)[None]
#         .expand(1, 2 * r + 1, 2 * r + 1, 2)
#         .reshape(1, (2 * r + 1) ** 2, 2)
#     )
#     for _ in range(B):
#         with torch.no_grad():
#             local_window_coords = (
#                 coords[_, :, :, None] + local_window[:, None, None]
#             ).reshape(1, h, w * (2 * r + 1) ** 2, 2)
#             window_feature = F.grid_sample(
#                 feature1[_ : _ + 1],
#                 local_window_coords,
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             window_feature = window_feature.reshape(c, h, w, (2 * r + 1) ** 2)
#         corr[_] = (
#             (feature0[_, ..., None] / (c**0.5) * window_feature)
#             .sum(dim=0)
#             .permute(2, 0, 1)
#         )
#     return corr


# class ConvRefiner(nn.Module):
#     def __init__(
#         self,
#         in_dim=6,
#         hidden_dim=16,
#         out_dim=2,
#         dw=False,
#         kernel_size=5,
#         hidden_blocks=3,
#         displacement_emb=None,
#         displacement_emb_dim=None,
#         local_corr_radius=None,
#         corr_in_other=None,
#         no_im_B_fm=False,
#         amp=False,
#         concat_logits=False,
#         use_bias_block_1=True,
#         use_cosine_corr=False,
#         disable_local_corr_grad=False,
#         is_classifier=False,
#         sample_mode="bilinear",
#         norm_type=nn.BatchNorm2d,
#         bn_momentum=0.1,
#     ):
#         super().__init__()
#         self.bn_momentum = bn_momentum
#         self.block1 = self.create_block(
#             in_dim,
#             hidden_dim,
#             dw=dw,
#             kernel_size=kernel_size,
#             bias=use_bias_block_1,
#         )
#         self.hidden_blocks = nn.Sequential(
#             *[
#                 self.create_block(
#                     hidden_dim,
#                     hidden_dim,
#                     dw=dw,
#                     kernel_size=kernel_size,
#                     norm_type=norm_type,
#                 )
#                 for hb in range(hidden_blocks)
#             ]
#         )
#         self.hidden_blocks = self.hidden_blocks
#         self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
#         if displacement_emb:
#             self.has_displacement_emb = True
#             self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
#         else:
#             self.has_displacement_emb = False
#         self.local_corr_radius = local_corr_radius
#         self.corr_in_other = corr_in_other
#         self.no_im_B_fm = no_im_B_fm
#         self.amp = amp
#         self.concat_logits = concat_logits
#         self.use_cosine_corr = use_cosine_corr
#         self.disable_local_corr_grad = disable_local_corr_grad
#         self.is_classifier = is_classifier
#         self.sample_mode = sample_mode

#     def create_block(
#         self,
#         in_dim,
#         out_dim,
#         dw=False,
#         kernel_size=5,
#         bias=True,
#         norm_type=nn.BatchNorm2d,
#     ):
#         num_groups = 1 if not dw else in_dim
#         if dw:
#             assert out_dim % in_dim == 0, (
#                 "outdim must be divisible by indim for depthwise"
#             )
#         conv1 = nn.Conv2d(
#             in_dim,
#             out_dim,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size // 2,
#             groups=num_groups,
#             bias=bias,
#         )
#         norm = (
#             norm_type(out_dim, momentum=self.bn_momentum)
#             if norm_type is nn.BatchNorm2d
#             else norm_type(num_channels=out_dim)
#         )
#         relu = nn.ReLU(inplace=True)
#         conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
#         return nn.Sequential(conv1, norm, relu, conv2)

#     def forward(self, x, y, flow, scale_factor=1, logits=None):
#         b, c, hs, ws = x.shape
#         x_hat = F.grid_sample(
#             y,
#             flow.permute(0, 2, 3, 1),
#             align_corners=False,
#             mode=self.sample_mode,
#         )
#         if self.has_displacement_emb:
#             im_A_coords = torch.meshgrid(
#                 (
#                     torch.linspace(
#                         -1 + 1 / hs, 1 - 1 / hs, hs, device=x.device
#                     ),
#                     torch.linspace(
#                         -1 + 1 / ws, 1 - 1 / ws, ws, device=x.device
#                     ),
#                 ),
#                 indexing="ij",
#             )
#             im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
#             im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
#             in_displacement = flow - im_A_coords
#             emb_in_displacement = self.disp_emb(
#                 40 / 32 * scale_factor * in_displacement
#             )
#             if self.local_corr_radius:
#                 if self.corr_in_other:
#                     # Corr in other means take a kxk grid around the predicted coordinate in other image
#                     local_corr = local_correlation(
#                         x,
#                         y,
#                         local_radius=self.local_corr_radius,
#                         flow=flow,
#                         sample_mode=self.sample_mode,
#                     )
#                 else:
#                     raise NotImplementedError(
#                         "Local corr in own frame should not be used."
#                     )
#                 if self.no_im_B_fm:
#                     x_hat = torch.zeros_like(x)
#                 d = torch.cat(
#                     (x, x_hat, emb_in_displacement, local_corr), dim=1
#                 )
#             else:
#                 d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
#         else:
#             if self.no_im_B_fm:
#                 x_hat = torch.zeros_like(x)
#             d = torch.cat((x, x_hat), dim=1)
#         if self.concat_logits:
#             d = torch.cat((d, logits), dim=1)
#         d = self.block1(d)
#         d = self.hidden_blocks(d)
#         d = self.out_conv(d.float())
#         displacement, certainty = d[:, :-1], d[:, -1:]
#         return displacement, certainty
