# =============================================================================
# FKU-Net: Fourier-KAN Enhanced U-Net with Boundary-Aware Gated Progressive Fusion
# Paper title suggestion:
# "FKU-Net: A Fourier-KAN Driven U-Net with Boundary-Aware Gated Progressive Fusion
#           for Precise Medical Image Segmentation"
#
# Key features:
# • MSRCA     : Multi-Scale Residual Convolution + EMA Attention
# • PureFKANBlock : Pure Enhanced Fourier-KAN path (no MSAB/MSCB branch)
# • BAGPF     : Boundary-Aware Gated Progressive Fusion (novel skip fusion)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from thop import profile, clever_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────
def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (int(x), int(x))
    return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ────────────────────────────────────────────────
# EMA – Efficient Multi-scale Attention
# ────────────────────────────────────────────────
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# ────────────────────────────────────────────────
# MSRCA – Multi-Scale Residual Convolution + EMA Attention
# ────────────────────────────────────────────────
class MSRCA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.ms_convs = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, k, padding=k//2, groups=out_ch, bias=False)
            for k in [3, 5, 7]
        ])
        self.ms_bn = nn.BatchNorm2d(out_ch * 3)
        self.attn  = EMA(out_ch * 3, factor=max(1, out_ch // 8))
        self.reduce = nn.Conv2d(out_ch * 3, out_ch, 1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        ms = torch.cat([c(out) for c in self.ms_convs], dim=1)
        out = self.reduce(self.attn(self.ms_bn(ms)))
        return out + res

# ────────────────────────────────────────────────
# BAGPF – Boundary-Aware Gated Progressive Fusion
# ────────────────────────────────────────────────
class BAGPF(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 3, 5, 7], reduction=16):
        super().__init__()
        self.dim_xl = dim_xl
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)

        # BFA-style dual branch refinement
        self.fg_conv = nn.Conv2d(dim_xl, dim_xl//2, 1)
        self.bd_conv = nn.Conv2d(dim_xl, dim_xl//2, 1)
        self.se_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_xl, dim_xl//16, 1),
            nn.ReLU(),
            nn.Conv2d(dim_xl//16, dim_xl, 1),
            nn.Sigmoid()
        )
        self.refine_tail = nn.Conv2d(dim_xl, dim_xl, 1)

        # Multi-dilation groups
        num_groups = len(d_list)
        group_size = dim_xl // num_groups
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_size, group_size, k_size, 1,
                          padding=(k_size + (k_size-1)*(d-1))//2, dilation=d,
                          groups=group_size)
            ) for d in d_list
        ])

        concat_ch = dim_xl * 2
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(concat_ch, concat_ch//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(concat_ch//reduction, concat_ch, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Conv2d(2, 1, 7, padding=3)
        self.gate = nn.Sequential(
            nn.Conv2d(concat_ch, dim_xl, 1),
            nn.BatchNorm2d(dim_xl),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(dim_xl, dim_xl, 1)

    def forward(self, xh, xl, pred=None, T=0.5):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=xl.shape[2:], mode='bilinear', align_corners=True)

        # Boundary-aware refinement
        refined = xl
        if pred is not None:
            p = torch.sigmoid(pred)
            fg = p
            bd = torch.clamp(1 - torch.abs(p - T) / T, 0., 1.)
            fg_feat = self.fg_conv(xl * fg)
            bd_feat = self.bd_conv(xl * bd)
            cat = torch.cat([fg_feat, bd_feat], 1)
            refined = self.refine_tail(cat * self.se_refine(cat)) + xl

        # Concat & multi-dilation
        cat = torch.cat([refined, xh], dim=1)
        groups = torch.chunk(cat, len(self.dilated), dim=1)
        dilated = torch.cat([blk(g) for blk, g in zip(self.dilated, groups)], dim=1)

        # Comprehensive attention A
        ca = self.ca(dilated)
        sa_in = torch.cat([torch.mean(dilated, 1, keepdim=True),
                           torch.max(dilated, 1, keepdim=True)[0]], dim=1)
        sa = torch.sigmoid(self.sa(sa_in))
        A = dilated * ca * sa

        # Gating
        G = self.gate(cat)

        fused = G * (xh * A) + (1 - G) * (refined * A)
        return self.tail(fused)

# ────────────────────────────────────────────────
# Pure Fourier-KAN Components
# ────────────────────────────────────────────────
class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=5, addbias=True, smooth_initialization=False):
        super().__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        grid_norm_factor = (torch.arange(gridsize) + 1) ** 2 if smooth_initialization else np.sqrt(gridsize)
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / (np.sqrt(inputdim) * grid_norm_factor)
        )
        if addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = x.reshape(-1, self.inputdim)
        k = torch.arange(1, self.gridsize + 1, device=x.device).reshape(1, 1, 1, self.gridsize)
        xrshp = x.reshape(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], dim=(-2, -1))
        if self.addbias:
            y += self.bias
        return y.reshape(outshape)

class DW_bn_relu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.flatten(2).transpose(1, 2)

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = NaiveFourierKANLayer(in_features, hidden_features)
        self.fc2 = NaiveFourierKANLayer(hidden_features, out_features)
        self.fc3 = NaiveFourierKANLayer(hidden_features, out_features)
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.dwconv_3 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, -1)
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, -1)).reshape(B, N, -1)
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, -1)).reshape(B, N, -1)
        x = self.dwconv_3(x, H, W)
        return x

class EnhancedFKANBlock(nn.Module):
    def __init__(self, dim, drop_path=0., drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.kan_layer = KANLayer(dim, dim, dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        residual = x
        x = self.norm(x)
        x = self.kan_layer(x, H, W)
        x = self.drop_path(x)
        return x + residual

class PatchEmbed(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class EnhancedFKANPath(nn.Module):
    def __init__(self, channels=320, depth=3, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(channels)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            EnhancedFKANBlock(channels, drop_path=dpr[i]) for i in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_token, Hp, Wp = self.patch_embed(x)
        for blk in self.blocks:
            x_token = blk(x_token, Hp, Wp)
        return x_token.permute(0, 2, 1).reshape(B, C, Hp, Wp)

# ────────────────────────────────────────────────
# PureFKANBlock – only Fourier-KAN path (no MSAB branch)
# ────────────────────────────────────────────────
class PureFKANBlock(nn.Module):
    def __init__(self, channels=320, depth=3, drop_path_rate=0.1):
        super().__init__()
        self.enhanced_fkan_path = EnhancedFKANPath(
            channels=channels,
            depth=depth,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x):
        identity = x
        x_fkan = self.enhanced_fkan_path(x)
        return x_fkan + identity

# ────────────────────────────────────────────────
# OverlapPatchEmbed & LayerNorm
# ────────────────────────────────────────────────
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=128, embed_dim=160):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# ────────────────────────────────────────────────
# Main Network: FKU-Net
# ────────────────────────────────────────────────
class FKU_Net(nn.Module):
    def __init__(self, num_classes=1, img_size=256, gt_ds=True, drop_path_rate=0.1):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder
        self.enc1 = MSRCA(1,   32)
        self.enc2 = MSRCA(32,  64)
        self.enc3 = MSRCA(64,  256)
        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(256)

        self.patch4 = OverlapPatchEmbed(img_size//8,  3, 2, 256, 320)
        self.patch5 = OverlapPatchEmbed(img_size//16, 3, 2, 320, 512)

        # Pure FKAN bottleneck blocks
        self.fkan4 = PureFKANBlock(channels=320, depth=3, drop_path_rate=drop_path_rate)
        self.fkan5 = PureFKANBlock(channels=512, depth=3, drop_path_rate=drop_path_rate)

        self.norm4 = nn.LayerNorm(320)
        self.norm5 = nn.LayerNorm(512)

        # Decoder
        self.dec1 = MSRCA(512, 320)
        self.dec2 = MSRCA(320, 256)
        self.dec3 = MSRCA(256, 64)
        self.dec4 = MSRCA(64,  32)
        self.dec5 = MSRCA(32,  32)

        self.dbn1 = nn.BatchNorm2d(320)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)

        # BAGPF skip fusions
        self.bagpf1 = BAGPF(64,  32)
        self.bagpf2 = BAGPF(256, 64)
        self.bagpf3 = BAGPF(320, 256)
        self.bagpf4 = BAGPF(512, 320)

        # Deep supervision heads
        if gt_ds:
            self.gt1 = nn.Conv2d(320, num_classes, 1)
            self.gt2 = nn.Conv2d(256, num_classes, 1)
            self.gt3 = nn.Conv2d(64,  num_classes, 1)
            self.gt4 = nn.Conv2d(32,  num_classes, 1)
            self.gt_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        B = x.shape[0]

        # Encoder path
        x1 = F.relu(F.max_pool2d(self.ebn1(self.enc1(x)),   2))
        x2 = F.relu(F.max_pool2d(self.ebn2(self.enc2(x1)),  2))
        x3 = F.relu(F.max_pool2d(self.ebn3(self.enc3(x2)),  2))

        x4_t, H4, W4 = self.patch4(x3)
        x4 = x4_t.reshape(B, H4, W4, 320).permute(0, 3, 1, 2)
        x4 = self.fkan4(x4)
        x4 = self.norm4(x4.flatten(2).transpose(1, 2)).reshape(B, 320, H4, W4)

        x5_t, H5, W5 = self.patch5(x4)
        x5 = x5_t.reshape(B, H5, W5, 512).permute(0, 3, 1, 2)
        x5 = self.fkan5(x5)
        x5 = self.norm5(x5.flatten(2).transpose(1, 2)).reshape(B, 512, H5, W5)

        # Decoder path + BAGPF fusions
        d1 = F.relu(F.interpolate(self.dbn1(self.dec1(x5)), scale_factor=2, mode='bilinear', align_corners=True))

        if self.gt_ds:
            gt4 = self.gt1(d1)
            x4 = self.bagpf4(x5, x4, gt4)
            gt4 = F.interpolate(gt4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            x4 = self.bagpf4(x5, x4)
        d1 += x4

        d2 = F.relu(F.interpolate(self.dbn2(self.dec2(d1)), scale_factor=2, mode='bilinear', align_corners=True))

        if self.gt_ds:
            gt3 = self.gt2(d2)
            x3 = self.bagpf3(d1, x3, gt3)
            gt3 = F.interpolate(gt3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            x3 = self.bagpf3(d1, x3)
        d2 += x3

        d3 = F.relu(F.interpolate(self.dbn3(self.dec3(d2)), scale_factor=2, mode='bilinear', align_corners=True))

        if self.gt_ds:
            gt2 = self.gt3(d3)
            x2 = self.bagpf2(d2, x2, gt2)
            gt2 = F.interpolate(gt2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            x2 = self.bagpf2(d2, x2)
        d3 += x2

        d4 = F.relu(F.interpolate(self.dbn4(self.dec4(d3)), scale_factor=2, mode='bilinear', align_corners=True))

        if self.gt_ds:
            gt1 = self.gt4(d4)
            x1 = self.bagpf1(d3, x1, gt1)
            gt1 = F.interpolate(gt1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.bagpf1(d3, x1)
        d4 += x1

        d5 = F.relu(F.interpolate(self.dec5(d4), scale_factor=2, mode='bilinear', align_corners=True))
        out = self.final(d5)

        if self.gt_ds:
            return (gt4 * self.gt_weights[0],
                    gt3 * self.gt_weights[1],
                    gt2 * self.gt_weights[2],
                    gt1 * self.gt_weights[3]), out
        return out

# ────────────────────────────────────────────────
# Quick test & complexity check
# ────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(42)
    model = FKU_Net(num_classes=1, img_size=256, gt_ds=True).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FKU-Net - Total trainable parameters: {total_params:,}")

    x = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            aux, main = out
            print("Aux shapes:   ", [p.shape for p in aux])
            print("Main shape:   ", main.shape)
        else:
            print("Output shape: ", out.shape)

    # FLOPs estimation
    try:
        flops, params = profile(model, inputs=(torch.randn(1, 1, 256, 256).to(device),), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {flops}   Params: {params}")
    except Exception as e:
        print("FLOPs calculation failed:", str(e))
