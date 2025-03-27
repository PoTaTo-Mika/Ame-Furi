import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, time_emb=None):
        out = self.double_conv(x)
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            out = out + time_emb[:, :, None, None, None]  # Broadcast time embedding for 3D
        return out

class AttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        scale = (C // 8) ** -0.5
        attn = torch.einsum("bcdhw,bcdHW->bdhwHW", q * scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bdhwHW,bcdHW->bcdhw", attn, v)
        return x + self.proj(out)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_channels=64, depth=5, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.inc = DoubleConv3D(n_channels, base_channels, time_emb_dim=time_emb_dim)
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(Down3D(in_ch, out_ch))

        self.up_layers = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.up_layers.append(Up3D(in_ch, out_ch))

        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x, time):
        time_emb = self.time_mlp(time)
        x_skips = []
        x = self.inc(x, time_emb)
        x_skips.append(x)
        for down in self.down_layers:
            x = down(x)
            x_skips.append(x)

        x_skips = x_skips[:-1][::-1]
        for i, up in enumerate(self.up_layers):
            x = up(x, x_skips[i])

        return self.outc(x)