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

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
    def forward(self, class_labels):
        return self.embedding(class_labels)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None, class_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.time_mlp = None
        self.class_mlp = None
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
            
        if class_emb_dim is not None:
            self.class_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(class_emb_dim, out_channels)
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, time_emb=None, class_emb=None):
        out = self.double_conv(x)
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            out = out + time_emb[:, :, None, None]
        if self.class_mlp is not None and class_emb is not None:
            class_emb = self.class_mlp(class_emb)
            out = out + class_emb[:, :, None, None]
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        scale = (C // 8) ** -0.5
        attn = torch.einsum("bchw,bcHW->bhwHW", q * scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhwHW,bcHW->bchw", attn, v)
        return x + self.proj(out)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust mid_channels to handle concatenated input
            self.conv = DoubleConv(in_channels + out_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, num_classes=5, base_channels=64, depth=5, time_emb_dim=128, class_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.class_embedding = ClassEmbedding(num_classes, class_emb_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, class_emb_dim),
            nn.SiLU(),
            nn.Linear(class_emb_dim, class_emb_dim)
        )

        self.inc = DoubleConv(n_channels, base_channels, 
                            time_emb_dim=time_emb_dim, 
                            class_emb_dim=class_emb_dim)
        
        self.down_layers = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch))

        self.up_layers = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            self.up_layers.append(Up(in_ch, out_ch))

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x, time, class_labels):
        time_emb = self.time_mlp(time)
        class_emb = self.class_mlp(self.class_embedding(class_labels))
        
        x_skips = []
        x = self.inc(x, time_emb, class_emb)
        x_skips.append(x)
        
        for down in self.down_layers:
            x = down(x)
            x_skips.append(x)

        x_skips = x_skips[:-1][::-1]
        for i, up in enumerate(self.up_layers):
            x = up(x, x_skips[i])

        return self.outc(x)