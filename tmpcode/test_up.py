import torch
import torch.nn as nn


class HeightPixelShuffleEncoder(nn.Module):
    def __init__(self, upscale_factor=4, base_dim=64):
        super().__init__()
        self.r = upscale_factor

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*4, self.r, kernel_size=3, padding=1),  # 准备做shuffle
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.stem(x)     # (B, base_dim, H, W)
        x = self.blocks(x)   # (B, r, H, W)

        # 高度方向 pixel shuffle
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, H, r, W)
        x = x.view(B, 1, self.r * H, W)          # (B, 1, r·H, W)
        return x



B, W, H = 2, 64, 128
r = 4
x = torch.randn(B, 1, W, H)

encoder = HeightPixelShuffleEncoder(upscale_factor=r)
import ipdb
ipdb.set_trace()
z = encoder(x)         # (B, 1, W, r·H)
