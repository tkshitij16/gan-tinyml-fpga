import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to same spatial size as skip connection
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StudentUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)      # 16 -> 32
        self.down2 = Down(base_ch * 2, base_ch * 4)  # 32 -> 64

        self.up1 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)  # (64+32)->32
        self.up2 = Up(base_ch * 2 + base_ch, base_ch)          # (32+16)->16

        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)     # B,16,H,W
        x2 = self.down1(x1)  # B,32,H/2,W/2
        x3 = self.down2(x2)  # B,64,H/4,W/4

        u1 = self.up1(x3, x2)  # B,32,H/2,W/2
        u2 = self.up2(u1, x1)  # B,16,H,W
        out = self.outc(u2)    # B,3,H,W
        return out
