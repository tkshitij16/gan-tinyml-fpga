#!/usr/bin/env python3
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU block used in encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + ConvBlock with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        in_channels: channels of the upsampled feature
        skip_channels: channels from the encoder skip connection
        out_channels: output channels after fusion
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # If there is a 1-pixel mismatch in H/W due to pooling, center-crop skip
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[
                :,
                :,
                dh // 2 : skip.shape[-2] - (dh - dh // 2),
                dw // 2 : skip.shape[-1] - (dw - dw // 2),
            ]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StudentUNet(nn.Module):
    """
    Lightweight U-Net style network for edges → shoes.

    Args:
        in_channels:  number of input channels (1 for grayscale edges)
        out_channels: number of output channels (3 for RGB shoe image)
        base_channels: base width of UNet (e.g. 32)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)          # 1  -> 32
        self.pool1 = nn.MaxPool2d(2)                              # 256 -> 128

        self.enc2 = ConvBlock(base_channels, base_channels * 2)   # 32 -> 64
        self.pool2 = nn.MaxPool2d(2)                              # 128 -> 64

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # 64 -> 128
        self.pool3 = nn.MaxPool2d(2)                                 # 64 -> 32

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)  # 128 -> 256

        # Decoder
        self.up3 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )
        self.up2 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )
        self.up1 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        # Final conv to RGB
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)       # (B, base, 256,256)
        p1 = self.pool1(x1)     # (B, base, 128,128)

        x2 = self.enc2(p1)      # (B, 2*base, 128,128)
        p2 = self.pool2(x2)     # (B, 2*base, 64,64)

        x3 = self.enc3(p2)      # (B, 4*base, 64,64)
        p3 = self.pool3(x3)     # (B, 4*base, 32,32)

        # Bottleneck
        b = self.bottleneck(p3)  # (B, 8*base, 32,32)

        # Decoder
        d3 = self.up3(b, x3)     # (B, 4*base, 64,64)
        d2 = self.up2(d3, x2)    # (B, 2*base, 128,128)
        d1 = self.up1(d2, x1)    # (B, base, 256,256)

        out = self.final_conv(d1)   # (B, out_channels, 256,256)
        out = torch.sigmoid(out)    # keep output in [0,1] for L1 against normalized teacher

        return out


class TinyStub(StudentUNet):
    """
    Backward-compatible stub used earlier in the project.
    Just a StudentUNet with default args.
    """

    def __init__(self):
        super().__init__(in_channels=1, out_channels=3, base_channels=32)
