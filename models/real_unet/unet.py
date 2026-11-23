import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip_features):
        x = self.up(x)
        x = torch.cat([skip_features, x], dim=1)
        return self.conv(x)


class RealUNet(nn.Module):
    def __init__(self, in_channels=8, features=(24, 48, 96, 192)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        current_in = in_channels
        for feature in features:
            self.encoder_blocks.append(EncoderBlock(current_in, feature))
            current_in = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.decoder_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.decoder_blocks.append(DecoderBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for block in self.encoder_blocks:
            feat, x = block(x)
            skips.append(feat)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip)

        return self.final_conv(x).squeeze(1)
