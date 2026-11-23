import torch
import torch.nn as nn


def apply_complex(fn, x):
    return torch.complex(fn(x.real), fn(x.imag))


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = apply_complex(self.norm1, x)
        x = apply_complex(self.relu, x)

        x = self.conv2(x)
        x = apply_complex(self.norm2, x)
        x = apply_complex(self.relu, x)
        return x


class ComplexEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ComplexConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = apply_complex(self.pool, features)
        return features, pooled


class ComplexDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=torch.cfloat)
        self.conv = ComplexConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ComplexUNet(nn.Module):
    def __init__(self, in_channels=4, features=(16, 32, 64, 128)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        current_in = in_channels
        for feature in features:
            self.encoder_blocks.append(ComplexEncoderBlock(current_in, feature))
            current_in = feature

        self.bottleneck = ComplexConvBlock(features[-1], features[-1] * 2)

        self.decoder_blocks = nn.ModuleList()
        for feature in reversed(features):
            self.decoder_blocks.append(ComplexDecoderBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1, dtype=torch.cfloat)

    def forward(self, x):
        skips = []
        for block in self.encoder_blocks:
            features, x = block(x)
            skips.append(features)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip)

        x = self.final_conv(x)
        mag = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-8)
        return mag.squeeze(1)


if __name__ == "__main__":
    model = ComplexUNet()
    real_equiv = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(f"Real-equivalent params: {real_equiv:,}")

    fake_x = torch.randn(2, 4, 384, 384, dtype=torch.cfloat)
    fake_y = torch.rand(2, 384, 384)

    out = model(fake_x)
    print("Output:", out.shape, out.dtype)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for step in range(5):
        loss = loss_fn(model(fake_x), fake_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1} | L1 Loss: {loss.item():.4f}")
