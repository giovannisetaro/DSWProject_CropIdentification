import torch
import torch.nn as nn
import torch.nn.functional as F

# --- UNet building blocks ---

class DoubleConv(nn.Module):
    """Two convolutional layers each followed by BatchNorm and ReLU activation."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling step in UNet: MaxPooling followed by DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling step in UNet: Upsample (or ConvTranspose2d) followed by DoubleConv."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if necessary to match size of x2 before concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Full UNet architecture with configurable input/output channels."""
    def __init__(self, in_ch=64, out_ch=10, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# --- Temporal Pixel-wise CNN ---

class TemporalPixelCNN(nn.Module):
    """
    Applies temporal convolutions independently to each pixel's time-series.
    Input shape: [Batch, Channels, Time, Height, Width] = [B, C, T, H, W]
    Output shape: [B, Feature_dim, H, W] — features aggregated over time per pixel.
    """
    def __init__(self, in_channels=4, out_channels=64, kernel_size=3,
                 nb_layers=2, pooling_type='max', dropout=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            input_dim = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv1d(input_dim, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False),  # Use non-inplace ReLU for AMP safety
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.temporal_net = nn.Sequential(*layers)
        self.pooling_type = pooling_type

    def forward(self, x):
        B, T, C, H, W = x.shape  # Batch, Time, Channels, Height, Width

        # Rearrange so channels before time dimension per pixel for Conv1d
        x = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, T]

        # Flatten batch and spatial dims, so Conv1d input shape: [B*H*W, C, T]
        x = x.reshape(-1, C, T)

        # Apply temporal 1D conv layers
        x = self.temporal_net(x)  # output shape [B*H*W, out_channels, T]

        # Pool across time to reduce dimension
        if self.pooling_type == 'max':
            x = x.max(dim=2).values
        elif self.pooling_type == 'mean':
            x = x.mean(dim=2)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        # Reshape back to [B, H, W, out_channels], then permute to [B, out_channels, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

# --- Full Model combining TemporalPixelCNN and UNet ---

class CropTypeClassifier(nn.Module):
    """
    Complete model combining temporal pixel-wise CNN with a UNet for spatial feature extraction.
    Input: [B, C, T, H, W], Output: [B, num_classes, H, W]
    """
    def __init__(self, num_classes, temporal_out_channels=64, kernel_size=3,
                 nb_temporal_layers=2, pooling_type='max', dropout=0.3):
        super().__init__()
        self.temporal_cnn = TemporalPixelCNN(
            in_channels=4,
            out_channels=temporal_out_channels,
            kernel_size=kernel_size,
            nb_layers=nb_temporal_layers,
            pooling_type=pooling_type,
            dropout=dropout
        )
        self.unet = UNet(in_ch=temporal_out_channels, out_ch=num_classes)

    def forward(self, x):
        x = self.temporal_cnn(x)  # [B, temporal_out_channels, H, W]
        x = self.unet(x)          # [B, num_classes, H, W]
        return x
