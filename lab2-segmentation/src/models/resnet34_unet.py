import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# ResNet34 encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# UNet decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        out = self.up(x)
        out = self.conv1(out)
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out = self.relu(out)
        return out


class ResNet34_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.enc1 = self._make_layer(64, 64, 3)
        self.enc2 = self._make_layer(64, 128, 4, stride=2)
        self.enc3 = self._make_layer(128, 256, 6, stride=2)
        self.enc4 = self._make_layer(256, 512, 3, stride=2)
        self.enc5 = self._make_layer(512, 256, 3)

        self.dec1 = DecoderBlock(256 + 512, 32)
        self.dec2 = DecoderBlock(32 + 256, 32)
        self.dec3 = DecoderBlock(32 + 128, 32)
        self.dec4 = DecoderBlock(32 + 64, 32)
        self.dec5 = DecoderBlock(32, 32)

        # self.out = nn.Conv2d(32, n_classes, kernel_size=1)
        self.out = EncoderBlock(32, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        enc1 = self.enc1(x)  # (B, 64, H/4, W/4)
        enc2 = self.enc2(enc1)  # (B, 128, H/8, W/8)
        enc3 = self.enc3(enc2)  # (B, 256, H/16, W/16)
        enc4 = self.enc4(enc3)  # (B, 512, H/32, W/32)
        enc5 = self.enc5(enc4)  # (B, 256, H/32, W/32)

        dec1 = self.dec1(torch.cat([enc5, enc4], dim=1))  # (B, 32, H/16, W/16)
        dec2 = self.dec2(torch.cat([dec1, enc3], dim=1))  # (B, 32, H/8, W/8)
        dec3 = self.dec3(torch.cat([dec2, enc2], dim=1))  # (B, 32, H/4, W/4)
        dec4 = self.dec4(torch.cat([dec3, enc1], dim=1))  # (B, 32, H/2, W/2)
        dec5 = self.dec5(dec4)  # (B, 32, H, W)

        out = self.out(dec5)
        return out

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(EncoderBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(EncoderBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        