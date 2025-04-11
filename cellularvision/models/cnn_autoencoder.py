import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6) -> None:
        super(SegmentationCNN, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        self.bottleneck = self.conv_block(256, 512)

        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.up3(b)
        d2 = self.up2(d3)
        d1 = self.up1(d2)

        out = self.final(d1)
        return out
