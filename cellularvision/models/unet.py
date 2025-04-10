import torch
import torch.nn as nn

from cellularvision.functional import Conv2dBlock
from typing import List, Tuple

class Encoder(nn.Module):
    def __init__(self, channels: List[int], kernel_size: int) -> None:
        super(Encoder, self).__init__()

        in_channels, out_channels = channels[:-2], channels[1:-1]
        self.conv_layers = nn.ModuleList([
            Conv2dBlock(in_channel, out_channel, kernel_size)
            for in_channel, out_channel in zip(in_channels, out_channels)
        ])
        self.pool_layers = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(len(in_channels))])
        self.bottleneck = nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        input, outputs = input, []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            input = conv_layer(input)
            outputs.append(input)
            input = pool_layer(input)

        bottleneck = self.bottleneck(input)
        return bottleneck, outputs
    
class Decoder(nn.Module):
    def __init__(self, channels: List[int], kernel_size: int) -> None:
        super(Decoder, self).__init__()

        in_channels, out_channels = channels[:-1], channels[1:]

        self.upconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
            for in_channel, out_channel in zip(in_channels[:-1], out_channels[:-1])
        ])

        self.deconv_layers = nn.ModuleList([
            Conv2dBlock(in_channel, out_channel, kernel_size)
            for in_channel, out_channel in zip(in_channels[:-1], out_channels[:-1])
        ])
        self.final_conv = nn.Conv2d(in_channels[-1], out_channels[-1], kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, input: torch.Tensor, skip_conns: List[torch.Tensor]) -> torch.Tensor:
        output = input
        for upconv, deconv, skip_conn in zip(self.upconv_layers, self.deconv_layers, skip_conns[::-1]):
            output = upconv(output)
            output = torch.cat([skip_conn, output], dim=1)
            output = deconv(output)

        output = self.final_conv(output)
        return output

class UNet(nn.Module):
    def __init__(self, num_classes: int, encoder_channels: List[int], kernel_size: int) -> None:
        super(UNet, self).__init__()
        self.encoder = Encoder(encoder_channels, kernel_size)

        decoder_channels = encoder_channels[::-1][:-1] + [num_classes+1]
        self.decoder = Decoder(decoder_channels, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, skip_conns = self.encoder(input)
        output = self.decoder(output, skip_conns)

        return output