import torch
import torch.nn as nn

from torchvision import models # type: ignore
from cellularvision.functional import Conv2dBlock
from typing import Tuple, List

class Encoder(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Encoder, self).__init__()

        self.conv_params = [
            (3, 32, 1), (32, 64, 1), (64, 128, 2),
            (128, 256, 2)
        ]

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2dBlock(
                    in_channels=conv_param[0], out_channels=conv_param[1],
                    kernel_size=3, num_blocks=conv_param[2]
                ),
                nn.BatchNorm2d(conv_param[1]),
                nn.ReLU()
            )
            for conv_param in self.conv_params
        ])
        self.pool_layer = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_params[0] = (num_classes + 1, 32, 1)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output, pool_indices = input, []

        for conv_block in self.conv_blocks:
            output = conv_block(output)
            output, indices = self.pool_layer(output)
            pool_indices.append(indices)

        return output, pool_indices
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super(DecoderBlock, self).__init__()

        self.upsample = nn.MaxUnpool2d(2, 2)
        self.decoder = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels, out_chann, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_chann),
                nn.ReLU()
            )
            for i in range(depth)
            for out_chann in [in_channels if i != depth - 1 else out_channels]
        ])

    def forward(self, input: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        upsampled_input = self.upsample(input, pooling_indices)
        output = self.decoder(upsampled_input)
        return output
    
class SegNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SegNet, self).__init__()

        self.encoder = Encoder(num_classes)
        conv_params = self.encoder.conv_params

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(conv_param[1], conv_param[0], conv_param[2])
            for conv_param in conv_params[::-1]
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, pooling_indices = self.encoder(input)
        for i, decoder_block in enumerate(self.decoder_blocks):
            indices = pooling_indices[::-1][i]
            output = decoder_block(output, indices)

        return output
