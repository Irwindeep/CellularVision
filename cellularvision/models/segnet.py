import torch
import torch.nn as nn

from torchvision import models # type: ignore
from cellularvision.functional import Conv2dBlock
from typing import Tuple, List

class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super(Encoder, self).__init__()

        encoder = (
            models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1) if pretrained
            else models.vgg11_bn()
        ).features
        encoder_list = list(encoder.children())

        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.encoder_channels = [3, 64, 128, 256, 512, 512]

        conv_block: List[nn.Module] = []
        for module in encoder_list:
            if isinstance(module, nn.MaxPool2d):
                self.conv_blocks.append(nn.Sequential(*conv_block))
                module.return_indices, conv_block = True, []
                self.pool_layers.append(module)

            else: conv_block.append(module)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output, pool_indices = input, []

        for conv_block, pool_layer in zip(self.conv_blocks, self.pool_layers):
            output = conv_block(output)
            output, indices = pool_layer(output)
            pool_indices.append(indices)

        return output, pool_indices
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super(DecoderBlock, self).__init__()

        self.upsample = nn.MaxUnpool2d(2, 2)
        self.decoder = Conv2dBlock(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, num_blocks=depth
        )

    def forward(self, input: torch.Tensor, pooling_indices: torch.Tensor) -> torch.Tensor:
        upsampled_input = self.upsample(input, pooling_indices)
        output = self.decoder(upsampled_input)
        return output
    
class SegNet(nn.Module):
    def __init__(
        self, num_classes: int, decoder_depths: List[int],
        pretrained: bool = True
    ) -> None:
        super(SegNet, self).__init__()

        self.encoder = Encoder(pretrained)
        in_channs = self.encoder.encoder_channels[::-1][:-1]
        out_channs = self.encoder.encoder_channels[::-1][1:-1] + [num_classes + 1]

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_chann, out_chann, depth)
            for in_chann, out_chann, depth in zip(in_channs, out_channs ,decoder_depths)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, pooling_indices = self.encoder(input)
        for i, decoder_block in enumerate(self.decoder_blocks):
            indices = pooling_indices[::-1][i]
            output = decoder_block(output, indices)

        return output
