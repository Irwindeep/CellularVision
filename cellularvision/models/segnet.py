import torch
import torch.nn as nn

from torchvision import models # type: ignore
from typing import Tuple, List

class Encoder(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super(Encoder, self).__init__()

        encoder = (
            models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1) if pretrained
            else models.vgg11_bn()
        ).features
        encoder_list = list(encoder.children())

        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        self.conv_params = [
            (num_classes + 1, 64, 1), (64, 128, 1), (128, 256, 2),
            (256, 512, 2), (512, 512, 2)
        ]

        conv_block: List[nn.Module] = []
        for i, module in enumerate(encoder_list):
            if isinstance(module, nn.MaxPool2d):
                self.conv_blocks.append(nn.Sequential(*conv_block))
                module.return_indices, conv_block = True, []
                self.pool_layers.append(module)

            else: conv_block.append(module)

        if pretrained:
            for block in self.conv_blocks:
                for params in block.parameters(): params.requires_grad = False

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
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super(SegNet, self).__init__()

        self.encoder = Encoder(num_classes, pretrained)
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