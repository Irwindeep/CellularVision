import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_blocks: int = 2) -> None:
        super(Conv2dBlock, self).__init__()

        in_channs, out_channs = [in_channels, *[out_channels]*(num_blocks-1)], [out_channels]*num_blocks
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    in_channs[i], out_channs[i], kernel_size=kernel_size,
                    padding=(kernel_size-1)//2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            for i in range(num_blocks)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.blocks(input)
