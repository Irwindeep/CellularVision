import torch
import torch.nn as nn

class DiceScore(nn.Module):
    def __init__(self) -> None:
        super(DiceScore, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, cls: int) -> torch.Tensor:
        segmentation = torch.argmax(input, dim=1)
        
        seg_indices, target_indices = (segmentation == cls), (target == cls)
        intersection = (seg_indices & target_indices).float().sum()

        seg_sum = seg_indices.float().sum()
        target_sum = target_indices.float().sum()

        return 2*intersection/(seg_sum + target_sum + 1e-8)
    
class IoUScore(nn.Module):
    def __init__(self) -> None:
        super(IoUScore, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, cls: int) -> torch.Tensor:
        segmentation = torch.argmax(input, dim=1)

        seg_indices, target_indices = (segmentation == cls), (target == cls)
        intersection = (seg_indices & target_indices).float().sum()
        union = (seg_indices | target_indices).float().sum()

        return intersection/(union + 1e-8)
