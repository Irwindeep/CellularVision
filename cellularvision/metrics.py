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

        if (seg_sum + target_sum).item() == 0: return torch.nan
        return 2*intersection/(seg_sum + target_sum)
    
class IoUScore(nn.Module):
    def __init__(self) -> None:
        super(IoUScore, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, cls: int) -> torch.Tensor:
        segmentation = torch.argmax(input, dim=1)

        seg_indices, target_indices = (segmentation == cls), (target == cls)
        intersection = (seg_indices & target_indices).float().sum()
        union = (seg_indices | target_indices).float().sum()

        if union.item() == 0: return torch.nan

        return intersection/union
