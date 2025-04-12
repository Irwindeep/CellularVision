import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from cellularvision.functional import BackboneWithFPN

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from cellularvision.dataset import PanNukeSegmentation
from cellularvision.utils import val_epoch, train_epoch

import matplotlib.pyplot as plt
from typing import Tuple, List, Any, Dict

plt.rcParams.update({"font.family": "monospace"})
torch.manual_seed(12)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class MaskRCNNDataset(PanNukeSegmentation):
    def __init__(self, root, split, transform = None, target_transform = None):
        super().__init__(root, split, transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        instances, categories = sample["instances"], sample["categories"]
        masks, boxes, labels = [], [], []
        width, height = image.size
        for instance, category in zip(instances, categories):
            mask_np = np.array(instance)
            pos = np.where(mask_np == 1)
            if len(pos[0]) == 0:
                continue
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_np)
            labels.append(category + 1)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.tensor([index]),
            "area": torch.as_tensor([(xmax - xmin) * (ymax - ymin) for xmin, ymin, xmax, ymax in boxes], dtype=torch.float32),
        }
        if self.transform:
            image = self.transform(image)
        return image, target

def load_datasets() -> Tuple[MaskRCNNDataset, MaskRCNNDataset]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(
            lambda x: torch.as_tensor(np.array(x), dtype=torch.long)
        )
    ])

    train_dataset = MaskRCNNDataset(
        root="pannuke", split="train",
        transform=transform, target_transform=target_transform
    )
    train_size = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = random_split(
        train_dataset,
        [len(train_dataset)-train_size, train_size]
    )

    return train_dataset, val_dataset

def load_model(pretrained: bool = True) -> MaskRCNN:
    weights = (
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained
        else None
    )
    resnet18 = torchvision.models.resnet18(weights=weights)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    backbone = IntermediateLayerGetter(resnet18, return_layers=return_layers)
    in_channels_list = [64, 128, 256, 512]
    out_channels = 128
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels)

    custom_backbone = BackboneWithFPN(backbone, fpn)

    rpn_anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    num_classes = 6
    model = MaskRCNN(
        backbone=custom_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    ).to(device)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 128, num_classes)

    return model

def train_model(
    model: MaskRCNN, optim: torch.optim.Optimizer,
    loss_fn: nn.Module, batch_size: int, epochs: int
) -> Tuple[List[float], List[float]]:
    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss, count, n = float("inf"), 0, len(str(epochs))

    for epoch in range(1, epochs+1):
        desc = f"Epoch [{epoch:0{n}d}/{epochs}]"
        train_loss = train_epoch(model, train_loader, loss_fn, optim, desc)
        val_loss = val_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, count = val_loss, 0
            torch.save(model.state_dict(), "model_weights/pspnet_weights.pth")
        else: count += 1

        if count >= 5:
            print("Stopping early with best model state...")
            model.load_state_dict(torch.load("model_weights/pspnet_weights.pth"))
            break

    return train_losses, val_losses

def main() -> None:
    batch_size = 16
    model = load_model()

    epochs, lr = 20, 5e-3
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = train_model(
        model, optim, loss_fn, batch_size, epochs
    )

if __name__=="__main__":
    main()
