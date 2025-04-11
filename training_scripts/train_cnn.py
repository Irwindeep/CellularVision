import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from cellularvision.dataset import PanNukeSegmentation
from cellularvision.models import SegmentationCNN
from cellularvision.utils import val_epoch, train_epoch

import matplotlib.pyplot as plt
from typing import Tuple, List

plt.rcParams.update({"font.family": "monospace"})
torch.manual_seed(12)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def load_datasets() -> Tuple[PanNukeSegmentation, PanNukeSegmentation]:
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

    train_dataset = PanNukeSegmentation(
        root="pannuke", split="train",
        transform=transform, target_transform=target_transform
    )
    train_size = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = random_split(
        train_dataset,
        [len(train_dataset)-train_size, train_size]
    )

    return train_dataset, val_dataset

def load_model() -> SegmentationCNN:
    model = SegmentationCNN().to(device)

    return model

def train_model(
    model: SegmentationCNN, optim: torch.optim.Optimizer,
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
            torch.save(model.state_dict(), "model_weights/cnn_weights.pth")
        else: count += 1

        if count >= 5:
            print("Stopping early with best model state...")
            model.load_state_dict(torch.load("model_weights/cnn_weights.pth"))
            break

    return train_losses, val_losses

def main() -> None:
    batch_size = 32
    model = load_model()

    epochs, lr = 50, 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = train_model(
        model, optim, loss_fn, batch_size, epochs
    )

    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss", fontweight="bold")
    plt.legend()
    plt.grid()
    plt.savefig("figures/train_losses/cnn.png")

if __name__=="__main__":
    main()
