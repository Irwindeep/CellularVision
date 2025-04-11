import numpy as np
from torchvision import transforms

from cellularvision.dataset import PanNukeSegmentation
from cellularvision.postprocessing.utils import get_segmentation_contours

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({"font.family": "monospace"})
np.random.seed(12)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: np.array(x))
])
train_dataset = PanNukeSegmentation(
    root="pannuke", split="train",
    transform=transform, target_transform=transform
)

rows, cols = 4, 6
indices = np.random.randint(0, len(train_dataset), size=rows*cols)

plt.figure(figsize=(12, 9))
plt.suptitle(
    "Training Dataset",
    fontweight="bold", fontsize=20
)

for i in range(rows*cols):
    image, mask = train_dataset[int(indices[i])]
    image_w_seg_contours = get_segmentation_contours(image, mask)

    plt.subplot(rows, cols, i+1)
    plt.imshow(image_w_seg_contours)
    rect = patches.Rectangle((0, 0), 222, 222, edgecolor="black", facecolor="none")
    plt.gca().add_patch(rect)

    plt.axis("off")

plt.tight_layout()
plt.savefig("figures/data_vis.png")
