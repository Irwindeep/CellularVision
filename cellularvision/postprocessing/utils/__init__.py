from .graph import graph_laplacian

import cv2
import numpy as np

from numpy.typing import NDArray

label_colors = np.array([
    [255, 255, 255],
    [0, 128, 255],
    [255, 0, 255],
    [0, 100, 0],
    [208, 187, 148],
    [27, 18, 18]
])

def get_segmentation_contours(
    image: NDArray[np.float32],
    segmentation_mask: NDArray[np.uint8]
) -> NDArray[np.float32]:
    labels = np.unique(segmentation_mask)
    labels = labels[labels != 0]

    for label in labels:
        color = tuple(label_colors[label].tolist())
        label_mask = np.uint8(segmentation_mask == label)

        contours, _ = cv2.findContours(cv2.Mat(label_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, thickness=2)

    return image

__all__ = [
    "get_segmentation_contours",
    "graph_laplacian"
]
