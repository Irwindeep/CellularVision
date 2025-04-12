import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth # type: ignore

from numpy.typing import NDArray

def mean_shift_segmentation(
    segmentation_output: NDArray[np.float32],
) -> NDArray[np.uint8]:
    num_classes, h, w = segmentation_output.shape

    segmentation_output = segmentation_output.transpose(1, 2, 0)
    segmentation_output = segmentation_output.reshape(-1, num_classes)

    bandwidth = estimate_bandwidth(
        segmentation_output,
        quantile=0.1, n_samples=250
    )
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(segmentation_output)

    cluster_labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    final_cluster_labels = np.array([
        np.argmax(center) for center in cluster_centers
    ])
    final_pixel_labels = final_cluster_labels[cluster_labels]
    segmentation_result = final_pixel_labels.reshape(h, w)

    return segmentation_result
