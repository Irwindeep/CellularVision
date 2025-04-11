import numpy as np

from scipy.sparse.linalg import eigsh # type: ignore
from sklearn.cluster import KMeans # type: ignore
from .utils import graph_laplacian

from typing import Optional
from numpy.typing import NDArray

def spectral_clustering(
    segmentation_output: NDArray[np.float32],
    num_clusters: int, num_eigenvectors: Optional[int] = None,
    gamma: float = 1.0, neighbor_distance: int = 1
) -> NDArray[np.int32]:
    L_sparse = graph_laplacian(
        segmentation_output, gamma, neighbor_distance
    )
    if num_eigenvectors is None:
        num_eigenvectors = num_clusters + 1
    
    _, eigenvectors = eigsh(L_sparse, k=num_eigenvectors, which='SM')
    features = eigenvectors[:, 1:num_clusters]

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(features)

    return labels
