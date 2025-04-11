import numpy as np

from scipy.sparse import spmatrix # type: ignore
from scipy.sparse.linalg import eigsh # type: ignore
from sklearn.cluster import KMeans # type: ignore

from typing import Optional
from numpy.typing import NDArray

def spectral_clustering(
    L_sparse: spmatrix, num_clusters: int,
    num_eigenvectors: Optional[int] = None
) -> NDArray[np.int32]:
    if num_eigenvectors is None:
        num_eigenvectors = num_clusters + 1
    
    _, eigenvectors = eigsh(L_sparse, k=num_eigenvectors, which='SM')
    features = eigenvectors[:, 1:num_clusters]

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(features)

    return labels
