import numpy as np
from scipy.sparse import coo_matrix, spmatrix, diags # type: ignore

from numpy.typing import NDArray

def graph_laplacian(
    segmentation_output: NDArray[np.float32],
    gamma: float = 1.0, neighbor_distance: int = 1
) -> spmatrix:
    _, H, W = segmentation_output.shape
    N = H * W
    features = segmentation_output.transpose(1, 2, 0)

    rows_all, cols_all, data_all = [], [], []
    
    offsets = [
        (di, dj)
        for di in range(-neighbor_distance, neighbor_distance + 1)
        for dj in range(-neighbor_distance, neighbor_distance + 1)
        if not (di == 0 and dj == 0)
    ]
    
    for di, dj in offsets:
        i_min, i_max = max(0, -di), H - max(0, di)
        j_min, j_max = max(0, -dj), W - max(0, dj)
        
        i_vals, j_vals = np.arange(i_min, i_max), np.arange(j_min, j_max)
        I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
        I_n, J_n = I + di, J + dj
        
        feats, feats_n = features[I, J], features[I_n, J_n]
        sq_norm = np.sum((feats - feats_n) ** 2, axis=-1)

        idx = (I * W + J).ravel()
        idx_neighbor = (I_n * W + J_n).ravel()
        
        weights = np.exp(-gamma * sq_norm).ravel()
        
        rows_all.append(idx)
        cols_all.append(idx_neighbor)
        data_all.append(weights)
    
    rows_all = np.concatenate(rows_all)
    cols_all = np.concatenate(cols_all)
    data_all = np.concatenate(data_all)
    
    W_sparse = coo_matrix((data_all, (rows_all, cols_all)), shape=(N, N))
    
    degrees = np.array(W_sparse.sum(axis=1)).flatten()
    D_sparse = diags(degrees)
    
    L_sparse = D_sparse - W_sparse
    return L_sparse
