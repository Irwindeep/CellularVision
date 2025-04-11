import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

def softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)

def region_growing_segmentation(
    logits: NDArray[np.float32],
    seg_init: NDArray,
    seeds: List[Tuple[int, int]], threshold: float
) -> NDArray[np.uint8]:
    logits = logits.transpose(1, 2, 0)
    H, W, _ = logits.shape
    segmentation = np.copy(seg_init)
    visited = np.zeros((H, W), dtype=bool)
    
    for seed in seeds:
        x_seed, y_seed = seed
        seed_label = seg_init[x_seed, y_seed]
        if visited[x_seed, y_seed]:
            continue
        
        queue = [(x_seed, y_seed)]
        visited[x_seed, y_seed] = True
        
        while queue:
            x, y = queue.pop(0)
            current_vector = logits[x, y, :]
            
            for dx, dy in [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny]:
                    neighbor_vector = logits[nx, ny, :]
                    diff_norm = np.linalg.norm(current_vector - neighbor_vector)
                    if diff_norm < threshold:
                        segmentation[nx, ny] = seed_label
                        visited[nx, ny] = True
                        queue.append((nx, ny))
    return segmentation
