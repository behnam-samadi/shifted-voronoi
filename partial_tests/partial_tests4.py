import numpy as np
import torch
from tqdm import tqdm

def voxelize_point_cloud(coord, k=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    points = torch.tensor(coord, dtype=torch.float32, device=device)

    # Compute bounding box
    min_bound = points.min(dim=0).values
    max_bound = points.max(dim=0).values
    diag = max_bound - min_bound + 1e-6

    # Choose voxel grid resolution
    n_voxels = int(np.ceil(k ** (1/3)))  # ~cube root of target leaf count
    voxel_size = diag / n_voxels

    # Assign voxel indices
    voxel_indices = ((points - min_bound) / voxel_size).floor().to(torch.int32)
    keys = voxel_indices[:, 0] + voxel_indices[:, 1] * n_voxels + voxel_indices[:, 2] * (n_voxels ** 2)

    # Find unique voxel keys and assign points
    unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)

    leaf_nodes = [[] for _ in range(len(unique_keys))]

    # Use tqdm for progress
    print(f"Assigning {points.shape[0]} points to ~{len(unique_keys)} voxels...")
    for i in tqdm(range(points.shape[0]), desc="Binning points", unit="pt"):
        idx = inverse_indices[i].item()
        leaf_nodes[idx].append(points[i])

    # Convert to tensors
    leaf_nodes = [torch.stack(bucket) for bucket in leaf_nodes]

    print(f"✅ Generated {len(leaf_nodes)} voxel-based leaves")
    return leaf_nodes, (min_bound, max_bound), voxel_size




if __name__ == "__main__":
    coord = np.load('/home/samadi/research/temp/sample_frame.npy')
    
    #coord = np.random.rand(n, 3) * 100
    k = 45000  # Approximate number of leaves

    leaves, bounds, voxel_size = voxelize_point_cloud(coord, k)
    print(f"Leaf 0 contains {leaves[0].shape[0]} points")
