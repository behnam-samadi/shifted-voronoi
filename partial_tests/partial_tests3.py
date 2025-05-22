import numpy as np
import torch

def voxelize_point_cloud(coord, k=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    points = torch.tensor(coord, dtype=torch.float32, device=device)

    # Compute bounding box
    min_bound = points.min(dim=0).values
    max_bound = points.max(dim=0).values
    diag = max_bound - min_bound + 1e-6

    # Estimate voxel resolution based on desired number of buckets
    n_voxels = int(np.ceil(k ** (1/3)))  # cube root to make ~k voxels
    voxel_size = diag / n_voxels

    # Assign voxel indices to each point
    voxel_indices = ((points - min_bound) / voxel_size).floor().to(torch.int32)

    # Flatten 3D voxel index to 1D key
    keys = voxel_indices[:, 0] + voxel_indices[:, 1] * n_voxels + voxel_indices[:, 2] * (n_voxels ** 2)

    # Group points by voxel using unique keys
    unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)

    leaf_nodes = [[] for _ in range(len(unique_keys))]
    for i, idx in enumerate(inverse_indices):
        leaf_nodes[idx.item()].append(points[i])

    # Convert to tensors
    leaf_nodes = [torch.stack(bucket) for bucket in leaf_nodes]

    print(f"Generated {len(leaf_nodes)} voxel-based leaves")
    return leaf_nodes, (min_bound, max_bound), voxel_size


if __name__ == "__main__":
    n = 1000000
    coord = np.random.rand(n, 3) * 100
    k = 45000  # Target approximate number of buckets

    leaves, bounds, voxel_size = voxelize_point_cloud(coord, k)
    print(f"First leaf contains {leaves[0].shape[0]} points")

