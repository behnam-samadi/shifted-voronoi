import torch


def compute_pca_direction(points):
    """
    points: [num_points, 3] tensor
    returns: principal direction vector (3,)
    """
    mean = points.mean(dim=0)
    centered = points - mean
    cov = centered.T @ centered / (centered.shape[0] - 1)
    #eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals, eigvecs = torch.symeig(cov, eigenvectors=True)
    principal_axis = eigvecs[:, -1]  # top eigenvector
    return principal_axis


def split_cluster_pca(points, indices):
    """
    Split a cluster into two balanced parts along the principal axis.
    points: [N, 3] full point cloud
    indices: indices of points in the cluster to split
    returns: two index tensors
    """
    cluster_points = points[indices]
    direction = compute_pca_direction(cluster_points)
    projections = (cluster_points @ direction).squeeze()
    sorted_indices = indices[projections.argsort()]

    mid = len(sorted_indices) // 2
    return sorted_indices[:mid], sorted_indices[mid:]


def balanced_spatial_bisection(points, num_clusters):
    """
    Perform recursive PCA-based bisection to obtain spatially compact, balanced clusters.
    points: [N, 3] torch tensor on GPU
    num_clusters: int, desired number of clusters
    returns: cluster_labels: [N] tensor of cluster indices
    """
    N = points.shape[0]
    device = points.device
    clusters = [(torch.arange(N, device=device), 0)]  # (indices, label)
    next_label = 1

    while len(clusters) < num_clusters:
        # Select largest cluster to split
        clusters.sort(key=lambda x: -x[0].shape[0])
        largest_cluster, label = clusters.pop(0)

        # Stop if cluster too small to split
        if largest_cluster.shape[0] < 2:
            clusters.append((largest_cluster, label))
            break

        left, right = split_cluster_pca(points, largest_cluster)
        clusters.append((left, label))
        clusters.append((right, next_label))
        next_label += 1

    # Create output label map
    cluster_labels = torch.empty(N, dtype=torch.long, device=device)
    for indices, label in clusters:
        cluster_labels[indices] = label

    return cluster_labels


import torch
import numpy as np
import faiss  # Make sure faiss-gpu is installed

# Generate a synthetic point cloud for demonstration
# Replace with your actual point cloud
np_points = np.random.rand(7000, 3).astype(np.float32)
points = torch.tensor(np_points, device='cuda')  # [700000, 3]


def faiss_gpu_kmeans(points, num_clusters, niter=20):
    """
    Clusters 3D points using FAISS GPU-based k-means.

    Args:
        points: [N, 3] torch tensor on CUDA
        num_clusters: number of desired clusters
        niter: number of k-means iterations

    Returns:
        cluster_labels: [N] tensor of cluster assignments
        counts: [num_clusters] tensor with number of points per cluster
    """
    # Move data to CPU and convert to float32 NumPy (FAISS requirement)
    points_np = points.detach().cpu().numpy().astype('float32')

    # Set up FAISS k-means with GPU support
    d = 3  # dimensionality
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=niter, verbose=True, gpu=False)
    kmeans.train(points_np)

    # Assign points to clusters
    D, I = kmeans.index.search(points_np, 1)  # I is [N, 1]
    cluster_labels = torch.tensor(I.squeeze(), device='cuda', dtype=torch.long)

    # Compute counts per cluster for debugging or analysis
    counts = torch.bincount(cluster_labels, minlength=num_clusters)

    return cluster_labels, counts

# Example point cloud
import numpy as np
import time
# Create a random point cloud
elapsed_time = -time.time()
np_points = np.random.rand(700000, 3).astype(np.float32)
points = torch.tensor(np_points, device='cuda')

# Cluster into 16 spatially-local balanced groups
num_clusters = 4500
cluster_labels = balanced_spatial_bisection(points, num_clusters=num_clusters)
counts = torch.bincount(cluster_labels, minlength=num_clusters)
elapsed_time += time.time()
print(elapsed_time)
# cluster_labels: [10000] tensor with values in [0, 15]