from lib.pointops2.functions import pointops
import torch
import faiss
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from tqdm import tqdm
import time

def added_to_test_py():
    # downsample_and_visualize(coord, idx_data[0])

    coord_temp = torch.FloatTensor(coord[::100]).cuda(non_blocking=True)
    offset_temp = torch.cuda.IntTensor([coord_temp.shape[0] - 10])
    new_offset_temp = torch.cuda.IntTensor([160])
    idx2 = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp)
    idx2 = idx2.cpu().numpy()
    downsample_and_visualize(coord_temp, idx2)


def test_on_furthest_point_sampling():
    rabbit_point_cloud = [
        [0.5, 0.2, 0.5],
        [0.52, 0.22, 0.52],
        [0.48, 0.22, 0.48],
        [0.5, 0.24, 0.5],
        [0.5, 0.3, 0.5],
        [0.5, 0.35, 0.5],
        [0.48, 0.35, 0.5],
        [0.52, 0.35, 0.5],
        [0.5, 0.15, 0.5],
        [0.52, 0.15, 0.52],
        [0.48, 0.15, 0.48],
        [0.54, 0.12, 0.54],
        [0.46, 0.12, 0.46],
        [0.5, 0.1, 0.5],
        [0.52, 0.1, 0.52],
        [0.48, 0.1, 0.48],
        [0.5, 0.08, 0.58],
        [0.5, 0.07, 0.6],
        [0.5, 0.09, 0.56],
        [0.52, 0.05, 0.48],
        [0.48, 0.05, 0.48],
        [0.52, 0.05, 0.52],
        [0.48, 0.05, 0.52],
        [0.5, 0.2, 0.52],
        [0.5, 0.2, 0.48],
        [0.5, 0.18, 0.5],
        [0.5, 0.12, 0.52],
        [0.5, 0.12, 0.48],
        [0.5, 0.1, 0.46],
    ]


    rabbit_point_cloud = np.load('/home/samadi/research/temp/sample_frame.npy')
    coord_temp = torch.FloatTensor(rabbit_point_cloud).cuda(non_blocking=True)
    offset_temp = torch.cuda.IntTensor([coord_temp.shape[0]])
    num_samples = rabbit_point_cloud.shape[0]
    new_offset_temp = torch.cuda.IntTensor([num_samples])
    #sample_indices = np.random.choice(rabbit_point_cloud.shape[0], size=num_samples, replace=False)

    fps_time = -time.time()
    idx2 = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp)
    idx2 = idx2.cpu().numpy()
    fps_time += time.time()
    print(fps_time)

    #print(idx2.cpu().numpy())
    selected_indices = np.random.choice(num_samples, size=500, replace=False)
    idx3 = idx2[selected_indices]
    idx4 = idx2[1000:1500]

    sample3 = rabbit_point_cloud[idx3]
    sample4 = rabbit_point_cloud[idx4]


    report_distances(sample3)
    report_distances(sample4)

    sample = rabbit_point_cloud[idx2]
    #sample = sampled_point_cloud


    pass


def report_distances(sample):
    pairwise_distances = pdist(sample)
    # Calculate mean and min pairwise distances
    mean_distance = np.mean(pairwise_distances)
    min_distance = np.min(pairwise_distances)

    tree = cKDTree(sample)

    # Query each point's nearest neighbor (excluding the point itself)
    # k=2 returns the point itself and its nearest neighbor
    distances, _ = tree.query(sample, k=2)

    # distances[:, 0] is zero (distance to itself), distances[:, 1] is the nearest neighbor
    nearest_distances = distances[:, 1]

    # Compute mean and min of nearest-neighbor distances
    mean_nn_distance = np.mean(nearest_distances)
    min_nn_distance = np.min(nearest_distances)

    print("Mean nearest-neighbor distance:", mean_nn_distance)
    print("Min nearest-neighbor distance:", min_nn_distance)


def faiss_kmeans_clustering(coords: np.ndarray, k: int, n_iters: int = 20, verbose: bool = True):
    """
    Perform k-means clustering using FAISS on 3D point cloud data.

    Parameters:
        coords (np.ndarray): (N, 3) point cloud (float32).
        k (int): Number of clusters.
        n_iters (int): Number of iterations for k-means.
        verbose (bool): Print progress info.

    Returns:
        cluster_labels (np.ndarray): (N,) Cluster index for each point.
        centroids (np.ndarray): (k, 3) Cluster centroids.
    """
    assert coords.ndim == 2 and coords.shape[1] == 3, "Input must be (N, 3) array"
    coords = coords.astype(np.float32)
    N, D = coords.shape

    # Initialize FAISS k-means object
    kmeans = faiss.Kmeans(d=D, k=k, niter=n_iters, verbose=verbose, gpu=False)

    # Train k-means on the point cloud
    kmeans.train(coords)

    # Assign cluster labels
    distances, labels = kmeans.index.search(coords, 1)  # 1 nearest cluster
    labels = labels.flatten()

    return labels, kmeans.centroids


# Example usage:
#if __name__ == "__main__":
    # Generate a fake 3D point cloud
#    coords = np.random.rand(10, 3).astype(np.float32)
#    k = 2  # number of clusters

#    labels, centers = faiss_kmeans_clustering(coords, k)
#    print("Cluster labels shape:", labels.shape)
#    print("Cluster centers shape:", centers.shape)

#test_on_furthest_point_sampling()


class OctreeNode:
    def __init__(self, points, bounds, device='cpu'):
        self.points = points
        self.bounds = bounds  # (min, max)
        self.children = []
        self.device = device

    def is_leaf(self):
        return len(self.children) == 0

    def split(self):
        min_bound, max_bound = self.bounds
        center = (min_bound + max_bound) / 2

        children_bounds = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    low = torch.tensor([dx, dy, dz], device=self.device) * (center - min_bound) + min_bound
                    high = low + (center - min_bound)
                    children_bounds.append((low, high))

        for bmin, bmax in children_bounds:
            mask = ((self.points >= bmin) & (self.points < bmax)).all(dim=1)
            child_points = self.points[mask]
            if child_points.shape[0] > 0:
                self.children.append(OctreeNode(child_points, (bmin, bmax), device=self.device))


def build_octree(coord, k, device='cuda' if torch.cuda.is_available() else 'cpu'):
    points = torch.tensor(coord, dtype=torch.float32, device=device)
    min_bound = points.min(dim=0).values
    max_bound = points.max(dim=0).values

    root = OctreeNode(points, (min_bound, max_bound), device=device)
    leaves = [root]

    pbar = tqdm(total=k+1, desc="Splitting octree leaves", unit="leaf")

    while len(leaves) <= k:
        leaves.sort(key=lambda n: n.points.shape[0], reverse=True)
        node_to_split = leaves.pop(0)
        prev_leaf_count = len(leaves)
        node_to_split.split()

        if node_to_split.children:
            leaves.extend(node_to_split.children)
        else:
            leaves.append(node_to_split)
            break

        new_leaf_count = len(leaves)
        pbar.n = new_leaf_count
        pbar.refresh()

    pbar.close()
    return leaves


coord = np.load('/home/samadi/research/temp/sample_frame.npy')
if __name__ == "__main__":
    k = 45000  # Minimum number of leaf nodes
    leaves = build_octree(coord, k)
    print(f"Generated {len(leaves)} leaf nodes.")
    for i, leaf in enumerate(leaves[:5]):  # Print first few leaf stats
        print(f"Leaf {i}: {leaf.points.shape[0]} points, bounds: {leaf.bounds}")
