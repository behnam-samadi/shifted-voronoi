from lib.pointops2.functions import pointops
import torch
import faiss
import numpy as np

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

    coord_temp = torch.FloatTensor(rabbit_point_cloud).cuda(non_blocking=True)
    offset_temp = torch.cuda.IntTensor([coord_temp.shape[0]])
    new_offset_temp = torch.cuda.IntTensor([5])
    initial_distances = [1e10] * coord_temp.shape[0]
    initial_distances[1] = 0
    initial_distances[18] = 0
    idx2 = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp, initial_distances)
    print(idx2.cpu().numpy())
    pass




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
if __name__ == "__main__":
    # Generate a fake 3D point cloud
    coords = np.random.rand(10, 3).astype(np.float32)
    k = 2  # number of clusters

    labels, centers = faiss_kmeans_clustering(coords, k)
    print("Cluster labels shape:", labels.shape)
    print("Cluster centers shape:", centers.shape)

#test_on_furthest_point_sampling()