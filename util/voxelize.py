import numpy as np
from collections import Sequence
import torch
from torch_geometric.nn import voxel_grid
import time
from lib.pointops2.functions import pointops
from scipy.spatial import cKDTree


def grid_sample(pos, batch_index, size, start=None, return_p2v=True):
    # pos: float [N, 3]
    # batch_szie: long int
    # size: float [3, ]
    # start: float [3, ] / None

    # print("pos.shape: {}, batch.shape: {}".format(pos.shape, batch.shape))
    # print("size: ", size)

    # batch [N, ]
    batch = torch.zeros(pos.shape[0])
    for i in range (1, len(batch_index)):
        batch[batch_index[i-1]:batch_index[i]] = i
        
    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    # print("unique.shape: {}, cluster.shape: {}, counts.shape: {}".format(unique.shape, cluster.shape, counts.shape))

    # input()

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)
    # max_point
    max_point = 48
    if k > max_point:
        counts = torch.where(counts > max_point, max_point, counts)
        p2v_map = p2v_map[:,0:max_point]

    return cluster, p2v_map, counts

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def remove_outliers(coord, outlier_ratio):
    num_points = coord.shape[0]
    coord_temp = torch.FloatTensor(coord).cuda(non_blocking=True)
    offset_temp = torch.cuda.IntTensor([num_points])
    new_offset_temp = torch.cuda.IntTensor([int(num_points*outlier_ratio)])
    outliers = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp)
    outliers = outliers.cpu().numpy()
    return outliers


def modified_fps_with_density(coord, k=10, num_samples=None, alpha=0.5):
    """
    Modified Farthest Point Sampling that also considers local density.

    Parameters:
        coord (np.ndarray): (N, 3) array of 3D coordinates
        k (int): Number of neighbors for local density estimation
        num_samples (int): Number of points to sample (default = same as input)
        alpha (float): Weight for combining distance and inverse density [0,1]
                       Higher alpha favors distance more; lower favors density.

    Returns:
        sampled_indices (np.ndarray): Indices of selected points
    """
    N = coord.shape[0]
    if num_samples is None:
        num_samples = N

    tree = cKDTree(coord)

    # Estimate local density: mean distance to k nearest neighbors
    _, knn_dists = tree.query(coord, k=k + 1)  # includes self at index 0
    local_density = np.mean(knn_dists[:, 1:], axis=1)  # (N,)

    # Normalize density: lower is denser
    normalized_density = (local_density - np.min(local_density)) / (np.ptp(local_density) + 1e-8)

    sampled_indices = []
    remaining = np.arange(N)

    # Initialize with a random point
    first_index = np.random.randint(0, N)
    sampled_indices.append(first_index)

    # Track minimum distance from each point to the selected set
    min_distances = np.linalg.norm(coord - coord[first_index], axis=1)

    for step in range(1, num_samples):
        print(step, ' from ', num_samples)
        # Combine distance and inverse density score
        score = alpha * min_distances - (1 - alpha) * normalized_density
        score[sampled_indices] = -np.inf  # exclude already selected points

        next_index = np.argmax(score)
        sampled_indices.append(next_index)

        # Update min distances
        dist_to_new_point = np.linalg.norm(coord - coord[next_index], axis=1)
        min_distances = np.minimum(min_distances, dist_to_new_point)

    return np.array(sampled_indices)



def proposed_voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    num_clusters = np.unique(key_sort).size
    coord_temp = torch.FloatTensor(coord).cuda(non_blocking=True)
    num_points = coord_temp.shape[0]
    offset_temp = torch.cuda.IntTensor([num_points])
    new_offset_temp = torch.cuda.IntTensor([num_clusters])
    fps_time = -time.time()

    outliers = remove_outliers(coord, 0.01)
    initial_distances = [1e10] * num_points
    outliers = np.array(outliers, dtype=int)
    initial_distances = np.array(initial_distances)
    initial_distances[outliers] = 0
    idx2 = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp, initial_distances)
    #modified_fps = modified_fps_with_density(coord, 10, num_clusters)
    #idx2 = modified_fps


    # Assume coord is (N, 3) and idx2 is (k,) with indices in 0..N
    # Both should already be on GPU (e.g., coord.cuda())
    centroids = coord_temp[idx2.long()]  # (k, 3)

    #distances, indices = pointops.knn_points(coord, centroids, 1)


    offset = torch.tensor([centroids.shape[0]], dtype=torch.int32, device='cuda')
    new_offset = torch.tensor([coord_temp.shape[0]], dtype=torch.int32, device='cuda')
    idx, dist = pointops.KNNQuery.apply(1, centroids, coord_temp, offset, new_offset)
    del centroids
    del coord_temp
    idx = idx.cpu().numpy().ravel()
    fps_time += time.time()
    print('overhead fps time: ', fps_time)
    key_sort = idx

    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count





    pass
def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count


    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count

    '''
    #_, idx = np.unique(key, return_index=True)
    #return idx

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, idx_start, count = np.unique(key_sort, return_counts=True, return_index=True)
    idx_list = np.split(idx_sort, idx_start[1:])
    return idx_list
    '''
