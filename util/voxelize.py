import numpy as np
from collections import Sequence
import torch
from torch_geometric.nn import voxel_grid
import time
from lib.pointops2.functions import pointops
from scipy.spatial import cKDTree




def remove_outliers(coord, outlier_ratio):
    num_points = coord.shape[0]
    coord_temp = torch.FloatTensor(coord).cuda(non_blocking=True)
    offset_temp = torch.cuda.IntTensor([num_points])
    new_offset_temp = torch.cuda.IntTensor([int(num_points*outlier_ratio)])
    outliers = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp)
    outliers = outliers.cpu().numpy()
    return outliers



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

def full_fps_voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    # Only for determining the number of clusters:
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
    initial_distances = np.array([1e10] * num_points)

    num_remaining_points = coord.shape[0]
    idx_sort = []
    while(num_remaining_points>0):
        print(num_remaining_points)
        idx = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp, initial_distances)
        idx = idx.cpu().numpy().ravel()
        idx_sort.append(idx)
        initial_distances[idx] = 0
        num_remaining_points -= num_clusters
        new_offset_temp = torch.cuda.IntTensor([min(num_clusters, num_remaining_points)])
    #count = torch.bincount(idx_sort, minlength=num_clusters)
    count = []
    for segment in idx_sort:
        count.append(len(segment))
    count = np.array(count)
    #count = np.bincount(idx_sort, minlength=num_clusters)
    return idx_sort, count


def full_fps_voxelize_on_gpu(coord, voxel_size=0.05, hash_type='fnv', mode=0):
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
    new_offset_temp = torch.cuda.IntTensor([num_points])
    idx = pointops.furthestsampling(coord_temp, offset_temp, new_offset_temp)
    idx = idx.cpu().numpy().ravel()
    chunk_size = num_clusters
    del coord_temp
    num_chunks = int(np.ceil(len(idx) / chunk_size))

    # Output arrays
    #idx_sort = np.full(num_points, -1, dtype=int)  # Maps each original point to a chunk index
    idx_sort = []
    chunk_counts = np.zeros(num_chunks, dtype=int)  # Stores number of points per chunk

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # Adjust last chunk if it would be too short
        if end > len(idx):
            end = len(idx)
            start = max(end - chunk_size, 0)

        chunk_indices = idx[start:end]
        #idx_sort[i] = chunk_indices
        idx_sort.append(chunk_indices)
        chunk_counts[i] = len(chunk_indices)
    count = chunk_counts
    return idx_sort, count


def grid_sample(pos, batch_index, size, start=None, return_p2v=True):
    # pos: float [N, 3]
    # batch_szie: long int
    # size: float [3, ]
    # start: float [3, ] / None

    # print("pos.shape: {}, batch.shape: {}".format(pos.shape, batch.shape))
    # print("size: ", size)

    # batch [N, ]
    batch = torch.zeros(pos.shape[0])
    for i in range(1, len(batch_index)):
        batch[batch_index[i - 1]:batch_index[i]] = i

    cluster = voxel_grid(pos, batch, size, start=start)  # [N, ]

    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    # print("unique.shape: {}, cluster.shape: {}, counts.shape: {}".format(unique.shape, cluster.shape, counts.shape))

    # input()

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k)  # [n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1)  # [n, k]
    p2v_map[mask] = torch.argsort(cluster)
    # max_point
    max_point = 48
    if k > max_point:
        counts = torch.where(counts > max_point, max_point, counts)
        p2v_map = p2v_map[:, 0:max_point]

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

    '''
    #_, idx = np.unique(key, return_index=True)
    #return idx

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, idx_start, count = np.unique(key_sort, return_counts=True, return_index=True)
    idx_list = np.split(idx_sort, idx_start[1:])
    return idx_list
    '''