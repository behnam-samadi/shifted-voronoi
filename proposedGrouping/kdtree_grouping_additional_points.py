import numpy as np
import torch
from tqdm import tqdm
import time

import networkx as nx
from sklearn.neighbors import NearestNeighbors


class KDTreeNode:
    def __init__(self, points, indices, depth=0):
        self.points = points  # torch.Tensor of shape [N, 3]
        self.indices = indices  # torch.Tensor of shape [N]
        self.left = None
        self.right = None
        self.axis = depth % 3
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    indices = torch.arange(points.shape[0], device=points.device)
    return _build_kdtree(points, indices, threshold, depth, pbar)


def _build_kdtree(points, indices, threshold, depth=0, pbar=None):
    node = KDTreeNode(points, indices, depth)

    if points.shape[0] <= threshold:
        return node

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2
    left_points, right_points = points[:median_idx], points[median_idx:]
    left_indices, right_indices = indices[:median_idx], indices[median_idx:]

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node


def collect_leaves(node, leaf_stats):
    if node.is_leaf():
        leaf_stats.append(node.points.shape[0])
    else:
        if node.left is not None:
            collect_leaves(node.left, leaf_stats)
        if node.right is not None:
            collect_leaves(node.right, leaf_stats)


def get_leaf_indices(node):
    leaves = []

    def _collect(n):
        if n.is_leaf():
            leaves.append(n.indices)
        else:
            if n.left is not None:
                _collect(n.left)
            if n.right is not None:
                _collect(n.right)

    _collect(node)
    return leaves


def round_robin(numpy_list, coord, feats):
    """
    Create round-robin chunks. If a leaf has fewer points than the max leaf size,
    instead of repeating indices, append new fake averaged points (coords + feats)
    and reference them by their new indices.
    """
    max_len = max(len(group) for group in numpy_list)
    chunks = []
    new_coords = []
    new_feats = []

    for i in range(max_len):
        chunk = []
        for group in numpy_list:
            if i < len(group):
                idx = group[i]  # real point index
                chunk.append(int(idx))
            else:
                # create fake averaged coord and feat
                mean_coord = np.mean(coord[group], axis=0)
                mean_feat = np.mean(feats[group], axis=0)

                new_idx = coord.shape[0] + len(new_coords)  # index after real points
                new_coords.append(mean_coord)
                new_feats.append(mean_feat)
                chunk.append(new_idx)
        chunks.append(chunk)

    if new_coords:
        coord = np.vstack([coord, np.array(new_coords)])
        feats = np.vstack([feats, np.array(new_feats)])

    return chunks, coord, feats


def proposed_grouping(coord, threshold):
    points = torch.from_numpy(coord).to('cuda')
    rough_max_nodes = 10_000
    with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)
    leaf_indices = get_leaf_indices(kdtree_root)
    shapes = []
    for i in range(len(leaf_indices)):
        shapes.append(leaf_indices[i].shape[0])
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    return numpy_list


def create_chunks(coord, feats, threshold):
    """
    coord: np.ndarray [N, 3]
    feats: np.ndarray [N, F]
    threshold: int
    """
    print("create chunks")
    current_time = str(time.time())
    numpy_list = proposed_grouping(coord, threshold)
    np.save("/home/samadi/research/tests/tree_structures/" + current_time + "_leaves.npy", numpy_list)

    chunks, new_coord, new_feats = round_robin(numpy_list, coord, feats)
    np.save("/home/samadi/research/tests/tree_structures/" + current_time + "_chunks.npy", chunks)

    return chunks, new_coord, new_feats
