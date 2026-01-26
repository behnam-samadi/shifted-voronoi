# Updated KD-Tree with A--B Farther-Point Median Split
# -------------------------------------------------------
# Replaces axis-based splitting with:
# 1) pick A = farthest point from arbitrary point p
# 2) pick B = farthest point from A
# 3) direction v = B - A
# 4) project all points onto v
# 5) split by median of projections

import numpy as np
import torch
from tqdm import tqdm
import time
import networkx as nx
from sklearn.neighbors import NearestNeighbors

class KDTreeNode:
    def __init__(self, points, indices, depth=0):
        self.points = points
        self.indices = indices
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


def farthest_point(points, start_point):
    # Compute farthest point from a reference point (single-pass)
    dists = torch.sum((points - start_point) ** 2, dim=1)
    idx = torch.argmax(dists)
    return points[idx]


def ab_direction(points):
    # Step 1: pick arbitrary start point p
    p = points[0]

    # Step 2: farthest from p => A
    A = farthest_point(points, p)

    # Step 3: farthest from A => B
    B = farthest_point(points, A)

    # Step 4: direction vector v
    v = B - A
    v_norm = torch.norm(v)
    if v_norm > 0:
        v = v / v_norm

    return v


def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    indices = torch.arange(points.shape[0], device=points.device)
    return _build_kdtree(points, indices, threshold, depth, pbar)


def _build_kdtree(points, indices, threshold, depth=0, pbar=None):
    node = KDTreeNode(points, indices, depth)

    if points.shape[0] <= threshold:
        return node

    # --- NEW SPLIT METHOD: A--B median projection ---
    v = ab_direction(points)
    proj = points @ v  # projection values

    # Balanced split (median)
    median_idx = points.shape[0] // 2
    sorted_proj_idx = torch.argsort(proj)

    left_idx = sorted_proj_idx[:median_idx]
    right_idx = sorted_proj_idx[median_idx:]

    left_points, right_points = points[left_idx], points[right_idx]
    left_indices, right_indices = indices[left_idx], indices[right_idx]

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


def round_robin(numpy_list):
    max_len = max(len(group) for group in numpy_list)
    chunks = []
    for i in range(max_len):
        chunk = []
        for group in numpy_list:
            idx = group[i % len(group)]
            chunk.append(idx)
        chunks.append(chunk)
    return chunks


def proposed_grouping(coord, threshold):
    points = torch.from_numpy(coord).to('cuda')
    rough_max_nodes = 10_000
    with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)
    leaf_indices = get_leaf_indices(kdtree_root)
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    return numpy_list


def create_chunks(coord, threshold):
    print("create chunks")
    numpy_list = proposed_grouping(coord, threshold)
    chunks = round_robin(numpy_list)
    chunks = [np.array(chunk) for chunk in chunks]
    return chunks, numpy_list
