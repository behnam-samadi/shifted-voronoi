import numpy as np
import torch
from tqdm import tqdm
import time
import networkx as nx
from sklearn.neighbors import NearestNeighbors


class KDTreeNode:
    def __init__(self, points, indices, depth=0):
        self.points = points  # torch.Tensor [N, 3]
        self.indices = indices  # torch.Tensor [N]
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


def _safe_eigh(cov):
    """Version-safe symmetric eigen decomposition (for 3×3 covariance)."""
    if hasattr(torch.linalg, "eigh"):
        vals, vecs = torch.linalg.eigh(cov)
    else:
        vals, vecs = torch.symeig(cov, eigenvectors=True)
    return vals, vecs


def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)
    indices = torch.arange(points.shape[0], device=points.device)
    return _build_kdtree(points, indices, threshold, depth, pbar)


def _build_kdtree(points, indices, threshold, depth=0, pbar=None):
    node = KDTreeNode(points, indices, depth)

    # stop if small leaf
    if points.shape[0] <= threshold:
        return node

    # --- PCA-based bisecting rule (efficient) ---
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean

    # covariance matrix
    cov = centered.T @ centered
    eigvals, eigvecs = _safe_eigh(cov)
    principal_axis = eigvecs[:, torch.argmax(eigvals)]

    # project onto axis and split by median
    projections = (centered @ principal_axis).flatten()
    median_val = torch.median(projections)

    left_mask = projections <= median_val
    right_mask = projections > median_val

    left_points, right_points = points[left_mask], points[right_mask]
    left_indices, right_indices = indices[left_mask], indices[right_mask]

    # fallback if degenerate split
    if left_points.shape[0] == 0 or right_points.shape[0] == 0:
        axis = depth % 3
        sorted_idx = points[:, axis].argsort()
        median_idx = len(points) // 2
        left_points, right_points = points[:median_idx], points[median_idx:]
        left_indices, right_indices = indices[:median_idx], indices[median_idx:]

    # recurse
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
    shapes = [li.shape[0] for li in leaf_indices]
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    return numpy_list


def create_chunks(coord, threshold):
    print("create chunks")
    current_time = str(time.time())
    numpy_list = proposed_grouping(coord, threshold)
    chunks = round_robin(numpy_list)
    chunks = [np.array(chunk) for chunk in chunks]
    return chunks, numpy_list
