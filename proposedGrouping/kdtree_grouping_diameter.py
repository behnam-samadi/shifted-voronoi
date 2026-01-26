import numpy as np
import torch
from tqdm import tqdm
import time


# =========================
# KD-Tree Node
# =========================
class KDTreeNode:
    def __init__(self, points, indices, depth=0):
        self.points = points          # torch.Tensor [N, 3]
        self.indices = indices        # torch.Tensor [N]
        self.left = None
        self.right = None
        self.axis = depth % 3
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None


# =========================
# KD-Tree Construction
# =========================
def build_kdtree(points, threshold, depth=0, pbar=None):
    indices = torch.arange(points.shape[0], device=points.device)
    return _build_kdtree(points, indices, threshold, depth, pbar)


def _build_kdtree(points, indices, threshold, depth, pbar):
    if pbar is not None:
        pbar.update(1)

    node = KDTreeNode(points, indices, depth)

    if points.shape[0] <= threshold:
        return node

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    mid = len(points) // 2
    left_points, right_points = points[:mid], points[mid:]
    left_indices, right_indices = indices[:mid], indices[mid:]

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(
            left_points, left_indices, threshold, depth + 1, pbar
        )

    if right_points.shape[0] > 0:
        node.right = _build_kdtree(
            right_points, right_indices, threshold, depth + 1, pbar
        )

    return node


# =========================
# Leaf Collection
# =========================
def get_leaf_nodes(node):
    leaves = []

    def _collect(n):
        if n.is_leaf():
            leaves.append(n)
        else:
            if n.left is not None:
                _collect(n.left)
            if n.right is not None:
                _collect(n.right)

    _collect(node)
    return leaves


# =========================
# Leaf Diameter Utilities
# =========================
def leaf_diameter_and_axis(points):
    """
    Diameter approximation using bounding box.
    Returns:
        diameter (float)
        axis (int)  -> dimension causing max diameter
    """
    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values
    ranges = maxs - mins
    diameter = ranges.max()
    axis = ranges.argmax().item()
    return diameter.item(), axis


def split_leaf_equal(points, indices, axis):
    """
    Split leaf into two equal-sized leaves along given axis
    """
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    mid = len(points) // 2

    return (
        points[:mid], indices[:mid],
        points[mid:], indices[mid:]
    )


# =========================
# Adaptive Leaf Refinement
# =========================
def refine_leaves_by_diameter(leaf_nodes, alpha):
    """
    Splits leaves whose diameter > alpha
    """
    refined = []

    for leaf in leaf_nodes:
        diameter, axis = leaf_diameter_and_axis(leaf.points)

        if diameter <= alpha or leaf.points.shape[0] <= 1:
            refined.append(leaf)
        else:
            p1, i1, p2, i2 = split_leaf_equal(
                leaf.points, leaf.indices, axis
            )

            refined.append(KDTreeNode(p1, i1, depth=leaf.depth + 1))
            refined.append(KDTreeNode(p2, i2, depth=leaf.depth + 1))

    return refined


# =========================
# Round-Robin Chunking
# =========================
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


# =========================
# Main Grouping Logic
# =========================
def proposed_grouping(coord, threshold, alpha):
    points = torch.from_numpy(coord).to('cuda')

    with tqdm(total=10_000, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)

    # Collect original leaves
    leaf_nodes = get_leaf_nodes(kdtree_root)

    # Refine oversized leaves
    refined_leaves = refine_leaves_by_diameter(leaf_nodes, alpha)

    # Extract indices
    numpy_list = [leaf.indices.cpu().numpy() for leaf in refined_leaves]
    return numpy_list


# =========================
# Final Chunk Creation
# =========================
def create_chunks(coord, threshold, alpha):
    print("Creating chunks...")
    numpy_list = proposed_grouping(coord, threshold, alpha)

    chunks = round_robin(numpy_list)
    chunks = [np.array(chunk) for chunk in chunks]

    return chunks, numpy_list
