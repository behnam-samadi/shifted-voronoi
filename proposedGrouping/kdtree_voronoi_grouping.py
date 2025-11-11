import numpy as np
import torch
from tqdm import tqdm
import time


class KDTreeNode:
    def __init__(self, points, indices, depth=0, parent=None):
        self.points = points  # torch.Tensor of shape [N, 3]
        self.indices = indices  # torch.Tensor of shape [N]
        self.left = None
        self.right = None
        self.axis = depth % 3
        self.depth = depth
        self.parent = parent  # <-- parent pointer for backtracking
        self.bbox_min = points.min(0).values if points.shape[0] > 0 else None
        self.bbox_max = points.max(0).values if points.shape[0] > 0 else None

    def is_leaf(self):
        return self.left is None and self.right is None

    def midpoint(self):
        if self.is_leaf():
            # Average coordinates of points in this leaf
            return self.points.mean(dim=0)
        else:
            # Fallback: midpoint of bounding box
            return (self.bbox_min + self.bbox_max) / 2


def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    indices = torch.arange(points.shape[0], device=points.device)
    return _build_kdtree(points, indices, threshold, depth, pbar)


def _build_kdtree(points, indices, threshold, depth=0, pbar=None, parent=None):
    node = KDTreeNode(points, indices, depth, parent)

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
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar, parent=node)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar, parent=node)

    return node


def get_leaf_nodes(node):
    """Collect all leaf KDTreeNode objects."""
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


def get_candidate_leaves(leaf, search_iteration):
    """
    Collect up to 2^search_iteration candidate leaf nodes
    by backtracking to parents and checking siblings.
    """
    candidates = set([leaf])
    current = leaf
    for _ in range(search_iteration):
        if current.parent is None:
            break
        parent = current.parent
        siblings = []
        if parent.left is not None:
            siblings.append(parent.left)
        if parent.right is not None:
            siblings.append(parent.right)
        for s in siblings:
            stack = [s]
            while stack:
                n = stack.pop()
                if n.is_leaf():
                    candidates.add(n)
                else:
                    if n.left:
                        stack.append(n.left)
                    if n.right:
                        stack.append(n.right)
        current = parent
    return list(candidates)


def group_by_midpoints(root, points, search_iteration):
    """
    Group points by nearest leaf midpoint among restricted candidate leaves.
    """
    leaves = get_leaf_nodes(root)
    leaf_midpoints = {leaf: leaf.midpoint() for leaf in leaves}

    point_to_group = {}
    for leaf in leaves:
        # Candidate midpoints for this leaf
        candidate_leaves = get_candidate_leaves(leaf, search_iteration)
        candidate_mids = torch.stack([leaf_midpoints[c] for c in candidate_leaves], dim=0)

        # Assign each point in this leaf
        leaf_points = points[leaf.indices]
        dists = torch.cdist(leaf_points.unsqueeze(0), candidate_mids.unsqueeze(0)).squeeze(0)
        nearest_idx = torch.argmin(dists, dim=1)

        for i, idx in enumerate(leaf.indices.tolist()):
            nearest_leaf = candidate_leaves[nearest_idx[i].item()]
            point_to_group[idx] = nearest_leaf

    # Build groups
    groups = {}
    for idx, leaf in point_to_group.items():
        if leaf not in groups:
            groups[leaf] = []
        groups[leaf].append(idx)

    return [np.array(v) for v in groups.values()]


def round_robin(numpy_list):
    """Interleave groups in a round-robin fashion into chunks."""
    max_len = max(len(group) for group in numpy_list)
    chunks = []
    for i in range(max_len):
        chunk = []
        for group in numpy_list:
            idx = group[i % len(group)]  # Wrap around
            chunk.append(idx)
        chunks.append(chunk)
    return chunks


def create_chunks(coord, threshold, search_iteration=4):
    """
    Build KDTree, group points by nearest leaf midpoint among limited candidates,
    then create round-robin chunks.
    """
    current_time = str(time.time())
    points = torch.from_numpy(coord).to("cuda")
    with tqdm(total=10_000, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)

    numpy_list = group_by_midpoints(kdtree_root, points, search_iteration)
    np.save("/home/samadi/research/tests/tree_structures/"+current_time+"_leaves.npy", numpy_list)
    chunks = round_robin(numpy_list)
    chunks = [np.array(chunk) for chunk in chunks]
    np.save("/home/samadi/research/tests/tree_structures/"+current_time+"_chunks.npy", chunks)
    return chunks
