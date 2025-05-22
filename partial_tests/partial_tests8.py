import numpy as np
import torch
from tqdm import tqdm

class KDTreeNode:
    def __init__(self, points, depth=0):
        self.points = points  # torch.Tensor of shape [N, 3]
        self.left = None
        self.right = None
        self.axis = depth % 3
        self.depth = depth

    def is_leaf(self):
        return self.left is None and self.right is None

def build_kdtree(points, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    node = KDTreeNode(points, depth)

    if points.shape[0] <= threshold:
        return node

    axis = depth % 3
    sorted_points = points[points[:, axis].argsort()]
    median_idx = len(sorted_points) // 2

    left_points = sorted_points[:median_idx]
    right_points = sorted_points[median_idx:]

    if left_points.shape[0] > 0:
        node.left = build_kdtree(left_points, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = build_kdtree(right_points, threshold, depth + 1, pbar)

    return node

def collect_leaves(node, leaf_stats):
    if node.is_leaf():
        leaf_stats.append(node.points.shape[0])
    else:
        if node.left is not None:
            collect_leaves(node.left, leaf_stats)
        if node.right is not None:
            collect_leaves(node.right, leaf_stats)

# ======================= USAGE =======================

np.random.seed(0)
coord = np.load('/home/samadi/research/temp/sample_frame.npy')  # Shape [N, 3]
points = torch.from_numpy(coord).to('cuda')

# Estimate max number of nodes (very rough for tqdm)
rough_max_nodes = 10_000

threshold = 100
with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
    kdtree_root = build_kdtree(points, threshold, pbar=pbar)

leaf_stats = []
collect_leaves(kdtree_root, leaf_stats)

print(f"Number of leaf boxes: {len(leaf_stats)}")
print(f"Min points in leaf: {min(leaf_stats)}")
print(f"Max points in leaf: {max(leaf_stats)}")
