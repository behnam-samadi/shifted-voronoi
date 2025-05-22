import numpy as np
import torch
from tqdm import tqdm

class OctreeNode:
    def __init__(self, indices, bounds, depth=0):
        self.indices = indices  # Indices into original point array
        self.bounds = bounds    # (min_bound, max_bound)
        self.children = []
        self.depth = depth

    def is_leaf(self):
        return len(self.children) == 0

def compute_child_bounds(parent_bounds, octant_idx):
    min_b, max_b = parent_bounds
    center = (min_b + max_b) / 2
    # Get binary bits from octant index
    dx = (octant_idx >> 0) & 1
    dy = (octant_idx >> 1) & 1
    dz = (octant_idx >> 2) & 1

    # Compute min and max for child box
    child_min = torch.stack([
        min_b[0] if dx == 0 else center[0],
        min_b[1] if dy == 0 else center[1],
        min_b[2] if dz == 0 else center[2],
    ])
    child_max = torch.stack([
        center[0] if dx == 0 else max_b[0],
        center[1] if dy == 0 else max_b[1],
        center[2] if dz == 0 else max_b[2],
    ])
    return (child_min, child_max)

def build_octree(points, indices, bounds, threshold, depth=0, max_depth=16, pbar=None):
    if pbar is not None:
        pbar.update(1)

    node = OctreeNode(indices, bounds, depth)

    if indices.shape[0] <= threshold or depth >= max_depth:
        return node

    pts = points[indices]
    center = (bounds[0] + bounds[1]) / 2

    # Assign each point to one of 8 octants
    octant_ids = (
        (pts[:, 0] > center[0]).long() +
        (pts[:, 1] > center[1]).long() * 2 +
        (pts[:, 2] > center[2]).long() * 4
    )

    for i in range(8):
        mask = (octant_ids == i)
        if mask.any():
            child_indices = indices[mask]
            child_bounds = compute_child_bounds(bounds, i)
            child_node = build_octree(points, child_indices, child_bounds, threshold, depth + 1, max_depth, pbar)
            node.children.append(child_node)

    return node

def collect_leaves(node, leaf_stats):
    if node.is_leaf():
        leaf_stats.append(node.indices.shape[0])
    else:
        for child in node.children:
            collect_leaves(child, leaf_stats)

# ======================= USAGE =======================

np.random.seed(0)
# Load points from file or generate random points
# coord = np.random.rand(100000, 3).astype(np.float32)
coord = np.load('/home/samadi/research/temp/sample_frame.npy')  # or your file path

points = torch.from_numpy(coord).to('cuda')
N = points.shape[0]
indices = torch.arange(N, device=points.device)

# Compute bounds
min_bound = points.min(dim=0).values
max_bound = points.max(dim=0).values

# Estimate a safe upper bound for progress bar
estimated_nodes = 10_000

threshold = 65  # Max points per leaf
with tqdm(total=estimated_nodes, desc="Building Octree") as pbar:
    octree_root = build_octree(points, indices, (min_bound, max_bound), threshold, pbar=pbar)

# Collect stats
leaf_stats = []
collect_leaves(octree_root, leaf_stats)

# Print stats
print(f"Number of leaf boxes: {len(leaf_stats)}")
print(f"Min points in leaf: {min(leaf_stats)}")
print(f"Max points in leaf: {max(leaf_stats)}")
