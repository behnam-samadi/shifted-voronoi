import numpy as np
import torch
from tqdm import tqdm

class OctreeNode:
    def __init__(self, points, bounds, depth=0):
        self.points = points
        self.bounds = bounds  # (min_bound, max_bound)
        self.children = []
        self.depth = depth

    def is_leaf(self):
        return len(self.children) == 0

def split_box(bounds):
    min_b, max_b = bounds
    center = (min_b + max_b) / 2
    grid = torch.meshgrid(
        torch.tensor([0.0, 1.0], device=min_b.device),
        torch.tensor([0.0, 1.0], device=min_b.device),
        torch.tensor([0.0, 1.0], device=min_b.device),
    )
    corners = torch.stack(grid, dim=-1).reshape(-1, 3)

    sub_boxes = []
    for corner in corners:
        box_min = min_b + corner * (center - min_b)
        box_max = center + corner * (max_b - center)
        sub_boxes.append((box_min, box_max))
    return sub_boxes

def build_octree(points, bounds, threshold, depth=0, pbar=None):
    if pbar is not None:
        pbar.update(1)

    node = OctreeNode(points, bounds, depth)
    if points.shape[0] <= threshold:
        return node

    children_boxes = split_box(bounds)
    for box_min, box_max in children_boxes:
        mask = ((points >= box_min) & (points <= box_max)).all(dim=1)
        child_points = points[mask]
        if child_points.shape[0] > 0:
            child_node = build_octree(child_points, (box_min, box_max), threshold, depth + 1, pbar)
            node.children.append(child_node)
    return node

def collect_leaves(node, leaf_stats):
    if node.is_leaf():
        leaf_stats.append(node.points.shape[0])
    else:
        for child in node.children:
            collect_leaves(child, leaf_stats)

# ======================= USAGE =======================

# Load your actual point cloud
np.random.seed(0)
# coord = np.random.rand(10000, 3).astype(np.float32)
coord = np.load('/home/samadi/research/temp/sample_frame.npy')

# Transfer to GPU
points = torch.from_numpy(coord).to('cuda')

# Compute bounds
min_bound = points.min(dim=0).values
max_bound = points.max(dim=0).values

# Estimate upper bound on total nodes (loose, for tqdm)
rough_max_nodes = 10_000  # You can adjust based on expected depth/threshold

# Build octree with progress bar
threshold = 30
with tqdm(total=rough_max_nodes, desc="Building Octree") as pbar:
    octree_root = build_octree(points, (min_bound, max_bound), threshold, pbar=pbar)

# Collect stats
leaf_stats = []
collect_leaves(octree_root, leaf_stats)

# Print stats
print(f"Number of leaf boxes: {len(leaf_stats)}")
print(f"Min points in leaf: {min(leaf_stats)}")
print(f"Max points in leaf: {max(leaf_stats)}")
