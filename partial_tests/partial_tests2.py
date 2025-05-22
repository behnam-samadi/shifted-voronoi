import numpy as np
import torch
from tqdm import tqdm

class OctreeNode:
    def __init__(self, points, bounds, device='cpu'):
        self.points = points
        self.bounds = bounds  # (min, max)
        self.children = []
        self.device = device

    def is_leaf(self):
        return len(self.children) == 0

    def split(self, sample_size=50, min_split_threshold=20):
        if self.points.shape[0] < min_split_threshold:
            return  # Too small to bother splitting

        # Sample points to estimate octants
        sample_size = min(sample_size, self.points.shape[0])
        sample_idx = torch.randperm(self.points.shape[0])[:sample_size]
        sample = self.points[sample_idx]

        min_bound, max_bound = self.bounds
        center = (min_bound + max_bound) / 2

        children_bounds = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    low = torch.tensor([dx, dy, dz], device=self.device) * (center - min_bound) + min_bound
                    high = low + (center - min_bound)
                    children_bounds.append((low, high))

        for bmin, bmax in children_bounds:
            # Apply filtering only on sample for speed
            mask = ((sample >= bmin) & (sample < bmax)).all(dim=1)
            if mask.any():
                # Then apply full mask on full data
                full_mask = ((self.points >= bmin) & (self.points < bmax)).all(dim=1)
                child_points = self.points[full_mask]
                if child_points.shape[0] > 0:
                    self.children.append(OctreeNode(child_points, (bmin, bmax), device=self.device))


def build_octree(coord, k, sample_size=50, min_split_threshold=20,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    points = torch.tensor(coord, dtype=torch.float32, device=device)

    min_bound = points.min(dim=0).values
    max_bound = points.max(dim=0).values

    root = OctreeNode(points, (min_bound, max_bound), device=device)
    leaves = [root]

    pbar = tqdm(total=k+1, desc="Splitting octree leaves (approx)", unit="leaf")

    while len(leaves) <= k:
        # Split largest leaf (heuristic)
        leaves.sort(key=lambda n: n.points.shape[0], reverse=True)
        node_to_split = leaves.pop(0)

        before_split = len(leaves)
        node_to_split.split(sample_size=sample_size, min_split_threshold=min_split_threshold)

        if node_to_split.children:
            leaves.extend(node_to_split.children)
        else:
            leaves.append(node_to_split)  # Could not split
            break

        pbar.n = len(leaves)
        pbar.refresh()

    pbar.close()
    return leaves


# Example
if __name__ == "__main__":
    coord = np.load('/home/samadi/research/temp/sample_frame.npy')
    k = 45000

    leaves = build_octree(coord, k)
    print(f"Generated {len(leaves)} approximate leaf nodes.")
    for i, leaf in enumerate(leaves[:5]):
        print(f"Leaf {i}: {leaf.points.shape[0]} points, bounds: {leaf.bounds}")
