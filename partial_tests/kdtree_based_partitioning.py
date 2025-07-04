import numpy as np
import torch
from tqdm import tqdm


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


def create_chunks_from_leaves_(points, leaf_indices, num_chunks):
    """
    Create chunks by selecting one point from each leaf using round-robin/modulo logic.

    Args:
        points: torch.Tensor of shape [N, 3]
        leaf_indices: list of torch.Tensor, each with indices from original point cloud
        num_chunks: int, number of chunks to generate

    Returns:
        List of torch.Tensor, each of shape [num_leaves, 3]
    """
    num_leaves = len(leaf_indices)
    leaf_counters = [0] * num_leaves  # One counter per leaf
    chunks = []

    for _ in range(num_chunks):
        chunk_indices = []
        for leaf_idx, leaf in enumerate(leaf_indices):
            idx_list = leaf.tolist()
            if len(idx_list) == 0:
                continue  # Skip empty leaves (shouldn't happen, but safe)

            counter = leaf_counters[leaf_idx]
            selected_index = idx_list[counter % len(idx_list)]
            chunk_indices.append(selected_index)

            leaf_counters[leaf_idx] += 1  # Advance the round-robin counter

        chunk_tensor = points[torch.tensor(chunk_indices, device=points.device)]
        chunks.append(chunk_tensor)

    return chunks


def create_chunks_from_leaves(points, leaf_indices, num_chunks):
    """
    Create chunks by selecting one point from each leaf using round-robin/modulo logic.

    Args:
        points: torch.Tensor of shape [N, 3]
        leaf_indices: list of torch.Tensor, each with indices from original point cloud
        num_chunks: int, number of chunks to generate

    Returns:
        List of torch.Tensor, each of shape [num_leaves, 3]
    """
    num_leaves = len(leaf_indices)
    leaf_counters = [0] * num_leaves  # One counter per leaf
    chunks = []

    for _ in range(num_chunks):
        chunk_indices = []
        for leaf_idx, leaf in enumerate(leaf_indices):
            idx_list = leaf.tolist()
            if len(idx_list) == 0:
                continue  # Skip empty leaves (shouldn't happen, but safe)

            counter = leaf_counters[leaf_idx]
            selected_index = idx_list[counter % len(idx_list)]
            chunk_indices.append(selected_index)

            leaf_counters[leaf_idx] += 1  # Advance the round-robin counter

        chunk_tensor = points[torch.tensor(chunk_indices, device=points.device)]
        chunks.append(chunk_tensor)

    return chunks


def round_robin(numpy_list):
    # Determine the number of total chunks needed (based on max group size)
    max_len = max(len(group) for group in numpy_list)

    # Round-robin selection
    chunks = []
    for i in range(max_len):
        chunk = []
        for group in numpy_list:
            idx = group[i % len(group)]  # Wrap around using modulo
            chunk.append(idx)
        chunks.append(chunk)
    return chunks

# ======================= USAGE =======================
def usage():
    np.random.seed(0)
    coord = np.load('/home/samadi/research/temp/sample_frame.npy')  # Shape [N, 3]
    print("n: ", coord.shape[0])
    points = torch.from_numpy(coord).to('cuda')

    rough_max_nodes = 10_000
    threshold = 82

    with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)

    leaf_indices = get_leaf_indices(kdtree_root)
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    chunks = round_robin(numpy_list)

    # If you want a single NumPy ndarray (stacked), use:
    #numpy_array = np.stack(numpy_list)
    #leaf_indices = leaf_indices.detach().cpu().numpy()
    #print(leaf_indices.cpu().numpy())
    #create_chunks_from_leaves(points, leaf_indices, len(leaf_indices))

    all_values = torch.cat([t.flatten() for t in leaf_indices]).cpu().numpy()

    # 1. Min and Max
    min_val = all_values.min()
    max_val = all_values.max()

    # 2. Check for repeated values
    unique_vals = set(all_values)
    has_repeats = len(unique_vals) < len(all_values)

    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    print("Repeated values found." if has_repeats else "All values are unique.")

    leaf_stats = [len(idx) for idx in leaf_indices]

    print(f"Number of leaf boxes: {len(leaf_stats)}")
    print(f"Min points in leaf: {min(leaf_stats)}")
    print(f"Max points in leaf: {max(leaf_stats)}")

    # Example: retrieve points in first leaf
    first_leaf_points = points[leaf_indices[0]]
    print(f"First leaf point shape: {first_leaf_points.shape}")


#usage()

def create_chunks_return_list_of_lists(coord, threshold):
    points = torch.from_numpy(coord).to('cuda')

    rough_max_nodes = 10_000
    #threshold = 100

    with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(points, threshold, pbar=pbar)

    leaf_indices = get_leaf_indices(kdtree_root)
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    chunks = round_robin(numpy_list)
    return chunks

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


def create_chunks(coord, threshold):
    numpy_list = proposed_grouping(coord, threshold)
    chunks = round_robin(numpy_list)
    # Convert list of lists into list of NumPy arrays
    chunks = [np.array(chunk) for chunk in chunks]
    return chunks
