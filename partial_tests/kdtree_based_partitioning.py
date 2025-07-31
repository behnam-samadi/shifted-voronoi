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

from tqdm import tqdm

def _build_kdtree(points, indices, threshold, depth=0, pbar=None):
    n_points = points.shape[0]

    # Stop recursion if <= threshold points or single point (leaf node)
    if n_points <= threshold or n_points == 1:
        if pbar is not None:
            pbar.update(1)
        return KDTreeNode(points, indices, depth)

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = n_points // 2
    median_value = points[median_idx, axis]

    left_mask = points[:, axis] < median_value
    right_mask = points[:, axis] >= median_value

    left_points = points[left_mask]
    left_indices = indices[left_mask]
    right_points = points[right_mask]
    right_indices = indices[right_mask]

    # DEBUG prints
    #print('##########')
    #print('Left points:', left_points.shape)
    #print('Right points:', right_points.shape)
    #print('Median val:', median_value)
    #print('##########')

    # Fallback (should almost never trigger now)
    if left_points.shape[0] == 0 or right_points.shape[0] == 0:
        left_points = points[:median_idx]
        left_indices = indices[:median_idx]
        right_points = points[median_idx:]
        right_indices = indices[median_idx:]
        #print('Fallback triggered')
        #print('Left points after fallback:', left_points.shape)
        #print('Right points after fallback:', right_points.shape)

    node = KDTreeNode(None, None, depth)
    node.axis = axis
    node.split = median_value

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)
    return node




def _build_kdtree________________(points, indices, threshold, depth=0, pbar=None):
    node = KDTreeNode(points=None, indices=None, depth=depth)  # no points in internal nodes

    if points.shape[0] <= threshold:
        # Leaf node: store points and indices
        node.points = points
        node.indices = indices
        if pbar:
            pbar.update(1)
        return node

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2
    median_val = points[median_idx, axis]

    left_points = points[:median_idx]
    left_indices = indices[:median_idx]
    right_points = points[median_idx:]
    right_indices = indices[median_idx:]

    if pbar and depth == 0:
        # initialize pbar total as number of leaves approx
        pbar.total = (points.shape[0] + threshold - 1) // threshold
        pbar.refresh()

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node




def _build_kdtree_________(points, indices, threshold, depth=0, pbar=None):
    n_points = points.shape[0]

    # Stop recursion if <= threshold points or single point
    if n_points <= threshold or n_points == 1:
        return KDTreeNode(points, indices, depth)

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = n_points // 2
    median_value = points[median_idx, axis]

    left_mask = points[:, axis] < median_value
    right_mask = points[:, axis] >= median_value

    left_points = points[left_mask]
    left_indices = indices[left_mask]
    right_points = points[right_mask]
    right_indices = indices[right_mask]

    # DEBUG prints
    print('##########')
    print('Left points:', left_points.shape)
    print('Right points:', right_points.shape)
    print('Median val:', median_value)
    print('##########')

    # Fallback (should almost never trigger now)
    if left_points.shape[0] == 0 or right_points.shape[0] == 0:
        left_points = points[:median_idx]
        left_indices = indices[:median_idx]
        right_points = points[median_idx:]
        right_indices = indices[median_idx:]
        print('Fallback triggered')
        print('Left points after fallback:', left_points.shape)
        print('Right points after fallback:', right_points.shape)

    node = KDTreeNode(None, None, depth)
    node.axis = axis
    node.split = median_value

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node





def _build_kdtree_______(points, indices, threshold, depth=0, pbar=None):
    if points.shape[0] <= threshold:
        return KDTreeNode(points, indices, depth)

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2
    median_value = points[median_idx, axis]

    # Use boolean masks
    left_mask = points[:, axis] < median_value
    right_mask = points[:, axis] >= median_value

    left_points = points[left_mask]
    left_indices = indices[left_mask]
    right_points = points[right_mask]
    right_indices = indices[right_mask]

    # DEBUG prints
    print('##########')
    print('Left points:', left_points.shape)
    print('Right points:', right_points.shape)
    print('Median val:', median_value)
    print('Points[:, axis]:', points[:, axis])
    print('Left mask:', left_mask)
    print('Right mask:', right_mask)
    print('##########')

    # Fallback if left_points is empty
    if left_points.shape[0] == 0 or right_points.shape[0] == 0:
        left_points = points[:median_idx]
        left_indices = indices[:median_idx]
        right_points = points[median_idx:]
        right_indices = indices[median_idx:]

        print('Fallback triggered')
        print('Left points after fallback:', left_points.shape)
        print('Right points after fallback:', right_points.shape)

    node = KDTreeNode(None, None, depth)
    node.axis = axis
    node.split = median_value

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node



def _build_kdtree____(points, indices, threshold, depth=0, pbar=None):
    if points.shape[0] <= threshold:
        return KDTreeNode(points, indices, depth)

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2
    median_value = points[median_idx, axis]

    # Try splitting by coordinate values:
    left_mask = points[:, axis] < median_value
    right_mask = points[:, axis] >= median_value

    left_points = points[left_mask]
    left_indices = indices[left_mask]
    right_points = points[right_mask]
    right_indices = indices[right_mask]

    # If left is empty (all equal coordinates), fallback to splitting by index
    if left_points.shape[0] == 0:
        left_points = points[:median_idx]
        left_indices = indices[:median_idx]
        right_points = points[median_idx:]
        right_indices = indices[median_idx:]

    node = KDTreeNode(None, None, depth)
    node.axis = axis
    node.split = median_value

    print('##########')
    print(left_points.shape)
    print(right_points.shape)
    print('##########')

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node





def _build_kdtree___(points, indices, threshold, depth=0, pbar=None):
    # If number of points less than or equal threshold, create leaf node storing these points
    if points.shape[0] <= threshold:
        return KDTreeNode(points, indices, depth)  # leaf node stores points

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2
    median_value = points[median_idx, axis]  # coordinate along split axis

    # Create internal node WITHOUT points, only store axis and split value (if needed)
    node = KDTreeNode(None, None, depth)  # You can modify KDTreeNode to accept split axis & value

    # Split points:
    # Left child: points with axis < median_value
    left_mask = points[:, axis] < median_value
    left_points = points[left_mask]
    left_indices = indices[left_mask]

    # Right child: points with axis >= median_value (includes median point)
    right_mask = points[:, axis] >= median_value
    right_points = points[right_mask]
    right_indices = indices[right_mask]

    print('##########')
    print(left_points.shape)
    print(right_points.shape)
    print('##########')

    # Build children recursively
    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    # You may want to store split axis and median_value on node for searching later:
    node.axis = axis
    node.split = median_value

    return node



def _build_kdtree__(points, indices, threshold, depth=0, pbar=None):
    node = KDTreeNode(points, indices, depth)

    if points.shape[0] <= threshold:
        return node

    axis = depth % 3
    sorted_idx = points[:, axis].argsort()
    points = points[sorted_idx]
    indices = indices[sorted_idx]

    median_idx = len(points) // 2

    # Include median point in right child by default
    left_points, right_points = points[:median_idx], points[median_idx:]
    left_indices, right_indices = indices[:median_idx], indices[median_idx:]

    # Avoid infinite recursion if right_points size doesn't shrink
    if right_points.shape[0] == points.shape[0]:
        # Put median point in left child instead
        left_points, right_points = points[median_idx:], points[:median_idx]
        left_indices, right_indices = indices[median_idx:], indices[:median_idx]

    print('------------')
    print(left_points.shape)
    print(right_points.shape)
    print('------------')
    print()

    if left_points.shape[0] > 0:
        node.left = _build_kdtree(left_points, left_indices, threshold, depth + 1, pbar)
    if right_points.shape[0] > 0:
        node.right = _build_kdtree(right_points, right_indices, threshold, depth + 1, pbar)

    return node


def _build_kdtree_(points, indices, threshold, depth=0, pbar=None):
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
    rough_max_nodes = 10_000
    with tqdm(total=rough_max_nodes, desc="Building KD-Tree") as pbar:
        kdtree_root = build_kdtree(coord, threshold, pbar=pbar)
    leaf_indices = get_leaf_indices(kdtree_root)
    shapes = []
    for i in range(len(leaf_indices)):
        shapes.append(leaf_indices[i].shape[0])
    numpy_list = [t.cpu().numpy() for t in leaf_indices]
    return numpy_list


def create_chunks(coord, threshold):
    points = torch.from_numpy(coord).to('cuda')
    numpy_list = proposed_grouping(points, threshold)
    chunks = round_robin(numpy_list)
    # Convert list of lists into list of NumPy arrays
    chunks = [np.array(chunk) for chunk in chunks]
    return chunks


def downsample_for_train(coord, threshold):
    numpy_list = proposed_grouping(coord, threshold)
    chunks = round_robin(numpy_list)
    # Convert list of lists into list of NumPy arrays
    chunks = [np.array(chunk) for chunk in chunks]
    return chunks



def estimate_max_k(N, max_leafs, min_k=1, max_k=None):
    """
    Estimates the maximum k such that a KD-tree built with a splitting rule
    (splitting until number of points ≤ k) results in no more than max_leafs leaves.
    Uses binary search over possible k values.

    Parameters:
    - N (int): Total number of points
    - max_leafs (int): Desired max number of leaf nodes
    - min_k (int): Lower bound of search
    - max_k (int): Upper bound of search (optional, defaults to N)

    Returns:
    - int: Estimated max k satisfying the constraint
    """
    if max_k is None:
        max_k = N

    def num_leaves(n, k):
        """Estimate number of leaves in a kd-tree recursively."""
        if n <= k:
            return 1
        left = n // 2
        right = n - left
        return num_leaves(left, k) + num_leaves(right, k)

    low, high = min_k, max_k
    best_k = max_k

    while low <= high:
        mid_k = (low + high) // 2
        leaves = num_leaves(N, mid_k)
        if leaves <= max_leafs:
            best_k = mid_k
            high = mid_k - 1
        else:
            low = mid_k + 1

    return best_k





