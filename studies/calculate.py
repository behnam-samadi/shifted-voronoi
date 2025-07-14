import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Directory structure
base_dir = "overlap_ratio"
baseline_dir = os.path.join(base_dir, "baseline")
proposed_dir = os.path.join(base_dir, "proposed")

def compute_internal_overlap_ratio(chunks):
    """
    For a list of chunk index arrays (within a single point cloud),
    compute the average pairwise IoU (intersection over union).
    """
    if len(chunks) < 2:
        return 0.0  # No pairs to compare

    iou_list = []
    for c1, c2 in combinations(chunks, 2):
        s1, s2 = set(c1), set(c2)
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0:
            iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0

# Compute internal overlaps for baseline and proposed
baseline_overlaps = []
proposed_overlaps = []

for i in range(0, 3):  # files 1.npy to 67.npy
    print(i)
    baseline_chunks = np.load(os.path.join(baseline_dir, f"{i}.npy"), allow_pickle=True)
    proposed_chunks = np.load(os.path.join(proposed_dir, f"{i}.npy"), allow_pickle=True)

    baseline_overlaps.append(compute_internal_overlap_ratio(baseline_chunks))
    proposed_overlaps.append(compute_internal_overlap_ratio(proposed_chunks))

# Compute averages
avg_baseline = np.mean(baseline_overlaps)
avg_proposed = np.mean(proposed_overlaps)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(baseline_overlaps, label='Baseline', marker='o')
plt.plot(proposed_overlaps, label='Proposed', marker='x')
plt.axhline(avg_baseline, color='blue', linestyle='--', label=f'Avg Baseline = {avg_baseline:.3f}')
plt.axhline(avg_proposed, color='orange', linestyle='--', label=f'Avg Proposed = {avg_proposed:.3f}')
plt.title("Internal Overlap Ratio per Point Cloud")
plt.xlabel("Point Cloud Index (1 to 67)")
plt.ylabel("Average Pairwise Overlap (IoU)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot for paper
plt.figure(figsize=(6, 6))
plt.boxplot([baseline_overlaps, proposed_overlaps],
            labels=["Baseline", "Proposed"],
            patch_artist=True,
            boxprops=dict(facecolor="lightgray"),
            medianprops=dict(color="red"))
plt.title("Distribution of Internal Chunk Overlaps")
plt.ylabel("Average Pairwise Overlap (IoU)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary output
print(f"🔹 Average Internal Overlap (Baseline): {avg_baseline:.4f}")
print(f"🔸 Average Internal Overlap (Proposed): {avg_proposed:.4f}")

if avg_proposed > avg_baseline:
    print("✅ The proposed method produces chunkings with **more internal overlap**.")
elif avg_proposed < avg_baseline:
    print("ℹ️ The proposed method produces chunkings with **less internal overlap**.")
else:
    print("➖ The proposed and baseline methods have identical average overlap.")
