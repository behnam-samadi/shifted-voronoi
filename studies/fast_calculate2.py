import os
import numpy as np
from collections import Counter

def redundancy_overlap_metric(chunks):
    all_indices = np.concatenate(chunks)
    count = Counter(all_indices)
    redundancy = np.array(list(count.values()))
    return np.mean(redundancy)

def coverage_efficiency(chunks):
    all_indices = np.concatenate(chunks)
    unique_indices = np.unique(all_indices)
    return len(unique_indices) / len(all_indices)

def analyze_folder(folder_path):
    redundancies = []
    efficiencies = []
    for i in range(0, 67):
        print(i)
        chunks = np.load(os.path.join(folder_path, f"{i}.npy"), allow_pickle=True)
        redundancies.append(redundancy_overlap_metric(chunks))
        efficiencies.append(coverage_efficiency(chunks))
    return np.mean(redundancies), np.mean(efficiencies)

# Analyze both
baseline_redundancy, baseline_eff = analyze_folder("overlap_ratio/baseline")
proposed_redundancy, proposed_eff = analyze_folder("overlap_ratio/proposed")

# Print results
print("🔹 Baseline:")
print(f"  Avg Redundancy: {baseline_redundancy:.3f}")
print(f"  Coverage Efficiency: {baseline_eff:.3f}")

print("🔸 Proposed:")
print(f"  Avg Redundancy: {proposed_redundancy:.3f}")
print(f"  Coverage Efficiency: {proposed_eff:.3f}")
