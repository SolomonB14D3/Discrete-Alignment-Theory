import torch
import numpy as np
import pandas as pd
from scipy import stats
import os

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_GRID = 256
SEEDS = 5

def simulate_n_symmetry(n):
    alignments = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        # ... [Spectral solver logic as previously drafted] ...
        # (Returns alignment for this specific seed)
        alignments.append(np.random.uniform(n/100, n/80)) # Mock for structure
    
    return np.mean(alignments), np.std(alignments)

def recursive_search(n_start, n_end, results, depth=0):
    mid = (n_start + n_end) // 2
    m1, s1 = simulate_n_symmetry(n_start)
    m2, s2 = simulate_n_symmetry(mid)
    m3, s3 = simulate_n_symmetry(n_end)
    
    results[n_start], results[mid], results[n_end] = (m1, s1), (m2, s2), (m3, s3)
    
    # Statistical trigger: if p-value < 0.05, zoom in
    _, p_val = stats.ttest_ind_from_stats(m1, s1, SEEDS, m3, s3, SEEDS)
    if p_val < 0.05 and depth < 4:
        recursive_search(n_start, mid, results, depth+1)
        recursive_search(mid, n_end, results, depth+1)

if __name__ == "__main__":
    os.makedirs("data/pillar3", exist_ok=True)
    data = {}
    recursive_search(1, 10000, data)
    df = pd.DataFrame([{"n": k, "align_mean": v[0], "align_std": v[1]} for k, v in data.items()])
    df.sort_values("n").to_csv("data/pillar3/symmetry_scaling_data.csv", index=False)
    print("Pillar 3: Recursive sweep with p-value branching complete.")
