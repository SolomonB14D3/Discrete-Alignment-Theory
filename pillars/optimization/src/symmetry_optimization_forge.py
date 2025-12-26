import numpy as np
import json
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
PHI = (1 + np.sqrt(5)) / 2

def generate_quasicrystal_3d(n, N_points=120000, cutoff=6.0):
    D = n // 2
    points_D = np.random.uniform(-np.pi, np.pi, (N_points * 5, D))
    theta = 2 * np.pi / n
    proj_matrix = np.zeros((3, D))
    for i in range(3):
        for j in range(D):
            proj_matrix[i, j] = np.cos(i * theta * j)
    points_3d = points_D @ proj_matrix.T
    norms = np.linalg.norm(points_3d, axis=1)
    return points_3d[norms < cutoff][:N_points]

def run_n_scan(n_range, seeds=10):
    summary_stats = {}
    
    for n in tqdm(n_range, desc="Symmetry Forge"):
        c_values = []
        for seed in range(seeds):
            np.random.seed(seed)
            pts = generate_quasicrystal_3d(n)
            # [Placeholder for your TGV/MLS Logic from earlier scripts]
            # ... calculation of delta_0 and R ...
            # C = delta_0 * R
            # c_values.append(C)
        
        summary_stats[int(n)] = {
            "C_mean": float(np.mean(c_values)),
            "C_sem": float(np.std(c_values) / np.sqrt(seeds)) # Standard Error
        }
    
    # Export Data for Archival Proof
    data_path = "../data/n_scan_results.json"
    with open(data_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"\n[SUCCESS] Archival data saved to {data_path}")

# Run for the 'Harmony Nodes'
run_n_scan([6, 10, 12, 18, 24], seeds=500)
