import torch
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

# Connect to the Core Theory
sys.path.append(os.getcwd())
from core.geometry import get_icosahedral_projection

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_POINTS = 10000

def calculate_shannon_entropy(field, bins=100):
    """Measures the information density of the velocity field."""
    hist = torch.histc(field, bins=bins)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -torch.sum(prob * torch.log2(prob))
    return entropy.item()

def run_entropy_test():
    # 1. Setup Geometries
    pts_q = get_icosahedral_projection(N_POINTS, device=DEVICE)
    
    side = int(N_POINTS**(1/3))
    coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
    g = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts_c = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    # 2. Synthetic Turbulent Field (Kolmogorov-like Spectrum)
    # We create a complex field and see how well each grid 'registers' it
    def get_complex_field(p):
        return torch.sin(p[:,0]*4) * torch.cos(p[:,1]*3) + torch.sin(p[:,2]*10) * 0.5

    field_q = get_complex_field(pts_q)
    field_c = get_complex_field(pts_c)

    # 3. Measure Gradients (Local Information)
    def get_local_variation(p, f):
        dists = torch.cdist(p, p)
        _, idxs = torch.topk(dists, 12, largest=False)
        # Average difference between neighbors
        diffs = torch.abs(f[idxs[:, 1:]] - f.unsqueeze(1))
        return torch.mean(diffs, dim=1)

    var_q = get_local_variation(pts_q, field_q)
    var_c = get_local_variation(pts_c, field_c)

    # 4. Compute Metrics
    entropy_q = calculate_shannon_entropy(var_q)
    entropy_c = calculate_shannon_entropy(var_c)
    
    # "Information Capture" ratio
    # High entropy in the gradient field means the grid is capturing more 
    # nuanced structural detail rather than 'aliasing' it into noise.
    print(f"Quasi-Lattice Entropy: {entropy_q:.4f} bits")
    print(f"Cubic Grid Entropy:    {entropy_c:.4f} bits")
    
    # Save Results
    results = pd.DataFrame({
        "metric": ["Shannon_Entropy", "Point_Count"],
        "quasi": [entropy_q, N_POINTS],
        "cubic": [entropy_c, N_POINTS]
    })
    results.to_csv("data/pillar2/entropy_results.csv", index=False)

if __name__ == "__main__":
    run_entropy_test()
