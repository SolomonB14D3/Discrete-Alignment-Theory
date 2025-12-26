import torch
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from core.geometry import get_icosahedral_projection

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_POINTS = 50000 # Increased for Pro resolution
BINS = 256       # Increased for PDF smoothness

def calculate_shannon_entropy(field):
    hist = torch.histc(field, bins=BINS)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -torch.sum(prob * torch.log2(prob)).item()

def run_master_entropy():
    pts_q = get_icosahedral_projection(N_POINTS, device=DEVICE)
    side = int(N_POINTS**(1/3))
    coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
    g = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts_c = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    field_func = lambda p: torch.sin(p[:,0]*4) * torch.cos(p[:,1]*3) + torch.sin(p[:,2]*10) * 0.5
    
    # Gradient calculation logic
    def get_var(p, f):
        dists = torch.cdist(p, p)
        _, idxs = torch.topk(dists, 12, largest=False)
        return torch.mean(torch.abs(f[idxs[:, 1:]] - f.unsqueeze(1)), dim=1)

    var_q, var_c = get_var(pts_q, field_func(pts_q)), get_var(pts_c, field_func(pts_c))
    e_q, e_c = calculate_shannon_entropy(var_q), calculate_shannon_entropy(var_c)
    
    reduction = ((e_c - e_q) / e_c) * 100
    
    results = pd.DataFrame({
        "metric": ["Shannon_Entropy", "Entropy_Reduction_Pct", "Point_Count"],
        "quasi": [e_q, reduction, N_POINTS],
        "cubic": [e_c, 0.0, N_POINTS]
    })
    os.makedirs("data/pillar2", exist_ok=True)
    results.to_csv("data/pillar2/entropy_results.csv", index=False)
    print(f"Pillar 2: Efficiency Gain of {reduction:.2f}% recorded.")

if __name__ == "__main__":
    run_master_entropy()
