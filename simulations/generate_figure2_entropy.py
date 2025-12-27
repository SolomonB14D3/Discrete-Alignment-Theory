# scripts/generate_figure2_entropy.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# Connect to the Core Theory
sys.path.append(os.getcwd())
from core.geometry import get_icosahedral_projection

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_POINTS = 10000

def get_variation_data():
    pts_q = get_icosahedral_projection(N_POINTS, device=DEVICE)
    side = int(N_POINTS**(1/3))
    coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
    g = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts_c = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    def get_complex_field(p):
        return torch.sin(p[:,0]*4) * torch.cos(p[:,1]*3) + torch.sin(p[:,2]*10) * 0.5

    def get_local_variation(p, f):
        dists = torch.cdist(p, p)
        _, idxs = torch.topk(dists, 12, largest=False)
        diffs = torch.abs(f[idxs[:, 1:]] - f.unsqueeze(1))
        return torch.mean(diffs, dim=1).cpu().numpy()

    var_q = get_local_variation(pts_q, get_complex_field(pts_q))
    var_c = get_local_variation(pts_c, get_complex_field(pts_c))
    return var_q, var_c

var_q, var_c = get_variation_data()

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

plt.hist(var_q, bins=100, alpha=0.6, color='#FFD700', label='Quasi-Lattice (Ordered)', density=True)
plt.hist(var_c, bins=100, alpha=0.4, color='#FF4500', label='Cubic Grid (Disordered)', density=True)

plt.title("Information Entropy: Geometric Ordering", fontsize=16, pad=20)
plt.xlabel("Local Gradient Magnitude (Variation)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(frameon=False)
plt.grid(alpha=0.1)

plt.savefig('plots/figure2_entropy_order.png', dpi=300)
print("Figure 2 generated: plots/figure2_entropy_order.png")
