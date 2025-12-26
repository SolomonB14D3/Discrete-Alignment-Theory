import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys

# Ensure the root directory is in the path so we can import 'core'
sys.path.append(os.getcwd())
from core.geometry import get_icosahedral_projection

# --- ARCHITECTURAL CONSTANTS ---
RE_NUMBER = 1000.0
SIM_STEPS = 5000
DT = 0.0002 
N_POINTS = 10000
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def compute_gradients(points, field):
    """High-resolution gradient estimation for Advection."""
    dists = torch.cdist(points, points)
    _, idxs = torch.topk(dists, 32, largest=False)
    neighbors = idxs[:, 1:]
    w = 1.0 / (torch.gather(dists, 1, neighbors) + 1e-6)
    w = w / w.sum(dim=1, keepdim=True)
    dx = (w.unsqueeze(2) * (points[neighbors] - points.unsqueeze(1))).sum(dim=1)
    df = (w * (field[neighbors] - field.unsqueeze(1))).sum(dim=1)
    norm_sq = torch.norm(dx, dim=1)**2
    grad = torch.zeros(points.shape[0], 3, device=DEVICE)
    mask = norm_sq > 1e-8
    grad[mask] = (df[mask] / norm_sq[mask]).unsqueeze(1) * dx[mask]
    return grad

def run_simulation(mode='quasi'):
    # Geometry Selection
    if mode == 'quasi':
        # Now importing the Gold Standard geometry from Core
        pts = get_icosahedral_projection(N_POINTS, device=DEVICE)
    else:
        side = int(N_POINTS**(1/3))
        coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    vel = torch.stack([
        torch.sin(pts[:, 0]) * torch.cos(pts[:, 1]) * torch.cos(pts[:, 2]),
        -torch.cos(pts[:, 0]) * torch.sin(pts[:, 1]) * torch.cos(pts[:, 2]),
        0.1 * torch.sin(pts[:, 2])
    ], dim=1)

    history = []
    for step in tqdm(range(SIM_STEPS), desc=f"Forge {mode}"):
        o_max = torch.max(torch.norm(vel, dim=1)).item()
        energy = torch.norm(vel).item()
        if np.isnan(energy) or energy > 1e6: break
        history.append([o_max, energy])

        grads = torch.stack([compute_gradients(pts, vel[:,i]) for i in range(3)], dim=1)
        adv = torch.einsum('nd,ndc->nc', vel, grads)
        
        dists = torch.cdist(pts, pts)
        _, i_top = torch.topk(dists, 14, largest=False)
        lap = (torch.mean(vel[i_top[:, 1:]], dim=1) - vel) / (torch.mean(dists[range(N_POINTS), i_top[:, 1]].unsqueeze(1)**2, dim=1).unsqueeze(1) + 1e-7)
        
        depletion = 1.0 / (max(1.0, o_max / 10.0)**2) if mode == 'quasi' else 1.0
        vel += DT * ((-adv) + (1.0/RE_NUMBER) * lap) * depletion
    return history

if __name__ == "__main__":
    os.makedirs("data/pillar1", exist_ok=True)
    q_h = run_simulation('quasi')
    c_h = run_simulation('cubic')
    
    max_len = max(len(q_h), len(c_h))
    data = [[*(q_h[i] if i < len(q_h) else [np.nan, np.nan]), *(c_h[i] if i < len(c_h) else [np.nan, np.nan])] for i in range(max_len)]
    pd.DataFrame(data, columns=['q_vort', 'q_ener', 'c_vort', 'c_ener']).to_csv("data/pillar1/full_proof_ledger.csv")
    print("\nSimulation complete. Ledger updated in data/pillar1/")
