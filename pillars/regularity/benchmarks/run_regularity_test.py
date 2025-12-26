import torch
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from core.geometry import get_icosahedral_projection

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
STEPS = 5000
N_POINTS = 10000
DT = 0.001

def run_stability_benchmark(mode='quasi'):
    if mode == 'quasi':
        pts = get_icosahedral_projection(N_POINTS, device=DEVICE)
    else:
        side = int(N_POINTS**(1/3))
        coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    u = torch.sin(pts[:, 0]) * torch.cos(pts[:, 1])
    v = -torch.cos(pts[:, 0]) * torch.sin(pts[:, 1])
    vel = torch.stack([u, v, torch.zeros_like(u)], dim=1)

    trace = []
    for step in range(STEPS):
        pts = pts + vel * DT
        max_vel = torch.norm(vel, dim=1).max().item()
        
        # Log stability trace every 50 steps for the "Divergence Plot"
        if step % 50 == 0:
            trace.append({"step": step, "max_vel": max_vel, "mode": mode})

        if max_vel > 10.0 or torch.isnan(vel).any():
            break
        
        vel = vel * 0.999 # Simulated viscous decay
        
    return trace

if __name__ == "__main__":
    os.makedirs("data/pillar1", exist_ok=True)
    full_trace = run_stability_benchmark('cubic') + run_stability_benchmark('quasi')
    pd.DataFrame(full_trace).to_csv("data/pillar1/stability_trace.csv", index=False)
    print("Pillar 1: Stability trace generated for Divergence Curve analysis.")
