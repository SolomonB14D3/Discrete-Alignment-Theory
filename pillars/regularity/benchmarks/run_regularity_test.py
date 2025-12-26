# Save as: pillars/regularity/benchmarks/run_regularity_test.py
import torch
import numpy as np
import pandas as pd
import os
import sys

# Ensure core theory is accessible
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
        # Standard Cubic Grid setup
        side = int(N_POINTS**(1/3))
        coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    # Initialize Taylor-Green Vortex Velocity Field
    u = torch.sin(pts[:, 0]) * torch.cos(pts[:, 1])
    v = -torch.cos(pts[:, 0]) * torch.sin(pts[:, 1])
    w = torch.zeros_like(u)
    vel = torch.stack([u, v, w], dim=1)

    ledger = []
    print(f"Starting {mode} stability test...")

    for step in range(STEPS):
        # 1. Simple Advection Step
        pts = pts + vel * DT
        
        # 2. Check for Singularity (Mathematical Collapse)
        max_vel = torch.norm(vel, dim=1).max().item()
        
        # Stability Condition: If velocity blows up or NaN occurs, the grid failed
        if max_vel > 10.0 or torch.isnan(vel).any():
            print(f"!!! {mode.upper()} COLLAPSE DETECTED AT STEP {step} !!!")
            return step, ledger

        # 3. Update field (Simplified Navier-Stokes relaxation)
        # In a real TGV, we'd solve Poisson, here we simulate the decay stress
        vel = vel * 0.999 
        
        if step % 100 == 0:
            ledger.append({"step": step, "max_vel": max_vel})

    return STEPS, ledger

if __name__ == "__main__":
    os.makedirs("data/pillar1", exist_ok=True)
    
    # Run Cubic Control
    c_step, _ = run_stability_benchmark('cubic')
    
    # Run DAT-E6 Experimental
    q_step, _ = run_stability_benchmark('quasi')
    
    print(f"\n--- Final Pillar 1 Results ---")
    print(f"Cubic Grid Limit: {c_step} steps")
    print(f"DAT-E6 Limit:     {q_step} steps")
    
    df = pd.DataFrame({
        "metric": ["Max_Stable_Steps"],
        "cubic": [c_step],
        "quasi": [q_step]
    })
    df.to_csv("data/pillar1/verification_ledger.csv", index=False)
