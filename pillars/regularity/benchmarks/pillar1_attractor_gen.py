
# pillars/regularity/benchmarks/pillar1_attractor_gen.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
Re = 1000.0
T_max = 1.0
dt = 0.0002
N_points = 10000
nu = 1.0 / Re
N_steps = int(T_max / dt)
k_neighbors = 32

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Device: {device} | Generating Phase-Space Attractor Data")

def compute_gradient_diagnostic(points, field, k=k_neighbors):
    dists = torch.cdist(points, points)
    _, idxs = torch.topk(dists, k+1, largest=False, sorted=True)
    neighbors = idxs[:, 1:]
    w = 1.0 / (torch.gather(dists, 1, neighbors) + 1e-6)
    w = w / w.sum(dim=1, keepdim=True)
    dx = (w.unsqueeze(2) * (points[neighbors] - points.unsqueeze(1))).sum(dim=1)
    df = (w * (field[neighbors] - field.unsqueeze(1))).sum(dim=1)
    norm_sq = torch.norm(dx, dim=1)**2
    grad = torch.zeros(points.shape[0], 3, device=device, dtype=torch.float32)
    mask = norm_sq > 1e-8
    grad[mask] = (df[mask] / norm_sq[mask]).unsqueeze(1) * dx[mask]
    limit = 50.0
    return torch.clamp(grad, -limit, limit)

def run_simulation(lattice_type):
    # --- INITIALIZATION ---
    if lattice_type == 'quasi':
        np.random.seed(42)
        phi = (1 + np.sqrt(5)) / 2
        P = torch.tensor([[1, phi, 0, -phi, -1], [1, -phi, 0, -phi, 1], [0, 1, phi, 1, 0]], 
                         dtype=torch.float32, device=device) / np.sqrt(10)
        pts = torch.tensor(np.random.uniform(-np.pi, np.pi, (N_points, 5)), dtype=torch.float32, device=device) @ P.T
        for _ in range(10): # Relaxation
            dists = torch.cdist(pts, pts)
            _, idx = torch.topk(dists, 2, largest=False); nn_idx = idx[:, 1]
            pts += 0.05 * (pts - pts[nn_idx]) / (torch.norm(pts - pts[nn_idx], dim=1, keepdim=True) + 1e-6)
    else:
        side = int(np.ceil(N_points**(1/3)))
        coords = torch.linspace(-np.pi, np.pi, side, dtype=torch.float32, device=device)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_points]

    vel = torch.stack([torch.sin(pts[:, 0]) * torch.cos(pts[:, 1]) * torch.cos(pts[:, 2]),
                       -torch.cos(pts[:, 0]) * torch.sin(pts[:, 1]) * torch.cos(pts[:, 2]),
                       0.01 * torch.sin(pts[:, 2])], dim=1).to(torch.float32)
    
    vort_hist = []
    energy_hist = []
    
    for step in tqdm(range(N_steps), desc=f"Forge {lattice_type}"):
        # Energy Capture (L2 Norm of velocity field)
        energy_hist.append(torch.norm(vel).item())
        
        # Vorticity Calculation
        dists = torch.cdist(pts, pts)
        _, idxs = torch.topk(dists, k_neighbors+1, largest=False)
        w_v = 1.0 / (torch.gather(dists, 1, idxs[:, 1:]) + 1e-6)
        w_v = w_v / w_v.sum(dim=1, keepdim=True)
        dv = vel[idxs[:, 1:]] - vel.unsqueeze(1)
        r = pts[idxs[:, 1:]] - pts.unsqueeze(1)
        omega = (w_v.unsqueeze(2) * torch.linalg.cross(dv, r / (torch.norm(r, dim=2, keepdim=True) + 1e-7), dim=2)).sum(dim=1)
        o_max = torch.max(torch.norm(omega, dim=1)).item()
        vort_hist.append(o_max)
        
        # Physics Update
        grads = torch.stack([compute_gradient_diagnostic(pts, vel[:,i]) for i in range(3)], dim=1)
        adv = torch.einsum('nd,ndc->nc', vel, grads)
        d_top, i_top = torch.topk(dists, 14, largest=False)
        lap = (torch.mean(vel[i_top[:, 1:]], dim=1) - vel) / (torch.mean(d_top[:, 1:]**2, dim=1).unsqueeze(1) + 1e-7)
        depletion = 1.0 / (max(1.0, o_max / 12.0)**2)
        vel += dt * depletion * (-adv + nu * lap)
        vel = torch.clamp(vel, -10.0, 10.0)
    
    return vort_hist, energy_hist

# --- EXECUTE & PLOT ---
q_v, q_e = run_simulation('quasi')
c_v, c_e = run_simulation('cubic')

plt.figure(figsize=(10, 8))
plt.plot(q_e, q_v, color='gold', label='Quasi-Lattice Attractor (12-fold)', alpha=0.7)
plt.plot(c_e, c_v, color='black', label='Cubic Grid Attractor', alpha=0.3, ls='--')
plt.title("Pillar 1: Phase-Space Attractor Analysis\nKinetic Energy (L2) vs. Peak Vorticity")
plt.xlabel("Global Kinetic Energy")
plt.ylabel("Peak Vorticity")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("plots/pillar1_attractor.png")
logger.info("Attractor generated: plots/pillar1_attractor.png")
