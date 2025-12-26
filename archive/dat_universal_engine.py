import torch
import numpy as np
import pandas as pd
from scipy import signal
import os
from datetime import datetime

# --- CONFIG ---
torch.manual_seed(42)
DEVICE = torch.device("cpu")

class DAT_Universal_Engine:
    def __init__(self, lattice_type='E6'):
        self.lattice_type = lattice_type
        # Calibrated for numerical stability in high-D
        if lattice_type == 'E6':
            self.d, self.k, self.gamma = 6, 0.5, 0.05
        elif lattice_type == 'E8':
            self.d, self.k, self.gamma = 8, 1.2, 0.08
        elif lattice_type == 'Leech':
            self.d, self.k, self.gamma = 24, 4.0, 0.15
            
        self.dt = 0.002 # High precision for high-D stability
        self.lattice = self.generate_lattice()
        self.proj_matrix = self.get_orthonormal_projection()
        self.current_state = self.lattice.clone()
        self.previous_state = self.lattice.clone()

    def generate_lattice(self) -> torch.Tensor:
        if self.lattice_type == 'E6':
            return torch.randn(2442, 6, device=DEVICE) 
        elif self.lattice_type == 'E8':
            roots = []
            for i in range(8):
                for j in range(i + 1, 8):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            v = np.zeros(8); v[i], v[j] = s1, s2; roots.append(v)
            for i in range(256):
                v = np.array([0.5 if b == '0' else -0.5 for b in bin(i)[2:].zfill(8)])
                if np.sum(v / 0.5) % 2 == 0: roots.append(v)
            return torch.tensor(np.array(roots), dtype=torch.float32, device=DEVICE)
        elif self.lattice_type == 'Leech':
            kissing = []
            for i in range(24):
                for j in range(i + 1, 24):
                    for s in [-2, 2]:
                        v = np.zeros(24); v[i], v[j] = s, s; kissing.append(v)
            return torch.tensor(np.array(kissing), dtype=torch.float32, device=DEVICE)

    def get_orthonormal_projection(self) -> torch.Tensor:
        random_mat = torch.randn(self.d, 3, device=DEVICE)
        q, _ = torch.linalg.qr(random_mat) 
        return q.T

    def apply_hamiltonian_dynamics(self, noise_level=0.0):
        # Hamiltonian gradient
        restoring_force = -self.k * (self.current_state - self.lattice)
        damping_force = -self.gamma * (self.current_state - self.previous_state)
        stochastic_force = torch.randn_like(self.current_state) * noise_level
        
        # Update with clamping to prevent numerical explosion (NaNs)
        new_state = self.current_state + (restoring_force + damping_force + stochastic_force) * self.dt
        new_state = torch.clamp(new_state, min=-20.0, max=20.0)
        
        self.previous_state = self.current_state.clone()
        self.current_state = new_state
        return self.current_state @ self.proj_matrix.T

def compute_beta_robust(timeseries: np.ndarray) -> float:
    # Ensure no NaNs or Infs enter the spectral analysis
    if not np.all(np.isfinite(timeseries)) or len(timeseries) < 128:
        return 0.0
    
    f, psd = signal.welch(timeseries, fs=1.0, nperseg=128, detrend='constant')
    psd = psd + 1e-9 
    mask = (f > 0.02) & (f < 0.4)
    log_f, log_psd = np.log10(f[mask]), np.log10(psd[mask])
    slope, _ = np.polyfit(log_f, log_psd, 1)
    return -slope

# --- EXECUTION ---
for lat_type in ['E8', 'Leech']:
    print(f"\n--- DAT 3.2: {lat_type} ---")
    engine = DAT_Universal_Engine(lattice_type=lat_type)
    vorticity_log, beta_log = [], []
    
    TOTAL_STEPS = 6000
    for f in range(TOTAL_STEPS):
        # EXPONENTIAL ANNEALING: noise = MAX * exp(-decay * time)
        if f < 1000:
            noise = 2.0
        else:
            # Slower decay to allow topological alignment
            noise = 2.0 * np.exp(-0.001 * (f - 1000))
        
        if f > 5000: noise = 0.0 # Hard freeze at the end
            
        proj_3d = engine.apply_hamiltonian_dynamics(noise_level=noise)
        v = torch.std(proj_3d).item() 
        vorticity_log.append(v)
        
        if f > 500:
            b = compute_beta_robust(np.array(vorticity_log[-500:]))
        else: b = 0.0
        beta_log.append(b)

        if f % 1000 == 0:
            print(f"Step {f}: Noise={noise:.3f} | β={b:.4f}")

    peak_beta = max(beta_log)
    print(f"RESULT {lat_type}: Peak β ≈ {peak_beta:.3f}")
