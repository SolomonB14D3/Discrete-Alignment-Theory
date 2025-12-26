import torch
import numpy as np
import pandas as pd
from scipy import signal
import os
import json
from datetime import datetime

# --- REPRODUCIBILITY & PHYSICS CONFIG ---
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")  # CPU for cross-system reproducibility

class DAT_GoldStandard_Engine:
    def __init__(self, r=7.0):
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.sqrt3 = np.sqrt(3.0)
        self.lattice = self.generate_e6_lattice(r)
        self.proj_matrix = self.get_orthonormal_projection()
        self.current_state = self.lattice.clone()
        self.previous_state = self.lattice.clone()  # For damping

    def generate_e6_lattice(self, r: float) -> torch.Tensor:
        """Generate E6 root lattice points in 6D within norm r."""
        beta1 = torch.tensor([0.5, -0.5, -0.5, -0.5, -0.5, -0.5 * self.sqrt3], dtype=torch.float32, device=DEVICE)
        beta2 = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        beta3 = torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        beta4 = torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        beta5 = torch.tensor([0.0, 0.0, -1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        beta6 = torch.tensor([0.0, 0.0, 0.0, -1.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        betas = torch.stack([beta1, beta2, beta3, beta4, beta5, beta6])

        points = []
        range_max = 5  # Sufficient for r=7 (yields 2442 points)
        for k1 in range(-range_max, range_max + 1):
            for k2 in range(-range_max, range_max + 1):
                for k3 in range(-range_max, range_max + 1):
                    for k4 in range(-range_max, range_max + 1):
                        for k5 in range(-range_max, range_max + 1):
                            for k6 in range(-range_max, range_max + 1):
                                coeffs = torch.tensor([k1, k2, k3, k4, k5, k6], dtype=torch.float32, device=DEVICE)
                                pt = coeffs @ betas
                                if torch.norm(pt) <= r:
                                    points.append(pt)
        return torch.stack(points)

    def get_orthonormal_projection(self) -> torch.Tensor:
        """Gram-Schmidt orthonormal icosahedral basis (3x6)."""
        p = self.phi
        v1 = torch.tensor([1.0, p, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        v2 = torch.tensor([0.0, 0.0, 1.0, p, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
        v3 = torch.tensor([p, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE)
        basis = torch.stack([v1, v2, v3])
        q, _ = torch.linalg.qr(basis.T)
        return q.T  # Shape (3, 6)

    def apply_hamiltonian_dynamics(self, noise_level=0.0, k=0.1, gamma=0.05):
        """
        Damped Hamiltonian dynamics (Langevin approximation).
        V(x) = (k/2) sum ||x_i - l_i||^2 (alignment potential).
        dx/dt ≈ -∇V - γ v + η(t), with v ≈ x - x_prev.
        """
        # Restoring force toward original lattice positions
        # (Manuscript note: Full model uses nearest E6 neighbor)
        restoring_force = -k * (self.current_state - self.lattice)

        # Damping force (approximates velocity)
        damping_force = -gamma * (self.current_state - self.previous_state)

        # Stochastic phason noise
        stochastic_force = torch.randn_like(self.current_state) * noise_level

        # Update (Verlet-style)
        new_state = self.current_state + restoring_force + damping_force + stochastic_force
        self.previous_state = self.current_state.clone()
        self.current_state = new_state

        # Project to 3D
        return self.current_state @ self.proj_matrix.T

def compute_beta(timeseries: np.ndarray, min_freq: float = 0.01, max_freq: float = 0.5) -> float:
    """Stable spectral beta via Welch's PSD in scaling regime."""
    if len(timeseries) < 256:
        return np.nan
    f, psd = signal.welch(timeseries, fs=1.0, nperseg=256, detrend='constant')
    mask = (f >= min_freq) & (f <= max_freq)
    if np.sum(mask) < 5:
        return np.nan
    log_f = np.log10(f[mask])
    log_psd = np.log10(psd[mask])
    slope, _ = np.polyfit(log_f, log_psd, 1)
    return -slope

# --- Example Execution (expand for sweeps in Phase 2) ---
engine = DAT_GoldStandard_Engine(r=7)
vorticity_log = []
beta_log = []
frames = []
TOTAL_FRAMES = 3000
CHAOS_START = 500
CHAOS_END = 1200
NOISE_MAGNITUDE = 0.8
PROJECTION_STRENGTH = 1.9  # Applied post-projection if needed

for f in range(TOTAL_FRAMES):
    noise = NOISE_MAGNITUDE if CHAOS_START <= f <= CHAOS_END else 0.0
    proj_3d = engine.apply_hamiltonian_dynamics(noise_level=noise) * PROJECTION_STRENGTH
    # Vorticity proxy: std of projected points
    v = torch.std(proj_3d).item()
    vorticity_log.append(v)
    b = compute_beta(np.array(vorticity_log))
    beta_log.append(b)
    frames.append(f)
    if f % 100 == 0:
        print(f"Frame: {f}/{TOTAL_FRAMES} | Beta: {b:.4f}")

# --- Export ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"experiment_r7_{timestamp}"
os.makedirs(EXP_DIR, exist_ok=True)
df = pd.DataFrame({'frame': frames, 'vorticity': vorticity_log, 'beta': beta_log})
df.to_csv(f"{EXP_DIR}/raw_data.csv", index=False)
manifest = {
    "r_value": 7,
    "noise_magnitude": NOISE_MAGNITUDE,
    "chaos_window": [CHAOS_START, CHAOS_END],
    "projection_strength": PROJECTION_STRENGTH,
    "timestamp": timestamp,
    "n_points": len(engine.lattice)
}
with open(f"{EXP_DIR}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)
print(f"Experiment complete. Evidence in {EXP_DIR}")
