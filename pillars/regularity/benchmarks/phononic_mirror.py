import torch
import numpy as np
import pandas as pd
import os
import logging

# 1. Scientific & Pathing Configuration
logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "..", "..", "..", "data", "pillar4")
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Reproducibility Seeds
torch.manual_seed(42)
np.random.seed(42)

# 3. Parameters from Code Reviews
N_POINTS = 10000
NOISE_AMP_REL = 0.05  # Relative to signal RMS
SAMPLE_SIZE = 1000
K_NEIGHBORS = 12

def get_icosahedral_projection(N_points, device=torch.device('cpu')):
    phi = (1 + np.sqrt(5)) / 2
    P = np.array([
        [1, phi, 0, -phi, -1],
        [1, -phi, 0, -phi, 1],
        [0, 1, phi, 1, 0]
    ]) / np.sqrt(10)
    points_5d = np.random.uniform(-np.pi, np.pi, (N_points, 5))
    return torch.tensor(points_5d @ P.T, device=device, dtype=torch.float32)

def compute_ipr(signal, neighbors):
    """
    Inverse Participation Ratio (IPR): Measures wave localization.
    IPR -> 1: Perfectly localized (Phononic Mirror active).
    IPR -> 1/N: Perfectly delocalized (Thermal Leakage).
    """
    local_sq_sum = torch.sum(signal[neighbors]**2, dim=1)
    total_sq_sum = torch.sum(signal**2)
    # Adding epsilon 1e-12 to prevent division by zero
    ipr = torch.mean((local_sq_sum**2) / (total_sq_sum**2 + 1e-12))
    return ipr.item()

def run_thermal_test(mode='quasi'):
    if mode == 'quasi':
        pts = get_icosahedral_projection(N_POINTS, device=DEVICE)
    else:
        side = int(np.ceil(N_POINTS**(1/3)))
        coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    # Signal: Sine wave representing thermal energy flow
    signal = torch.sin(pts[:, 0] * 2)
    signal_rms = torch.std(signal)
    noisy_signal = signal + torch.randn_like(signal, device=DEVICE) * signal_rms * NOISE_AMP_REL

    # Statistical Sampling (Reviewer Fix #4: Removes bias from point ordering)
    indices = torch.randperm(len(pts))[:SAMPLE_SIZE]
    pts_sample = pts[indices]
    signal_sample = noisy_signal[indices]

    # Find k-nearest neighbors for local interference check
    dists = torch.cdist(pts_sample, pts_sample)
    _, idx = torch.topk(dists, K_NEIGHBORS + 1, largest=False)
    idx = idx[:, 1:]  # Exclude self

    return compute_ipr(signal_sample, idx)

if __name__ == "__main__":
    logging.info("Starting Pillar 4: Phononic Mirror / Wave Localization Analysis...")
    
    q_ipr = run_thermal_test('quasi')
    c_ipr = run_thermal_test('cubic')

    # Localization improvement: Higher IPR in Quasi-Lattice = Successful Mirror
    improvement = ((q_ipr - c_ipr) / (c_ipr + 1e-12)) * 100

    print(f"\n--- Pillar 4 Results ---")
    print(f"Quasi-Lattice IPR (Localization): {q_ipr:.8f}")
    print(f"Cubic Grid IPR (Localization):    {c_ipr:.8f}")
    print(f"Phononic Mirror Strength:        {improvement:.2f}% improvement over Cubic")

    res = pd.DataFrame({
        "metric": ["Localization_IPR", "Mirror_Strength_Percent"],
        "quasi": [q_ipr, improvement],
        "cubic": [c_ipr, 0.0]
    })
    
    output_file = os.path.join(DATA_DIR, "thermal_reflection_results.csv")
    res.to_csv(output_file, index=False)
    logging.info(f"Results archived to {output_file}")
