# simulations/phason_slip_detector.py
# Validates Pillar 3: Phase-Space Scaling & Phason Slip Dynamics
import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree

# --- CONFIGURATION ---
PHASON_TARGET = (1 + np.sqrt(5)) / 2  # Golden Ratio resonance
N_WINDOWS = 7                        # Spatial correlation windows
SLIP_THRESHOLD = 0.15                # Critical strain threshold
SIZES = [500, 1000, 2000, 4000]      # System sizes for scaling analysis
N_REPEATS = 3                        # Statistical repeats

def generate_quasi_points(N):
    phi = (1 + np.sqrt(5)) / 2
    P = np.array([
        [1, phi, 0, -phi, -1],
        [1, -phi, 0, -phi, 1],
        [0, 1, phi, 1, 0]
    ]) / np.sqrt(10)
    points_5d = np.random.uniform(-np.pi, np.pi, (N, 5))
    return points_5d @ P.T

def generate_cubic_points(N):
    side = int(np.ceil(N**(1/3)))
    x = np.linspace(-np.pi, np.pi, side)
    xv, yv, zv = np.meshgrid(x, x, x)
    points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)[:N]
    return points

def compute_local_strain(points, window_size=N_WINDOWS):
    tree = KDTree(points)
    dists, _ = tree.query(points, k=window_size + 1)
    neighbor_dists = dists[:, 1:]
    ideal_spacing = 1.0  # Normalized
    local_density = window_size / np.mean(neighbor_dists, axis=1)
    strain = np.abs(local_density - ideal_spacing) / ideal_spacing
    return strain

def analyze_scaling_laws():
    print("📏 Validating Pillar 3: Phason Slip Scaling Laws...")
    results = []
    
    for N in SIZES:
        q_slips, c_slips = [], []
        for _ in range(N_REPEATS):
            # DAT-E6
            q_pts = generate_quasi_points(N)
            q_strain = compute_local_strain(q_pts)
            q_slips.append(np.sum(q_strain > SLIP_THRESHOLD))
            
            # Cubic
            c_pts = generate_cubic_points(N)
            c_strain = compute_local_strain(c_pts)
            c_slips.append(np.sum(c_strain > SLIP_THRESHOLD))
            
        results.append({
            "N": N,
            "quasi_slips_avg": np.mean(q_slips),
            "cubic_slips_avg": np.mean(c_slips)
        })

    df = pd.DataFrame(results)
    
    # Calculate Scaling Exponents (alpha)
    alpha_q = np.polyfit(np.log(df['N']), np.log(df['quasi_slips_avg'] + 1e-8), 1)[0]
    alpha_c = np.polyfit(np.log(df['N']), np.log(df['cubic_slips_avg'] + 1e-8), 1)[0]
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/PHASON_SLIP_SCALING.csv", index=False)
    
    print(f"✅ Analysis Complete.")
    print(f"📊 Quasi Exponent: {alpha_q:.2f} (Sub-linear)")
    print(f"📊 Cubic Exponent: {alpha_c:.2f} (Linear/Super-linear)")
    return alpha_q, alpha_c

if __name__ == "__main__":
    analyze_scaling_laws()
