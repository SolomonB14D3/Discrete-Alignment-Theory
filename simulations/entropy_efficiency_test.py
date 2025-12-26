import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
RE = 1000.0
N_POINTS = 2500
DT = 0.01
NU = 1.0 / RE
N_STEPS = 400
PHASON_TARGET = (1 + np.sqrt(5)) / 2  # The Golden Ratio φ ≈ 1.618034
DELTA0 = (np.sqrt(5) - 1) / 4          # Depletion Constant

def generate_quasi_points(N):
    phi = (1 + np.sqrt(5)) / 2
    P = np.array([[1, phi, 0, -phi, -1], [1, -phi, 0, -phi, 1], [0, 1, phi, 1, 0]]) / np.sqrt(10)
    return np.random.uniform(-np.pi, np.pi, (N, 5)) @ P.T

def generate_cubic_points(N):
    side = int(np.ceil(N**(1/3)))
    x = np.linspace(-np.pi, np.pi, side)
    xv, yv, zv = np.meshgrid(x, x, x)
    return np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)[:N]

def compute_topological_entropy(points, k=13):
    """Measures spatial information density via neighbor distribution."""
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k+1)
    r_k = dists[:, -1]
    densities = 1.0 / (r_k**3 + 1e-12)
    probs = densities / (np.sum(densities) + 1e-12)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def run_resilience_sim(points, name, is_quasi=False):
    print(f"🌀 Simulating Pillar 2 Resilience: {name}...")
    tree = KDTree(points)
    dists, idxs = tree.query(points, k=13)
    weights = 1.0 / (dists[:, 1:]**2 + 1e-8)
    weights /= np.sum(weights, axis=1, keepdims=True)

    velocity = np.zeros((len(points), 3))
    velocity[:, 0] = np.sin(points[:, 0]) # Initial Laminar State

    entropy_h, phason_h = [], []
    beta = PHASON_TARGET if is_quasi else 1.0

    for step in range(N_STEPS):
        if step == 100:
            print(f"  ⚡ SHOCK: Injecting High-Entropy Noise into {name}...")
            velocity += np.random.normal(0, 0.6, velocity.shape)

        neighbor_vels = velocity[idxs[:, 1:]]
        lap = np.sum(weights[:, :, None] * neighbor_vels, axis=1) - velocity
        
        # Physics Update with corrected Syntax and Phason Relaxation
        if is_quasi:
            # Quasi-crystal minimizes phason strain toward Golden Ratio
            beta += 0.005 * (PHASON_TARGET - beta) + np.random.normal(0, 0.002)
            depletion = 1.0 / (1.0 + (np.max(np.abs(lap)) / 2.0)**2)
        else:
            beta += np.random.normal(0, 0.01) # Cubic lattice has no resonance target
            depletion = 1.0

        velocity += DT * depletion * (NU * lap)
        
        entropy_h.append(compute_topological_entropy(points + velocity * 0.1))
        phason_h.append((beta - PHASON_TARGET)**2)

    return entropy_h, phason_h

# Execute Simulation
q_ent, q_phas = run_resilience_sim(generate_quasi_points(N_POINTS), "DAT-E6", is_quasi=True)
c_ent, c_phas = run_resilience_sim(generate_cubic_points(N_POINTS), "Cubic", is_quasi=False)

# Recovery Analysis (Time Constant τ)
def exp_decay(t, a, tau, c): return a * np.exp(-t / tau) + c
t_axis = np.arange(len(q_ent[100:]))
popt_q, _ = curve_fit(exp_decay, t_axis, q_ent[100:], p0=[0.5, 30, q_ent[-1]])
popt_c, _ = curve_fit(exp_decay, t_axis, c_ent[100:], p0=[0.5, 100, c_ent[-1]])

# Output Results
os.makedirs("data", exist_ok=True)
pd.DataFrame({"step": np.arange(N_STEPS), "quasi_H": q_ent, "cubic_H": c_ent}).to_csv("data/ENTROPY_EFFICIENCY_VALIDATION.csv", index=False)

print(f"\n✅ Pillar 2 Simulation Complete.")
print(f"📈 DAT-E6 Recovery τ: {popt_q[1]:.2f} steps")
print(f"📈 Cubic Recovery τ: {popt_c[1]:.2f} steps")
print(f"🚀 Resilience Advantage: {popt_c[1]/popt_q[1]:.2f}x faster self-organization.")
