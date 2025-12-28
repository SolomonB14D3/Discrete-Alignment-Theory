import numpy as np
import matplotlib.pyplot as plt
from dat_core import get_h3_lattice

def simulate_wave_diffusion(lattice, steps=500):
    n_points = len(lattice)
    phi = (1 + np.sqrt(5)) / 2
    sigma = 1 / phi  # 0.618
    
    # Initialize wave packet at the first site
    psi = np.zeros(n_points, dtype=complex)
    psi[0] = 1.0 
    
    # Compute Gaussian-coupled Laplacian
    dist_matrix = np.linalg.norm(lattice[:, None] - lattice[None, :], axis=2)
    coupling = np.exp(-dist_matrix**2 / sigma**2)
    laplacian = coupling - np.diag(np.sum(coupling, axis=1))
    
    ipr_history = []
    for _ in range(steps):
        # Schrödinger-style evolution with normalization (Grok's stability fix)
        psi += -1j * phi * (laplacian @ psi) * 0.01
        psi /= np.linalg.norm(psi) 
        
        # Calculate IPR: Higher values = More localization/shielding
        ipr = np.sum(np.abs(psi)**4) / (np.sum(np.abs(psi)**2)**2)
        ipr_history.append(ipr)
        
    return ipr_history

# --- Setup Fair Comparison ---
# Use 216 points (6x6x6 grid) to avoid sampling artifacts
print("🔬 Benchmarking 216 unique Cubic points vs 216 unique H3 points...")

h3_lattice = get_h3_lattice(n_points=216)
cubic_lattice = np.array(list(np.ndindex(6, 6, 6)), dtype=float)
cubic_lattice -= 2.5 # Grok's centering fix: (0-5 range shifted to -2.5 to 2.5)

# Run simulations
h3_ipr = simulate_wave_diffusion(h3_lattice)
cubic_ipr = simulate_wave_diffusion(cubic_lattice)

# Final Diagnostics
print(f"Final IPR (H3 Shield): {h3_ipr[-1]:.4f}")
print(f"Final IPR (Cubic Conductor): {cubic_ipr[-1]:.4f}")
print(f"DAT Shielding Factor: {h3_ipr[-1]/cubic_ipr[-1]:.2f}x")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(cubic_ipr, label='Cubic Lattice (Standard Aluminum Baseline)', color='gray', linestyle='--')
plt.plot(h3_ipr, label='H3 Manifold (DAT-E6 Shield)', color='gold', linewidth=2.5)
plt.yscale('log')
plt.title("Empirical Hardening: Thermal Localization (IPR) Comparison")
plt.ylabel("Inverse Participation Ratio (Localization Strength)")
plt.xlabel("Interaction Step")
plt.legend()
plt.grid(True, which="both", alpha=0.2)
plt.savefig('docs/benchmarking_localization_v2.png')
print("✅ Results saved to docs/benchmarking_localization_v2.png")
