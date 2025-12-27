import numpy as np
import matplotlib.pyplot as plt
import time
import os
from itertools import product

# --- PILLAR 4 PARAMETERS ---
N_POINTS = 1000
TARGET_BETA = 1.734
PHONON_FREQ = 1.618  # The "Golden" resonant frequency
TIME_STEPS = 1000    # Duration of wave propagation
DELTA_QUANT = 0.618  # Coupling distance based on the Golden Ratio
COUPLING_STRENGTH = 0.01

def generate_validated_lattice():
    """Recreates the validated H3 lattice from Pillar 2 with broadcasting fix."""
    phi = (1 + np.sqrt(5)) / 2
    CP3 = list(product([-1, 1], repeat=3))
    fv1 = np.array([0.5, phi/2, (phi-1)/2, 0])
    fv2 = np.array([(phi-1)/2, 0.5, phi/2, 0])
    fv3 = np.array([phi/2, (phi-1)/2, 0.5, 0])
    xx = []
    for sign in CP3:
        s = np.concatenate((np.array(sign), np.array([1])))
        xx.append(fv1 * s)
    for sign in CP3:
        s = np.concatenate((np.array(sign), np.array([1])))
        xx.append(fv2 * s)
    for sign in CP3:
        s = np.concatenate((np.array(sign), np.array([1])))
        xx.append(fv3 * s)
    
    mm = np.array(xx)
    MM = np.vstack((mm, mm[:, [1, 0, 3, 2]], 
                    mm[:, [2, 3, 0, 1]], mm[:, [3, 2, 1, 0]]))
    
    # 4D to 3D Stereographic Projection
    norms = np.linalg.norm(MM, axis=1)
    U_norm = MM / norms[:, np.newaxis]
    
    # Corrected broadcasting: (96, 3) / (96, 1)
    proj = U_norm[:, :3] / (1 - U_norm[:, 3, np.newaxis] + 1e-8)
    unique_proj = np.unique(np.round(proj, 10), axis=0)
    
    # Scale to validated Beta
    current_mean = np.mean(np.linalg.norm(unique_proj, axis=1))
    scale = TARGET_BETA / current_mean
    h3_pool = unique_proj * scale
    
    # Filter to the validated shell
    norms_h3 = np.linalg.norm(h3_pool, axis=1)
    h3_pool = h3_pool[(norms_h3 > 1.0) & (norms_h3 < 2.5)]
    
    # Deterministic selection for 1000 points
    np.random.seed(42)
    indices = np.random.randint(0, len(h3_pool), N_POINTS)
    return h3_pool[indices]

def simulate_phonon_scattering(lattice):
    """Simulates wave propagation and measures thermal localization (IPR)."""
    n_points = len(lattice)
    psi = np.zeros(n_points, dtype=complex)
    
    # Inject thermal pulse at the 'left' edge (minimum X coordinate)
    source_idx = np.argmin(lattice[:, 0]) 
    psi[source_idx] = 1.0 + 0j
    
    # Calculate coupling (hopping probability) between scattering centers
    dist_matrix = np.linalg.norm(lattice[:, None] - lattice[None, :], axis=2)
    # Exponential decay of coupling based on Delta Quant (0.618)
    coupling = np.exp(-dist_matrix**2 / (DELTA_QUANT**2))
    
    ipr_history = []
    
    print("🌊 Simulating wave propagation...")
    for t in range(TIME_STEPS):
        # Hamiltonian evolution (Schrodinger-style scattering)
        # The Laplacian represents the network connectivity of the H3 points
        laplacian = coupling @ psi - np.sum(coupling, axis=1) * psi
        psi += -1j * PHONON_FREQ * laplacian * COUPLING_STRENGTH
        
        # Calculate Inverse Participation Ratio (IPR)
        # IPR = sum(|psi|^4) / (sum(|psi|^2))^2
        # If heat is trapped in 1 atom, IPR = 1. If spread over N, IPR = 1/N.
        prob_density = np.abs(psi)**2
        norm_factor = np.sum(prob_density)**2
        if norm_factor > 1e-10:
            ipr = np.sum(prob_density**2) / norm_factor
        else:
            ipr = 0
        ipr_history.append(ipr)
        
        if t % 200 == 0:
            print(f" Step {t}: IPR = {ipr:.6f}")
            
    return psi, ipr_history

# --- EXECUTION ---
start_time = time.time()
print("🛡️ Initializing Pillar 4: Phononic Mirror Validation...")

# 1. Generate the labyrinth
lattice = generate_validated_lattice()
print(f"✅ Labyrinth loaded: {len(lattice)} vertices in H3 configuration.")

# 2. Run the Wave Simulation
final_wave, ipr_history = simulate_phonon_scattering(lattice)
end_time = time.time()

# 3. Final Metrics
print("\n--- PILLAR 4 RESULTS ---")
print(f"Total Compute Time: {end_time - start_time:.4f}s")
print(f"Final Localization Index (IPR): {ipr_history[-1]:.6f}")

# 4. Visualization
os.makedirs('docs', exist_ok=True)
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Phonon Density Heatmap
ax1 = fig.add_subplot(121, projection='3d')
intensity = np.abs(final_wave)**2
# Normalize color for visibility
scatter = ax1.scatter(lattice[:,0], lattice[:,1], lattice[:,2], 
                      c=intensity, cmap='hot', s=15, alpha=0.7)
ax1.set_title("Phonon Mirror: Energy Density Map")
fig.colorbar(scatter, ax=ax1, label='Heat Intensity')

# Subplot 2: IPR Trend (Proof of Trapping)
ax2 = fig.add_subplot(122)
ax2.plot(ipr_history, color='cyan', linewidth=2)
ax2.set_title("Thermal Localization Trend")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("IPR (Localization)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/pillar4_thermal_shield_v1.png')
print(f"\n✅ Analysis saved to docs/pillar4_thermal_shield_v1.png")
