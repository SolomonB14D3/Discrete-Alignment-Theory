import numpy as np
import matplotlib.pyplot as plt
from dat_core import get_h3_lattice

def simulate_steered_signal(steps=600, switch_at=300):
    phi = (1 + np.sqrt(5)) / 2
    
    # 1. Initialize System (Pre-Switch State)
    # Using fixed seed ensures point identity consistency
    lattice_pre = get_h3_lattice(n_points=216, phason_offset=0.0, seed=42)
    n_points = len(lattice_pre)
    
    # Inject localized signal at the first node
    psi = np.zeros(n_points, dtype=complex)
    psi[0] = 1.0  
    
    ipr_history = []
    
    print(f"📡 Launching Topological Router ({n_points} active nodes)...")
    
    for t in range(steps):
        # 2. Trigger Phason Shift
        if t == switch_at:
            print("🎛️  Switch Event: Executing Phason Shift & Wave Mapping...")
            
            # Generate new manifold configuration (90-degree rotation in perp-space)
            lattice_post = get_h3_lattice(n_points=216, phason_offset=np.pi/2, seed=42)
            
            # CRITICAL: Nearest-Neighbor Wavefunction Mapping
            # This prevents the signal from 'teleporting' or vanishing during the switch.
            dist_map = np.linalg.norm(lattice_pre[:, None] - lattice_post[None, :], axis=2)
            nearest = np.argmin(dist_map, axis=1)
            
            psi_mapped = np.zeros(n_points, dtype=complex)
            for i in range(n_points):
                psi_mapped[nearest[i]] += psi[i]
            
            # Renormalize after mapping to conserve energy
            psi = psi_mapped / (np.linalg.norm(psi_mapped) + 1e-10)
            
            # Update physical lattice reference
            lattice_pre = lattice_post

        # 3. Physics: Unitary Evolution
        dist_matrix = np.linalg.norm(lattice_pre[:, None] - lattice_pre[None, :], axis=2)
        # Gaussian coupling representing phonon hopping probability
        coupling = np.exp(-dist_matrix**2 / (0.618**2))
        laplacian = coupling - np.diag(np.sum(coupling, axis=1))
        
        # Time evolution (Schrödinger equation analog)
        psi += -1j * phi * (laplacian @ psi) * 0.01
        
        # Normalization to correct Euler integration drift
        psi /= (np.linalg.norm(psi) + 1e-10)
        
        # 4. Analytics: Inverse Participation Ratio (IPR)
        # High IPR = Signal Locked (Shielding)
        # Low IPR = Signal Moving/Spreading (Routing)
        ipr = np.sum(np.abs(psi)**4) / (np.sum(np.abs(psi)**2)**2)
        ipr_history.append(ipr)

    return ipr_history

# --- Execution ---
try:
    ipr_trace = simulate_steered_signal()

    # --- Visualization ---
    plt.figure(figsize=(10, 6), dpi=150)
    # Use dark background for high contrast
    try:
        plt.style.use('dark_background')
    except:
        print("⚠️ Dark background style not found, using default.")
        
    plt.plot(ipr_trace, color='#FF6BFF', linewidth=2.5, label='Steered H3 Signal')
    plt.axvline(x=300, color='white', linestyle='--', alpha=0.7, label='Phason Switch Event')
    
    plt.title("Pillar 6: Topological Signal Steering via Phason Shift", fontsize=14, fontweight='bold')
    plt.ylabel("Localization (IPR)", fontsize=12)
    plt.xlabel("Interaction Step", fontsize=12)
    plt.grid(True, which='both', alpha=0.2, color='gray')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = 'docs/pillar6_topological_switch.png'
    plt.savefig(output_path)
    print(f"✅ Benchmark complete. Visualization saved to {output_path}")

except Exception as e:
    print(f"❌ Error during simulation: {e}")
