"""
DAT Pillar 4/6: Phason-Transistor Diagnostic (v3.6 - Enhanced Pool + Stress Fix)
Function: Implements 4D Perpendicular Rotation (Phason Drift) + Pillar 5 Stress Deformation
with expanded unique 4D point pool for richer topology.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import solve_ivp
from scipy.spatial import KDTree
import networkx as nx
import time
import matplotlib

matplotlib.use('Agg')

def project_h3(u_norm):
    """Standard DAT Stereographic Projection from 4D to 3D."""
    return u_norm[:, :3] / (1 - u_norm[:, 3, np.newaxis] + 1e-8)

def get_h3_4d_base(n_points=216, seed=42):
    """Generates base 4D H3 coordinates with significantly expanded pool."""
    np.random.seed(seed)
    phi = (1 + np.sqrt(5)) / 2
    CP3 = list(product([-1, 1], repeat=3))
    fv = [np.array([0.5, phi/2, (phi-1)/2, 0]),
          np.array([(phi-1)/2, 0.5, phi/2, 0]),
          np.array([phi/2, (phi-1)/2, 0.5, 0])]
    
    xx = [f * np.concatenate((np.array(sign), [1])) for f in fv for sign in CP3]
    mm = np.array(xx)
    MM = np.vstack((mm, mm[:, [1,0,3,2]], mm[:, [2,3,0,1]], mm[:, [3,2,1,0]]))
    
    # Expanded phi-shell inflation for much larger unique pool
    all_shells = []
    for k in range(-3, 5):  # Wider range for diversity
        shell = MM * (phi ** k)
        all_shells.append(shell)
    combined = np.vstack(all_shells)
    norms = np.linalg.norm(combined, axis=1, keepdims=True) + 1e-10
    u_norm = combined / norms
    
    # Unique points with higher precision rounding
    unique_4d = np.unique(np.round(u_norm, 10), axis=0)  # Tighter rounding for more uniques
    
    print(f"Generated {len(unique_4d)} unique 4D points.")
    
    if len(unique_4d) < n_points:
        print(f"Warning: Only {len(unique_4d)} unique 4D points available, sampling with replacement.")
        idx = np.random.choice(len(unique_4d), n_points, replace=True)
    else:
        idx = np.random.choice(len(unique_4d), n_points, replace=False)
    return unique_4d[idx]

def get_fair_balanced_lattices(n_points=216, target_beta=1.734, noise_level=0.05, seed=42):
    """Generates H3 and Cubic lattices matched by Mean Nearest Neighbor (MNN)."""
    h3_4d = get_h3_4d_base(n_points, seed)
    h3_3d = project_h3(h3_4d)
    
    h3_3d += np.random.normal(0, noise_level, h3_3d.shape)
    h3_3d *= target_beta / np.mean(np.linalg.norm(h3_3d, axis=1))
    
    tree = KDTree(h3_3d)
    nn_dist_h3 = np.mean(tree.query(h3_3d, k=2)[0][:, 1])
    
    side = int(np.ceil(n_points**(1/3)))
    grid = np.arange(side) * nn_dist_h3
    cube_lat = np.array(list(product(grid, grid, grid)))[:n_points]
    cube_lat -= np.mean(cube_lat, axis=0)
    cube_lat += np.random.normal(0, noise_level, cube_lat.shape)
    
    return h3_4d, h3_3d, cube_lat

def apply_stress_deformation(lattice, stress_tensor):
    """Applies Pillar 5 Stress Deformation to lattice coordinates."""
    deformed_lat = lattice @ stress_tensor.T  # Apply strain matrix to coordinates
    return deformed_lat

def run_simulation_ensemble(lattice_3d, lattice_4d=None, t_max=50, num_starts=20, 
                            phason_shift_at=None, phason_angle=0.1, stress_tensor=None):
    """Unitary evolution with 4D Phason Rotation and Pillar 5 Stress Deformation."""
    n = len(lattice_3d)
    sigma = 0.8
    t_span = np.linspace(0, t_max, 500)
    
    # Apply stress deformation if provided
    current_lat = lattice_3d.copy()
    if stress_tensor is not None:
        current_lat = apply_stress_deformation(current_lat, stress_tensor)
    
    dist = np.linalg.norm(current_lat[:, None] - current_lat[None, :], axis=2)
    coupling = np.exp(-dist**2 / (sigma**2))
    laplacian = coupling - np.diag(np.sum(coupling, axis=1))

    def schrodinger(t, psi):
        return -1j * (laplacian @ psi)

    all_ipr = []
    for i in range(num_starts):
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[np.random.randint(0, n)] = 1.0
        
        if phason_shift_at is not None and lattice_4d is not None:
            # Phase 1: Pre-rotation
            sol_pre = solve_ivp(schrodinger, [0, phason_shift_at], psi0,
                                t_eval=t_span[t_span <= phason_shift_at], method='RK45')
            
            # Phason rotation in 4D
            theta = phason_angle
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta),  np.cos(theta)]])
            u_rot = lattice_4d.copy()
            u_rot[:, 2:4] = u_rot[:, 2:4] @ rot_matrix.T
            rotated_3d = project_h3(u_rot)
            
            # Re-apply stress to rotated lattice
            if stress_tensor is not None:
                rotated_3d = apply_stress_deformation(rotated_3d, stress_tensor)
            
            dist_post = np.linalg.norm(rotated_3d[:, None] - rotated_3d[None, :], axis=2)
            laplacian_post = np.exp(-dist_post**2 / (sigma**2))
            laplacian_post -= np.diag(np.sum(laplacian_post, axis=1))
            
            sol_post = solve_ivp(schrodinger, [phason_shift_at, t_max], sol_pre.y[:, -1],
                                 t_eval=t_span[t_span > phason_shift_at], method='RK45')
            psi_evol = np.vstack([sol_pre.y.T, sol_post.y.T])
        else:
            sol = solve_ivp(schrodinger, [0, t_max], psi0,
                            t_eval=t_span, method='RK45')
            psi_evol = sol.y.T

        all_ipr.append([np.sum(np.abs(p)**4) for p in psi_evol])

    evals = np.sort(np.linalg.eigvalsh(-laplacian))
    return t_span, np.mean(all_ipr, axis=0), np.std(all_ipr, axis=0), evals

# --- Execution ---
print("🚀 Launching Pillar 6 Phason-Drift Diagnostic (v3.6 - Enhanced Pool)...")
N_POINTS = 216
PHASON_ANGLE = np.pi / 4
T_SHIFT = 25.0
STRESS_TENSOR = np.diag([1.1, 1.0, 0.9])  # Anisotropic strain

h3_4d, h3_3d, cube_3d = get_fair_balanced_lattices(n_points=N_POINTS)

# Run H3 with 4D Rotation + Stress
t, h3_m, h3_s, h3_ev = run_simulation_ensemble(h3_3d, h3_4d, phason_shift_at=T_SHIFT, 
                                               phason_angle=PHASON_ANGLE, stress_tensor=STRESS_TENSOR)
# Run Cubic baseline with Stress
_, cb_m, cb_s, cb_ev = run_simulation_ensemble(cube_3d, stress_tensor=STRESS_TENSOR)

# --- Plotting ---
plt.figure(figsize=(10, 6), dpi=150)
plt.style.use('dark_background')
plt.plot(t, h3_m, color='#00FFCC', label='H3 (4D Phason + Pillar 5 Stress)')
plt.fill_between(t, h3_m - h3_s, h3_m + h3_s, color='#00FFCC', alpha=0.15)
plt.plot(t, cb_m, color='#666666', linestyle='--', label='Cubic (Static Control + Stress)')
plt.axvline(x=T_SHIFT, color='white', linestyle=':', label='Phason Shift Event')
plt.title(f"DAT Pillar 6: Topological Switching (Angle={PHASON_ANGLE:.2f} rad)")
plt.ylabel("Phonon Localization (IPR)"); plt.xlabel("Time")
plt.legend(); plt.grid(alpha=0.1)
plt.savefig('pillar6_phason_switch.png')

print(f"✅ Simulation Complete. Phason Shift at t={T_SHIFT} with Pillar 5 Stress Coupling applied.")
print(f"✅ Check 'pillar6_phason_switch.png' for the localization behavior.")
