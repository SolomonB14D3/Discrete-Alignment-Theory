import torch
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# Settings for High-Res Discovery
N_GRID = 256 # Doubled from your NumPy version thanks to MPS speed
STEPS = 1500
RE = 1000.0
DT = 0.01
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def simulate_point(n):
    """MPS-accelerated 2D spectral solver with n-fold symmetry enforcement."""
    # Setup spectral grid on GPU
    kx = torch.fft.fftfreq(N_GRID, d=2*np.pi/N_GRID, device=DEVICE)
    KX, KY = torch.meshgrid(kx, kx, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0,0] = 1e-12

    # Symmetry Mask
    theta = torch.atan2(KY, KX)
    theta_mod = theta % (2 * np.pi / n)
    symmetry_mask = torch.exp(-50 * (theta_mod - np.pi / n)**2)

    # Initial Vorticity (Kolmogorov -5/3 Spectrum)
    vort_hat = torch.randn(N_GRID, N_GRID, dtype=torch.complex64, device=DEVICE)
    vort_hat *= torch.where(K2 > 1, K2**(-5/6), 0.0)
    vort_hat *= symmetry_mask

    for step in range(STEPS):
        # Velocity from vorticity
        u_hat = 1j * KY * vort_hat / K2
        v_hat = -1j * KX * vort_hat / K2
        
        # Nonlinear advection (Pseudospectral)
        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real
        omega = torch.fft.ifft2(vort_hat).real
        
        # Gradient calculation
        grad_y, grad_x = torch.gradient(omega)
        nonlinear = torch.fft.fft2(u * grad_x + v * grad_y)
        
        # Update (Viscous dissipation + nonlinearity)
        vort_hat = vort_hat * torch.exp(-K2 * DT / RE) - DT * nonlinear
        
        if step % 20 == 0:
            vort_hat *= symmetry_mask

    # Final Metrics
    vort_final = torch.fft.ifft2(vort_hat).real
    delta_0 = torch.max(torch.abs(vort_final)).item()
    R = torch.mean(torch.sqrt(K2[K2 > 1])).item()
    
    # Alignment Score (Information Ordering)
    grad_y, grad_x = torch.gradient(vort_final)
    theta_flat = torch.atan2(grad_y, grad_x).flatten().cpu().numpy()
    hist, _ = np.histogram(theta_flat, bins=int(n*4), range=(-np.pi, np.pi))
    alignment = np.max(hist) / (np.mean(hist) + 1e-12)

    return {'C': delta_0 * R, 'Align': alignment}

def find_change_points(n_start, n_end, step_size, results={}):
    """Recursive search for Phase Transitions in the symmetry space."""
    n_vals = np.arange(n_start, n_end + step_size, step_size)
    n_vals = [n for n in n_vals if n not in results]
    
    for n in tqdm(n_vals, desc=f"Scanning n={n_start}-{n_end}"):
        # Run 3 seeds for statistical stability
        seeds = [simulate_point(n) for _ in range(3)]
        results[n] = {
            'C_mean': np.mean([s['C'] for s in seeds]),
            'Align_mean': np.mean([s['Align'] for s in seeds])
        }

    # Recursive logic: if there is a big jump in Alignment, drill down
    if step_size > 10:
        aligns = [results[n]['Align_mean'] for n in sorted(results.keys()) if n_start <= n <= n_end]
        if np.std(aligns) > 0.5: # Threshold for 'Interest'
            mid = (n_start + n_end) // 2
            find_change_points(n_start, mid, step_size // 2, results)
            find_change_points(mid, n_end, step_size // 2, results)
    return results

if __name__ == "__main__":
    os.makedirs("data/pillar3", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Focus the search on the critical 1-200 range where phason flips occur
    # then jump to 10,000 to see long-tail scaling.
    final_data = find_change_points(1, 1000, 100)
    
    # Save Data
    n_vals = sorted(final_data.keys())
    df = pd.DataFrame([{ "n": n, **final_data[n] } for n in n_vals])
    df.to_csv("data/pillar3/symmetry_scaling_data.csv", index=False)
    
    # Generate Figure 3
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df['n'], df['C_mean'], 'o-', color='#00FFCC')
    ax1.set_title("Pillar 3: Non-Monotonic Scaling Law")
    ax1.set_ylabel("Stability Coefficient (C)")
    ax1.set_xscale('log')
    
    ax2.plot(df['n'], df['Align_mean'], 's-', color='#FFD700')
    ax2.set_ylabel("Alignment Score (Resilience)")
    ax2.set_xlabel("Symmetry Order (n)")
    ax2.set_xscale('log')
    
    plt.savefig("plots/figure3_scaling_law.png")
    print(f"Pillar 3 Complete. Data saved to data/pillar3/ and plots/figure3_scaling_law.png")
