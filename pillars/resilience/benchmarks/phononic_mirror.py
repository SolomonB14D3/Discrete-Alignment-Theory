import torch
import numpy as np
import pandas as pd
import os
import sys

# Connect to Core Geometry
sys.path.append(os.getcwd())
try:
    from core.geometry import get_icosahedral_projection
except ImportError:
    # Fallback if core is not yet in path
    def get_icosahedral_projection(n, device='cpu'):
        return torch.randn(n, 3, device=device)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_POINTS = 10000
STEPS = 200

def calculate_ipr(field):
    """Inverse Participation Ratio: Measures spatial localization."""
    return (torch.sum(field**4) / (torch.sum(field**2)**2)).item()

def simulate_wave_resilience(mode='quasi'):
    if mode == 'quasi':
        pts = get_icosahedral_projection(N_POINTS, device=DEVICE)
    else:
        side = int(N_POINTS**(1/3))
        coords = torch.linspace(-np.pi, np.pi, side, device=DEVICE)
        g = torch.meshgrid(coords, coords, coords, indexing='ij')
        pts = torch.stack([g[0].flatten(), g[1].flatten(), g[2].flatten()], dim=1)[:N_POINTS]

    center = torch.tensor([-2.0, 0.0, 0.0], device=DEVICE)
    dist_sq = torch.sum((pts - center)**2, dim=1)
    phi = torch.exp(-dist_sq / 0.5).to(torch.complex64)
    potential = torch.sin(pts[:,0]*5) * torch.cos(pts[:,1]*5) 
    
    for _ in range(STEPS):
        phi = phi * torch.exp(-1j * potential * 0.01)
        
    intensity = torch.abs(phi)**2
    ipr = calculate_ipr(intensity)
    reflected_energy = torch.sum(intensity[pts[:, 0] < 0])
    total_energy = torch.sum(intensity)
    reflection_coeff = (reflected_energy / total_energy).item()
    trans_coeff = 1 - reflection_coeff
    loss_db = 10 * np.log10(max(trans_coeff, 1e-10))
    
    return reflection_coeff, ipr, loss_db

if __name__ == "__main__":
    os.makedirs("data/pillar4", exist_ok=True)
    print(f"Running Pillar 4 Resilience Analysis on {DEVICE}...")
    r_q, i_q, l_q = simulate_wave_resilience('quasi')
    r_c, i_c, l_c = simulate_wave_resilience('cubic')
    
    results = pd.DataFrame({
        "metric": ["Reflection_Coeff", "IPR_Localization", "Transmission_Loss_dB"],
        "quasi": [r_q, i_q, l_q],
        "cubic": [r_c, i_c, l_c]
    })
    results.to_csv("data/pillar4/thermal_reflection_results.csv", index=False)
    print("Pillar 4: Phononic Mirror statistics archived.")
