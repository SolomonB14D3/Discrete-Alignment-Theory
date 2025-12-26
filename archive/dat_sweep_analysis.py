import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch  # <--- ADD THIS LINE
from dat_gold_standard import DAT_GoldStandard_Engine, compute_beta

def run_sweep():
    # Experimental Magnitudes
    strains = [0.2, 0.5, 0.8]
    colors = ['#27ae60', '#f39c12', '#c0392b']
    labels = ['Low Strain (σ=0.2)', 'Med Strain (σ=0.5)', 'High Strain (σ=0.8)']
    
    plt.figure(figsize=(12, 7))
    
    results_summary = []

    for sigma, color, label in zip(strains, colors, labels):
        print(f"Running Sweep: {label}...")
        engine = DAT_GoldStandard_Engine(r=7)
        v_log = []
        b_log = []
        
        # Simulation Loop (Matching Manuscript Parameters)
        for f in range(3000):
            noise = sigma if 500 <= f <= 1200 else 0.0
            proj = engine.apply_hamiltonian_dynamics(noise_level=noise)
            
            v_log.append(torch.std(proj).item())
            b_log.append(compute_beta(np.array(v_log)))
        
        # Plotting the signature
        plt.plot(b_log, color=color, label=label, lw=1.5, alpha=0.8)
        
        # Capture Statistics for the Manuscript Table
        results_summary.append({
            "Strain": sigma,
            "Min_Beta_Chaos": min(b_log[500:1200]),
            "Peak_Beta_Recovery": max(b_log[1200:1500]),
            "Final_Stasis_Beta": b_log[-1]
        })

    # Formatting for Publication
    plt.axvspan(500, 1200, color='gray', alpha=0.1, label='Chaos Phase')
    plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.3, label='Pink Noise Target (β=1.0)')
    plt.title("DAT 2.0: Multi-Trial Phase Transition Analysis", fontsize=14)
    plt.xlabel("Temporal Evolution (Frames)", fontsize=12)
    plt.ylabel("Spectral Beta (β)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.savefig("DAT_Sweep_Results.png", dpi=300)
    
    # Save the Table
    pd.DataFrame(results_summary).to_csv("sweep_statistics.csv", index=False)
    print("Sweep complete. Plot saved as 'DAT_Sweep_Results.png'")

if __name__ == "__main__":
    run_sweep()
