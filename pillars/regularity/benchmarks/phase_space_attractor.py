# pillars/regularity/benchmarks/phase_space_attractor.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def generate_attractor_plot(q_history, c_history, q_vel_norms, c_vel_norms):
    plt.figure(figsize=(10, 8))
    
    # Plotting the "Phase Flow"
    plt.plot(q_vel_norms, q_history, color='gold', alpha=0.6, label='Quasi-Lattice Attractor', lw=1)
    plt.plot(c_vel_norms, c_history, color='black', alpha=0.3, label='Cubic Grid Attractor', lw=1, ls='--')
    
    # Marking the final state (convergence points)
    plt.scatter(q_vel_norms[-1], q_history[-1], c='gold', s=100, edgecolors='black', zorder=5)
    plt.scatter(c_vel_norms[-1], c_history[-1], c='black', s=100, edgecolors='white', zorder=5)

    plt.title("Pillar 1: Phase-Space Attractor\nKinetic Energy vs. Peak Vorticity")
    plt.xlabel("Global Kinetic Energy (Velocity Norm)")
    plt.ylabel("Peak Vorticity Intensity")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/phase_space_attractor.png")
    print("Attractor plot saved to plots/phase_space_attractor.png")

# Note: This logic should be integrated into your next run to capture 'vel_norms'
