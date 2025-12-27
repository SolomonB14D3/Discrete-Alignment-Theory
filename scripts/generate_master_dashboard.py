import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the verified datasets
p1_df = pd.read_csv('data/DEPLETION_CONSTANT_VALIDATION.csv')
p3_df = pd.read_csv('data/PHASON_SLIP_SCALING.csv')

plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel 1: Pillar 1 (Regularity) ---
ax1.plot(p1_df['step'], p1_df['cubic_omega'], label='Cubic (Standard)', color='gray', alpha=0.5)
ax1.plot(p1_df['step'], p1_df['quasi_omega'], label='DAT-E6 (Resilient)', color='blue', linewidth=2)
ax1.axhline(y=0.309, color='red', linestyle='--', label='Theoretical Cap ($\delta_0$)')
ax1.set_title("Pillar 1: Vorticity Depletion")
ax1.set_xlabel("Simulation Step")
ax1.set_ylabel("Vorticity ($\omega$)")
ax1.legend()

# --- Panel 2: Pillar 3 (Scaling) ---
ax2.plot(p3_df['n'], p3_df['entropy_delay'], marker='o', color='purple', markersize=4, label='Entropy Delay')
ax2.axvline(x=12.0, color='green', linestyle=':', label='Harmony Plateau (n=12)')
ax2.annotate('Golden Ratio $(\phi)$', xy=(12, 1.618), xytext=(15, 1.7),
             arrowprops=dict(facecolor='black', shrink=0.05))
ax2.set_title("Pillar 3: Symmetry Scaling")
ax2.set_xlabel("Lattice Order (n)")
ax2.set_ylabel("Frustration Index")
ax2.legend()

plt.tight_layout()
plt.savefig('plots/master_manuscript_dashboard.png', dpi=300)
print("✅ Master Dashboard generated at plots/master_manuscript_dashboard.png")
