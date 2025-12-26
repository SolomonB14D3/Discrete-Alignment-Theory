import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the plots directory exists
os.makedirs('plots', exist_ok=True)

# Load the "Smoking Gun" data
df = pd.read_csv('data/pillar1/full_proof_ledger.csv')
cubic_valid = df[df['c_ener'].notna()]

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot A: Energy Evolution (Log Scale)
ax1.plot(df.index, df['q_ener'], color='#FFD700', linewidth=2, label='DAT-E6 (Regular)')
ax1.plot(cubic_valid.index, cubic_valid['c_ener'], color='#FF4500', linewidth=1.5, linestyle='--', label='Cubic (Singularity)')
ax1.set_yscale('log')
ax1.set_title("Global Energy Regularity", fontsize=16, color='white', pad=20)
ax1.set_xlabel("Time Step", fontsize=12)
ax1.set_ylabel("Kinetic Energy (L2 Norm)", fontsize=12)
ax1.grid(alpha=0.2)
ax1.legend(frameon=False)

# Plot B: The Singularity Event (Zoom in on steps 170-196)
zoom_start, zoom_end = 170, 196
ax2.plot(df.index[zoom_start:zoom_end+10], df['q_vort'][zoom_start:zoom_end+10], color='#FFD700', marker='o', markersize=4, label='Quasi')
ax2.plot(cubic_valid.index[zoom_start:], cubic_valid['c_vort'][zoom_start:], color='#FF4500', marker='x', markersize=6, label='Cubic Singularity')
ax2.annotate('Cubic Collapse\n(Step 195)', xy=(195, 8631), xytext=(175, 6000),
             arrowprops=dict(facecolor='white', shrink=0.05), fontsize=10)

ax2.set_title("Detail: Geometric Breakdown", fontsize=16, color='white', pad=20)
ax2.set_xlabel("Time Step", fontsize=12)
ax2.set_ylabel("Peak Vorticity", fontsize=12)
ax2.grid(alpha=0.2)
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig('plots/figure1_singularity_proof.png', dpi=300)
print("Figure 1 generated: plots/figure1_singularity_proof.png")
