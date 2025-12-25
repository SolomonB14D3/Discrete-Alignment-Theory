import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

# --- CONFIG ---
# Use glob to find your most recent CSVs automatically
FILES = {
    'E6': 'experiment_r7_*/raw_data.csv', # Adjust if you have an E6 csv
    'E8': 'DAT3_E8_*.csv',
    'Leech': 'DAT3_Leech_*.csv'
}

plt.figure(figsize=(10, 6), facecolor='white')
colors = {'E6': '#1f77b4', 'E8': '#ff7f0e', 'Leech': '#2ca02c'}

for label, pattern in FILES.items():
    matches = glob.glob(pattern)
    if matches:
        # Load the latest match
        df = pd.read_csv(matches[-1])
        # Smooth the beta to remove spectral jitter
        smoothed_beta = df['beta'].rolling(window=100).mean()
        plt.plot(df['frame'], smoothed_beta, label=f'{label} (Peak β: {df["beta"].max():.2f})', 
                 color=colors[label], lw=2)

# Theoretical Target Line (Black Noise)
plt.axhline(y=3.0, color='black', linestyle='--', alpha=0.5, label='Theoretical Black Noise (β=3.0)')
# Chaos Threshold Line (White Noise)
plt.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5, label='Pure Chaos (β=0.0)')

plt.title('DAT 3.0: Topological Resilience Across Manifolds', fontsize=16)
plt.xlabel('Simulation Time Steps', fontsize=12)
plt.ylabel('Spectral Exponent (β)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.ylim(-0.5, 3.5)

plt.tight_layout()
plt.savefig('DAT_3.0_Phase_Comparison.pdf')
print("Consolidated Phase Map saved to DAT_3.0_Phase_Comparison.pdf")
