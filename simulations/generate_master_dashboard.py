import pandas as pd
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel 1: Pillar 1 (Drag) ---
p1_path = 'data/AIRFOIL_DRAG_VALIDATION.csv' if os.path.exists('data/AIRFOIL_DRAG_VALIDATION.csv') else 'AIRFOIL_DRAG_VALIDATION.csv'
if os.path.exists(p1_path):
    df1 = pd.read_csv(p1_path)
    axes[0].plot(df1['step'], df1['cubic_Cd'], color='black', alpha=0.3, label='Cubic')
    axes[0].plot(df1['step'], df1['quasi_Cd'], color='royalblue', linewidth=2, label='DAT-E6')
    axes[0].set_title('Pillar 1: Structural Drag')
    axes[0].legend()

# --- Panel 2: Pillar 2 (Entropy) ---
p2_path = 'data/ENTROPY_EFFICIENCY_VALIDATION.csv'
if os.path.exists(p2_path):
    df2 = pd.read_csv(p2_path)
    axes[1].plot(df2['step'], df2['cubic_H'], color='black', alpha=0.3)
    axes[1].plot(df2['step'], df2['quasi_H'], color='gold', linewidth=2)
    axes[1].set_title('Pillar 2: Entropy Recovery')

# --- Panel 3: Pillar 3 (Scaling) ---
p3_path = 'data/PHASON_SLIP_SCALING.csv'
if os.path.exists(p3_path):
    df3 = pd.read_csv(p3_path)
    axes[2].loglog(df3['N'], df3['cubic_slips_avg'], 'o--', color='black', alpha=0.3, label='Cubic')
    axes[2].loglog(df3['N'], df3['quasi_slips_avg'], 's-', color='crimson', linewidth=2, label='DAT-E6')
    axes[2].set_title('Pillar 3: Scaling Law')
    axes[2].legend()

plt.tight_layout()
plt.savefig('plots/master_dashboard.png')
print("✅ Master Dashboard saved to plots/master_dashboard.png")
