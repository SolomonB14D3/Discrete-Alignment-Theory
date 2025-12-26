import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- STYLE CONFIGURATION ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300
})
sns.set_palette("viridis")

def create_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # --- PANEL A: Global Regularity (Vorticity Bounding) ---
    try:
        df_reg = pd.read_csv("data/pillar1/DEPLETION_CONSTANT_VALIDATION.csv")
        # Plot only the highest Re for clarity in the main dashboard
        re_max = df_reg['Re'].max()
        subset = df_reg[df_reg['Re'] == re_max]
        
        axes[0, 0].plot(subset['step'], subset['quasi_omega'], label='DAT-E6 (Quasi)', color='teal', lw=2)
        axes[0, 0].plot(subset['step'], subset['cubic_omega'], label='Cubic Grid', color='crimson', linestyle='--', lw=1.5)
        axes[0, 0].set_title(f"A: Global Regularity (Re={re_max:.0e})")
        axes[0, 0].set_xlabel("Simulation Step")
        axes[0, 0].set_ylabel("Max Vorticity ($\omega_{max}$)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"Data Missing\n{e}", ha='center')

    # --- PANEL B: Information Efficiency (Shannon Entropy) ---
    try:
        df_eff = pd.read_csv("data/pillar2/entropy_results.csv")
        # Assuming format: metric, cubic, quasi
        sns.barplot(data=df_eff, x='metric', y='quasi', ax=axes[0, 1], color='teal', label='Quasi')
        sns.barplot(data=df_eff, x='metric', y='cubic', ax=axes[0, 1], color='silver', alpha=0.5, label='Cubic')
        axes[0, 1].set_title("B: Information Efficiency")
        axes[0, 1].set_ylabel("Shannon Entropy (Bits)")
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=15)
        axes[0, 1].legend()
    except:
        axes[0, 1].text(0.5, 0.5, "Data Missing (Pillar 2)", ha='center')

    # --- PANEL C: Symmetry Scaling (Resonance Singularities) ---
    try:
        df_sym = pd.read_csv("data/pillar3/symmetry_scaling_data.csv")
        axes[1, 0].errorbar(df_sym['n'], df_sym['align_mean'], yerr=df_sym['align_std'], 
                           fmt='-o', markersize=4, color='darkblue', ecolor='gray', capsize=2, label='Alignment')
        axes[1, 0].axvline(5625, color='orange', linestyle=':', label='Stability Singularity')
        axes[1, 0].set_title("C: Symmetry Resonance Scaling")
        axes[1, 0].set_xlabel("Symmetry Order ($n$)")
        axes[1, 0].set_ylabel("Field Alignment")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    except:
        axes[1, 0].text(0.5, 0.5, "Data Missing (Pillar 3)", ha='center')

    # --- PANEL D: Resilience (Phononic Mirror / IPR) ---
    try:
        df_res = pd.read_csv("data/pillar4/thermal_reflection_results.csv")
        sns.lineplot(data=df_res, x='frequency', y='IPR', hue='lattice', ax=axes[1, 1])
        axes[1, 1].set_title("D: Phononic Mirror Resilience")
        axes[1, 1].set_xlabel("Excitation Frequency (Hz)")
        axes[1, 1].set_ylabel("Inverse Participation Ratio (IPR)")
        axes[1, 1].grid(alpha=0.3)
    except:
        # Fallback to alloy comparison if IPR is missing
        axes[1, 1].text(0.5, 0.5, "Data Missing (Pillar 4)", ha='center')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/master_manuscript_dashboard.png")
    print("✅ Master Manuscript Dashboard saved to plots/master_manuscript_dashboard.png")

if __name__ == "__main__":
    create_dashboard()
