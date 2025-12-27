import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- STYLE CONFIGURATION ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 300
})
sns.set_palette("viridis")

def create_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Panel A: Global Regularity (Vorticity Bounding)
    p1 = "data/pillar1/DEPLETION_CONSTANT_VALIDATION.csv"
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        re_max = df['Re'].max()
        subset = df[df['Re'] == re_max]
        axes[0, 0].plot(subset['step'], subset['quasi_omega'], label='DAT-E6', color='teal', lw=2)
        axes[0, 0].plot(subset['step'], subset['cubic_omega'], label='Cubic', color='crimson', ls='--', alpha=0.7)
        axes[0, 0].set_title(f"A: Global Regularity (Re={re_max:.0e})")
        axes[0, 0].set_ylabel(r"Max Vorticity ($\omega_{max}$)")
        axes[0, 0].legend()
    
    # Panel B: Information Efficiency (Shannon Entropy)
    p2 = "data/pillar2/entropy_results.csv"
    if os.path.exists(p2):
        df = pd.read_csv(p2)
        sns.barplot(data=df, x='metric', y='quasi', ax=axes[0, 1], color='teal', label='DAT-E6')
        sns.barplot(data=df, x='metric', y='cubic', ax=axes[0, 1], color='silver', alpha=0.5, label='Cubic')
        axes[0, 1].set_title("B: Information Efficiency")
        axes[0, 1].set_ylabel("Entropy (Bits)")

    # Panel C: Symmetry Resonance (Symmetry Sweep)
    p3 = "data/pillar3/symmetry_scaling_data.csv"
    if os.path.exists(p3):
        df = pd.read_csv(p3)
        axes[1, 0].plot(df['n'], df['align_mean'], '-o', color='darkblue', markersize=4)
        axes[1, 0].axvline(5625, color='orange', ls=':', label=r'$\delta_0$ Singularity')
        axes[1, 0].set_title("C: Symmetry Resonance Scaling")
        axes[1, 0].set_xlabel(r"Symmetry Order ($n$)")
        axes[1, 0].set_ylabel("Field Alignment")

    # Panel D: Resilience (Phononic Mirror / IPR)
    p4 = "data/pillar4/thermal_reflection_results.csv"
    if os.path.exists(p4):
        df = pd.read_csv(p4)
        # Handle dynamic column naming for Pillar 4
        x_col = 'frequency' if 'frequency' in df.columns else df.columns[0]
        y_col = 'IPR' if 'IPR' in df.columns else df.columns[1]
        
        sns.lineplot(data=df, x=x_col, y=y_col, hue='lattice' if 'lattice' in df.columns else None, ax=axes[1, 1])
        axes[1, 1].set_title("D: Phononic Mirror Resilience")
        axes[1, 1].set_ylabel("IPR (Localization)")
        axes[1, 1].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/master_manuscript_dashboard.png")
    print("✅ Master Dashboard generated: plots/master_manuscript_dashboard.png")

if __name__ == "__main__":
    create_dashboard()
