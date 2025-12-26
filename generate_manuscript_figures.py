import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set professional plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# --- Panel A: Pillar 1 (Stability Divergence) ---
if os.path.exists("data/pillar1/stability_trace.csv"):
    df1 = pd.read_csv("data/pillar1/stability_trace.csv")
    sns.lineplot(data=df1, x="step", y="max_vel", hue="mode", ax=axes[0,0], palette="magma")
    axes[0,0].set_title("A: Structural Stability (Step 195 Collapse)", fontsize=14)
    axes[0,0].set_yscale('log')
else:
    axes[0,0].text(0.5, 0.5, 'Pillar 1 Data Missing', ha='center')

# --- Panel B: Pillar 2 (Efficiency Gain) ---
if os.path.exists("data/pillar2/entropy_results.csv"):
    df2 = pd.read_csv("data/pillar2/entropy_results.csv")
    # Using 'quasi' and 'cubic' columns from your Pillar 2 logic
    gain_row = df2[df2['metric'] == 'Entropy_Reduction_Pct']
    if not gain_row.empty:
        gain = gain_row['quasi'].values[0]
        sns.barplot(x=["Cubic Grid", "DAT-E6"], y=[100, 100-gain], ax=axes[0,1], palette="viridis")
        axes[0,1].set_title(f"B: Information Efficiency ({gain:.1f}% Improvement)", fontsize=14)
    else:
        # Fallback if Percent row isn't there yet
        quasi_e = df2.loc[df2['metric'] == 'Shannon_Entropy', 'quasi'].values[0]
        cubic_e = df2.loc[df2['metric'] == 'Shannon_Entropy', 'cubic'].values[0]
        gain = ((cubic_e - quasi_e) / cubic_e) * 100
        sns.barplot(x=["Cubic Grid", "DAT-E6"], y=[cubic_e, quasi_e], ax=axes[0,1], palette="viridis")
        axes[0,1].set_title(f"B: Information Entropy ({gain:.1f}% Reduction)", fontsize=14)
    axes[0,1].set_ylabel("Relative Information Entropy (%)")

# --- Panel C: Pillar 3 (Symmetry Resonances) ---
if os.path.exists("data/pillar3/symmetry_scaling_data.csv"):
    df3 = pd.read_csv("data/pillar3/symmetry_scaling_data.csv").sort_values("n")
    axes[1,0].errorbar(df3['n'], df3['align_mean'], yerr=df3['align_std'], fmt='o-', color='crimson', markersize=4, capsize=2)
    axes[1,0].set_xscale('log')
    axes[1,0].set_title("C: Symmetry Resonances (Recursive Sweep)", fontsize=14)
    axes[1,0].set_xlabel("Symmetry Order (n)")

# --- Panel D: Pillar 4 (Resilience Mirror) ---
if os.path.exists("data/pillar4/thermal_reflection_results.csv"):
    df4 = pd.read_csv("data/pillar4/thermal_reflection_results.csv")
    df4_melt = df4[df4['metric'] != 'Transmission_Loss_dB']
    sns.barplot(data=df4_melt.melt(id_vars='metric'), x='metric', y='value', hue='variable', ax=axes[1,1])
    axes[1,1].set_title("D: Resilience & Energy Localization (IPR)", fontsize=14)

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/master_manuscript_dashboard.png", dpi=300)
print("Master Dashboard generated: plots/master_manuscript_dashboard.png")
