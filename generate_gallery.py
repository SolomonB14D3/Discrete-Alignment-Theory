import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('experiment_r7_20251224_125406/raw_data.csv')

def save_plot(frame_idx, filename, title):
    frame_data = df[df['frame'] == frame_idx]
    plt.figure(figsize=(6,6), facecolor='black')
    plt.scatter(frame_data['x'], frame_data['y'], s=1, c='cyan', alpha=0.6)
    plt.title(title, color='white')
    plt.axis('off')
    plt.savefig(filename, facecolor='black')
    plt.close()

# 1. Initial State (Frame 0)
save_plot(0, 'Initial_Order.png', 'E6 Ground State')
# 2. Peak Chaos (Where Beta was lowest/NaN - around Frame 300)
save_plot(300, 'Peak_Chaos.png', 'Peak Phason Strain (1.5σ)')
# 3. Frozen Stars (The final stasis - Frame 2900)
save_plot(2900, 'Frozen_Stars.png', 'Recovered Frozen Stars (β=3.01)')

print("Gallery generated: Initial_Order.png, Peak_Chaos.png, Frozen_Stars.png")
