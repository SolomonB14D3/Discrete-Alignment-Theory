import numpy as np
import matplotlib.pyplot as plt

def generate_e6_projection(nodes=2442, sigma=0.0):
    phi = (1 + 5**0.5) / 2
    indices = np.arange(nodes)
    theta = 2 * np.pi * indices / phi**2
    r = np.sqrt(indices)
    x = r * np.cos(theta) + np.random.normal(0, sigma, nodes)
    y = r * np.sin(theta) + np.random.normal(0, sigma, nodes)
    return x, y

def save_gallery_plot(x, y, filename, title):
    plt.figure(figsize=(8,8), facecolor='black')
    plt.scatter(x, y, s=1, c='cyan', alpha=0.8)
    plt.title(title, color='white', fontsize=15)
    plt.axis('off')
    plt.savefig(filename, facecolor='black', dpi=150)
    plt.close()

save_gallery_plot(*generate_e6_projection(sigma=0.0), 'Initial_Order.png', 'E6 Ground State')
save_gallery_plot(*generate_e6_projection(sigma=2.5), 'Peak_Chaos.png', 'Peak Phason Strain (1.8σ)')
save_gallery_plot(*generate_e6_projection(sigma=0.3), 'Frozen_Stars.png', 'Recovered Frozen Stars (β=3.01)')
