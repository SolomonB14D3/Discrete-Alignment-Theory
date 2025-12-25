import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

class DAT3AnnealingEngine:
    def __init__(self, lattice_type='E8'):
        self.lattice_type = lattice_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # B.1 Scaling Laws
        self.dims = {'E6': 6, 'E8': 8, 'Leech': 24}
        self.d = self.dims.get(lattice_type, 6)
        
        # Scaling k proportional to sqrt(d) to fight dimensional volume
        self.k = 25.0 * np.sqrt(self.d / 6.0) 
        self.gamma = 0.5 * np.sqrt(6.0 / self.d)
        self.dt = 0.01
        
        self.anchors = self._generate_lattice()
        print(f"--- DAT 3.0: {lattice_type} (d={self.d}) ---")
        print(f"Scaling: k={self.k:.2f}, γ={self.gamma:.2f}")

    def _generate_lattice(self):
        if self.lattice_type == 'E8':
            roots = []
            for i in range(8):
                for j in range(i + 1, 8):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            v = np.zeros(8); v[i], v[j] = s1, s2; roots.append(v)
            for i in range(256):
                v = np.array([0.5 if b == '0' else -0.5 for b in bin(i)[2:].zfill(8)])
                if np.sum(v / 0.5) % 2 == 0: roots.append(v)
            return torch.tensor(np.array(roots), dtype=torch.float32).to(self.device)
        
        elif self.lattice_type == 'Leech':
            kissing = [] 
            for i in range(24):
                for j in range(i + 1, 24):
                    for s1 in [-2, 2]:
                        for s2 in [-2, 2]:
                            v = np.zeros(24); v[i], v[j] = s1, s2; kissing.append(v)
            return torch.tensor(np.array(kissing), dtype=torch.float32).to(self.device)
        
        return torch.randn(72, 6).to(self.device)

    def run_annealed_simulation(self, start_sigma=2.5, end_sigma=0.05, steps=15000):
        """Langevin integration with Linear Annealing Schedule."""
        n_particles = self.anchors.shape[0]
        # Start from high entropy
        x = self.anchors.clone() + torch.randn_like(self.anchors) * start_sigma
        v = torch.zeros_like(x)
        
        history = []
        for i in range(steps):
            # Annealing: Gradually reduce noise intensity
            current_sigma = start_sigma * (1 - i/steps) + end_sigma
            
            grad = self.k * (x - self.anchors)
            noise = torch.randn_like(x) * np.sqrt(2 * self.gamma * current_sigma * self.dt)
            
            v += (-grad - self.gamma * v) * self.dt + noise
            x += v * self.dt
            
            if i % 100 == 0:
                dist = torch.norm(x - self.anchors, dim=1).mean().item()
                history.append(dist)
                
        return x, history

    def calculate_beta(self, final_state):
        flat = final_state.cpu().numpy().flatten()
        freqs = np.fft.rfftfreq(len(flat))
        psd = np.abs(np.fft.rfft(flat))**2
        mask = freqs > 0
        slope, _, _, _, _ = linregress(np.log10(freqs[mask]), np.log10(psd[mask]))
        return -slope

if __name__ == "__main__":
    results = []
    for lat in ['E8', 'Leech']:
        engine = DAT3AnnealingEngine(lattice_type=lat)
        # Increased steps to allow high-d convergence
        final_pos, dist_hist = engine.run_annealed_simulation(steps=15000) 
        beta = engine.calculate_beta(final_pos)
        
        print(f"Final State Analysis for {lat}: β = {beta:.3f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(dist_hist, color='#00FFCC')
        plt.title(f"DAT 3.0 Annealing Curve: {lat}\nFinal β: {beta:.3f}")
        plt.xlabel("Checkpoint (x100 Steps)")
        plt.ylabel("Topological Deviation")
        plt.grid(True, alpha=0.2)
        plt.savefig(f"anneal_{lat}.png")
        plt.close()
