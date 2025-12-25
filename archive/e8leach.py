import numpy as np
import torch
import matplotlib.pyplot as plt

class DATLatticeGenerator:
    @staticmethod
    def generate_e8_roots():
        roots = []
        # 1. Permutations of (±1, ±1, 0^6)
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        v = np.zeros(8)
                        v[i], v[j] = s1, s2
                        roots.append(v)
        # 2. All combinations of ±0.5 with even sum
        for i in range(256):
            binary = bin(i)[2:].zfill(8)
            v = np.array([0.5 if b == '0' else -0.5 for b in binary])
            if np.sum(v / 0.5) % 2 == 0:
                roots.append(v)
        return torch.tensor(np.array(roots), dtype=torch.float32)

    @staticmethod
    def generate_leech_subset():
        # Using a 24D construction of (±2, ±2, 0^22)
        d = 24
        kissing = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [-2, 2]:
                    for s2 in [-2, 2]:
                        v = np.zeros(d)
                        v[i], v[j] = s1, s2
                        kissing.append(v)
        return torch.tensor(np.array(kissing), dtype=torch.float32)

def visualize_projection(nodes, title, filename):
    # Create a random but deterministic projection matrix to 2D
    np.random.seed(42)
    dim = nodes.shape[1]
    proj = np.random.randn(dim, 2)
    # Orthogonalize for clarity
    q, _ = np.linalg.qr(proj)
    
    projected = nodes.numpy() @ q
    
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.scatter(projected[:, 0], projected[:, 1], s=10, color='#00FFCC', alpha=0.7)
    plt.title(title, color='white', fontsize=15)
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    print("Generating Lattices...")
    
    # E8 Generation
    e8_nodes = DATLatticeGenerator.generate_e8_roots()
    visualize_projection(e8_nodes, "E8 Root System Projection (DAT 3.0)", "e8_test.png")
    
    # Leech Generation
    leech_nodes = DATLatticeGenerator.generate_leech_subset()
    visualize_projection(leech_nodes, "Leech Lattice Subset Λ24 (DAT 3.0)", "leech_test.png")
    
    print("Done. Check your folder for e8_test.png and leech_test.png.")
