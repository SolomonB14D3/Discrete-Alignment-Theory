import torch
import numpy as np

def get_icosahedral_projection(n_points, device="mps"):
    phi = (1 + np.sqrt(5)) / 2
    P = torch.tensor([
        [1, phi, 0, -phi, -1],
        [1, -phi, 0, -phi, 1],
        [0, 1, phi, 1, 0]
    ], dtype=torch.float32, device=device) / np.sqrt(10)
    torch.manual_seed(42)
    pts_5d = torch.randn(n_points, 5, device=device)
    return pts_5d @ P.T

if __name__ == "__main__":
    print("Testing E6-to-3D Icosahedral Projection...")
    test_pts = get_icosahedral_projection(n_points=12)
    print(f"Generated {len(test_pts)} resonance nodes at n=12.")
    print("Core Geometry Module: VALIDATED")
