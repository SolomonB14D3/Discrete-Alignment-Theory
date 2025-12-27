import numpy as np
import json
import os

def generate_scaling_data():
    print("📈 Pillar 3: Generating Kissing Number Scaling Data...")
    n = np.linspace(2, 24, 100)
    delta = 0.309
    phi = (1 + 5**0.5) / 2
    
    # Scaling Law A(n)
    A_n = 12 / np.sin(np.pi / (n - delta))
    # Entropy Delay Tau
    tau_d = phi**((12 - np.abs(n - 12)) / 12)
    
    data = {
        "n_values": n.tolist(),
        "scaling_amplitude": A_n.tolist(),
        "entropy_delay": tau_d.tolist(),
        "frustration_peak": 12.0,
        "harmony_plateau": 12.0
    }
    
    os.makedirs("data/pillar3", exist_ok=True)
    with open("data/pillar3/scaling_verification.json", "w") as f:
        json.dump(data, f, indent=4)
    print("✅ Pillar 3 data synchronized for Figure 3.")

if __name__ == "__main__":
    generate_scaling_data()
