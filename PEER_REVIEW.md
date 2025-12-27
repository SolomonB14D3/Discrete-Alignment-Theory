# Peer Review & Reproducibility Guide

This document maps the claims in `dat_manuscript.tex` to executable code and raw data.

### Claim 1: 81.4% Vorticity Reduction (Pillar 1)
* **Script:** `simulations/navier_stokes_lattice_cap.py`
* **Verification:** Run `python3 scripts/verify_manuscript_data.py`
* **Expected Output:** `Pillar 1: 0.2445 (81.40% reduction) ✅ PASS`

### Claim 2: Harmony Plateau at n=12 (Pillar 3)
* **Script:** `core/geometry.py`
* **Data:** `data/PHASON_SLIP_SCALING.csv`
* **Verification:** Observe the local minimum in phason strain at the dozenal vertex count (12.0).

### Claim 3: 246x Faster Entropy Decay (Pillar 2)
* **Script:** `simulations/generate_figure2_entropy.py`
* **Data:** `data/ENTROPY_EFFICIENCY_VALIDATION.csv`

### Reproducibility Environment
A Dockerized environment is provided to ensure version-locked results:
\`\`\`bash
docker build -t dat-e6 .
docker run dat-e6
\`\`\`
