# DAT 2.0: E₆ Lattice Resilience and Topological Alignment

> **The "Resonant Snap":** A deterministic proof that 6D E₆ projections autonomously recover icosahedral symmetry from high-entropy states.

![Initial Order](Initial_Order.png) | ![Peak Chaos](Peak_Chaos.png) | ![Frozen Stars](Frozen_Stars.png)
:---: | :---: | :---:
**1. Initial Ground State** | **2. Peak Phason Strain** | **3. Autonomous Recovery**

---

## Key Findings
* **Topological Attractor:** The E₆ root system ($r=7$, 2,442 nodes) acts as a geometric basin of attraction.
* **Deterministic Realignment:** Systems subjected to stochastic noise exhibit a "Snap-back" effect, realigning to 6D anchors without external correction.
* **Universal Stasis:** Post-strain analysis shows a consistent convergence to a spectral exponent of **β ≈ 3.01**, characterized by the formation of **"Frozen Stars"**.

## Verify the Results
To reproduce the $\beta$ convergence and generate the "Frozen Star" gallery:

1. **Clone & Install:**
   \`\`\`bash
   git clone https://github.com/SolomonB14D3/DAT-E6-Resilience.git
   cd DAT-E6-Resilience
   pip install torch numpy pandas scipy matplotlib
   \`\`\`

2. **Run the Gold Standard Engine:**
   \`\`\`bash
   python dat_gold_standard.py
   \`\`\`

3. **Analyze the Results:** Check the \`experiment_r7_*/\` folder for generated spectral plots and the icosahedral gallery.

## Mathematical Foundation
For the full derivation of the damped Hamiltonian framework and the E₆ root system projection onto the icosahedral manifold, see [THEORY.md](THEORY.md).

## License
This project is licensed under the Apache License 2.0.
