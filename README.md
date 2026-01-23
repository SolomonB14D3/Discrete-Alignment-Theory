# Icosahedral Quasicrystal Toolkit

Computational tools for generating icosahedral quasicrystal structures via E₆ → H₃ Coxeter projection, with applications to **phonon localization**, **phason-driven switching**, and **thermal transport asymmetry**.

## Applications

### 1. Thermal Diode Design

Icosahedral quasicrystals exhibit asymmetric phonon transport — compression and tension directions have different vibrational density of states. This enables thermal rectification:

| Property | Icosahedral | Cubic | Advantage |
|----------|-------------|-------|-----------|
| IPR (localization) | 4.2× baseline | 1.0× | Stronger phonon trapping |
| DOS asymmetry | 76.7% | ~0% | Directional heat flow |
| Thermal conductivity | Anomalously low | Normal | Better insulation |

**Potential use**: Solid-state thermal diodes, heat management in electronics, thermoelectric devices.

### 2. Phason-Based Mechanical Switching

Phason strain in quasicrystals causes bond reconfiguration without destroying the structure:

| Metric | Value |
|--------|-------|
| Bond reconfiguration | ~13,000 bonds (18.88%) |
| Structural integrity | 81% maintained |
| Phason drift | 0.756 units (3D) |
| Reversibility | Lock-and-key transition |

**Potential use**: Mechanical actuators, shape-memory-like behavior, programmable metamaterials.

### 3. Quasicrystal Lattice Generation

Generate icosahedral point sets for molecular dynamics, phonon calculations, or structural analysis:

```python
from dat_core import get_h3_lattice

# Generate 10,000-point icosahedral quasicrystal
lattice = get_h3_lattice(n_points=10000)

# With phason strain (for switching studies)
strained = get_h3_lattice(n_points=10000, phason_offset=0.1)
```

## Quick Start

```bash
git clone https://github.com/SolomonB14D3/Discrete-Alignment-Theory.git
cd Discrete-Alignment-Theory
pip install numpy torch

# Generate quasicrystal and compute properties
python dat_core.py

# Phonon localization comparison (icosahedral vs cubic)
python pillar4_thermal_diagnostic.py

# Phason switching simulation
python phason_slip_detector.py
```

## How It Works

### E₆ → H₃ Projection

The toolkit uses the mathematical structure of Lie algebra root systems to generate quasicrystalline point sets:

```
E₆ Lie Algebra (72 roots, 6D)
    ↓ Z₂ outer automorphism
F₄ (48 roots, 4D)
    ↓ Non-crystallographic projection
H₃ (icosahedral symmetry, 3D)
    ↓ Cut-and-project
Quasicrystal point set
```

This is the standard approach from Levine & Steinhardt (1984), implemented with φ-based basis vectors and stereographic projection.

### The Golden Ratio in Icosahedral Geometry

The golden ratio φ = (1+√5)/2 appears naturally in icosahedral structures:
- Vertex coordinates involve φ (e.g., icosahedron vertices at (0, ±1, ±φ))
- Inflation factor for Penrose-like tilings is φ
- Basis vectors for cut-and-project use φ-ratios

The constant δ₀ = 1/(2φ) ≈ 0.309 characterizes the angular geometry of the icosahedron (related to vertex angle θ = arccos(1/√5) ≈ 63.43°).

## Core Modules

### `dat_core.py` — Lattice Generator
- Generates H₃ quasicrystal point sets from φ-based 4D vectors
- Supports phason strain (rotation in perpendicular space)
- Shell inflation for deep point pools
- Deterministic seeding for reproducibility

### `core/geometry.py` — Projection Engine
- 5D → 3D icosahedral projection
- φ-based projection matrix
- GPU-accelerated (PyTorch MPS)

### `pillar4_thermal_diagnostic.py` — Phonon Analysis
- Computes IPR (Inverse Participation Ratio) for phonon modes
- Compares icosahedral vs cubic lattice localization
- Density of states calculation

### `phason_slip_detector.py` — Switching Dynamics
- Applies phason strain to quasicrystal
- Tracks bond reconfiguration
- Measures structural integrity during switching

## Physics Background

### Phonon Localization in Quasicrystals

Unlike periodic crystals (where Bloch's theorem guarantees extended states), quasicrystals exhibit:
- **Critical wave functions**: neither fully extended nor exponentially localized
- **Hierarchical gaps**: energy spectrum has fractal-like gap structure
- **Anomalous transport**: thermal conductivity scales differently with temperature

This is well-established physics — see Janot (1994), Steinhardt & Ostlund (1987).

### Phason Dynamics

Phasons are the unique excitations of quasicrystals — fluctuations in the perpendicular-space component of the higher-dimensional embedding. Unlike phonons (which cost elastic energy quadratically), phasons can reorganize local structure:
- Tile flips in 2D Penrose tilings
- Bond switching in 3D icosahedral quasicrystals
- Diffusive rather than propagating dynamics

Real phason physics has been observed in Al-Pd-Mn, Al-Cu-Fe, and other icosahedral quasicrystals.

## What This Is NOT

This toolkit was originally part of "Discrete Alignment Theory" which claimed connections to Navier-Stokes regularity and other Millennium Prize Problems. Those claims have been **withdrawn** — see the [full analysis](https://github.com/SolomonB14D3/navier-stokes-h3/blob/main/analytical_proof_attempt.md). Specifically:

- ~~"Bounds vortex stretching"~~ — a bounded multiplicative depletion cannot change the supercritical Z^(3/2) exponent
- ~~"81.4% turbulence reduction"~~ — was a numerical solver artifact
- ~~"Universal δ₀ across Millennium Problems"~~ — connections falsified or insufficient

What remains is legitimate computational materials science.

## References

- Levine, D. & Steinhardt, P. (1984). Quasicrystals: A New Class of Ordered Structures. *Phys. Rev. Lett.* 53, 2477.
- Janot, C. (1994). *Quasicrystals: A Primer*. Oxford.
- Steinhardt, P. & Ostlund, S. (1987). *The Physics of Quasicrystals*. World Scientific.
- Baake, M. & Grimm, U. (2013). *Aperiodic Order, Vol. 1*. Cambridge.
- Coxeter, H.S.M. (1973). *Regular Polytopes*. Dover.

## Future Work

- [ ] LAMMPS integration for proper phonon transport simulations
- [ ] Quantify thermal rectification ratio vs temperature
- [ ] Phason switching energy barriers (NEB calculations)
- [ ] Compare with real Al-Pd-Mn quasicrystal data
- [ ] Extend to decagonal (2D quasiperiodic) structures

## License

MIT License
