# 🔗 DAT 2.0 Manuscript-to-Repository Mapping

This table maps the figures and claims in the DAT 2.0 Manuscript to the corresponding computational evidence in this repository.

| Manuscript Element | Theoretical Pillar | Repository Source / Script | Data Artifact |
| :--- | :--- | :--- | :--- |
| **Figure 1**: Vorticity Cap | Pillar 1: Regularity | `simulations/vorticity_test.py` | `data/pillar1/` |
| **Figure 2**: Beta Jumps | Pillar 2: Memory | `scripts/generate_figure2_entropy.py` | `VALIDATION_REPORT.txt` |
| **Figure 3**: Scaling Law | Pillar 3: Entropy | `notebooks/density_hypothesis.py` | `plots/pillar3_scaling_law.png` |
| **Figure 4**: Leakage | Pillar 4: Mirroring | `simulations/thermal_conductivity_bench.py` | `data/THERMAL_LOCALIZATION_MAP.json` |
| **Figure 5**: Plasma Ext. | Extensions | `simulations/dat_extensions.py` | `data/dat_extensions_simulations.csv` |
| **Appendix F**: Recovery | Pillar 2: Realignment| `core/dat_universal_engine.py` | N/A |
| **Appendix G**: Thermal | Pillar 4: Localization| `data/THERMAL_LOCALIZATION_MAP.json` | JSON Schema |

