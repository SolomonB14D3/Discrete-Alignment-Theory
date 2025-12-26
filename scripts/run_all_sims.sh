#!/bin/bash
echo "🚀 Starting DAT-E6 Master Simulation Suite..."
python pillars/regularity/benchmarks/run_regularity_test.py
python pillars/efficiency/benchmarks/entropy_analysis.py
python pillars/optimization/benchmarks/symmetry_sweep_recursive.py
python pillars/resilience/benchmarks/phononic_mirror.py
python simulations/thermal_conductivity_bench.py
python simulations/validate_results.py
python generate_manuscript_figures.py
echo "✅ All simulations complete. Results archived in data/ and plots/."