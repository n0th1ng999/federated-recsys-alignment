#!/usr/bin/env bash
set -e

echo "================================================================="
echo "Federated Recommender System Alignment - Reproducibility Suite"
echo "================================================================="

echo ""
echo "[Step 1/3] Preprocessing EARS Pilot Dataset..."
python3 data/ears/preprocess_ears.py

echo ""
echo "[Step 2/3] Running EARS Pilot Micro-Federation Benchmark..."
python3 experiments/benchmark_pilot_ears.py

echo ""
echo "[Step 3/3] Running MovieLens 100K Expanded Federated Benchmark..."
python3 experiments/benchmark_movielens.py

echo ""
echo "[Step 4/4] Regenerating Publication Figures (300 DPI)..."
python3 experiments/generate_figures.py

echo ""
echo "================================================================="
echo "All benchmarks and figures generated successfully!"
echo "Results stored in experiments/ and figures/."
echo "================================================================="
