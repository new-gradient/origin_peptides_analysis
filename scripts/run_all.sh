#!/usr/bin/env bash
# Reproduce the full machine-learning analysis end to end.
# Run from the repository root:  bash scripts/run_all.sh
set -euo pipefail

PY="${PYTHON:-python}"

echo "==> [1/3] Recompute prevalence targets from raw MS time-series (provenance check)"
"$PY" scripts/01_compute_targets.py

echo "==> [2/3] Train models (500-config sweep per target; validation-selected)"
"$PY" scripts/02_train_models.py --iterations 500

echo "==> [3/3] SHAP analysis, Figure 3 panels, and Table S2 glossary"
"$PY" scripts/03_shap_and_figures.py

echo "Done. Models in models/, figures and tables in results/."
