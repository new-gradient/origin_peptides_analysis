#!/usr/bin/env python3
"""
SHAP analysis, Figure 3 panels, and the Table S2 feature glossary.

For each regression target this:
  1. loads the validation-selected model from ``models/regression/<target>/``;
  2. computes SHAP values for every peptide;
  3. writes the per-peptide SHAP table to ``results/shap/`` (CSV + XLSX);
  4. renders the Figure 3A heatmap to ``results/figures/``.

Then, from the peak-intensity SHAP table, it renders the three Figure 3B-D
peptide-group panels, and finally writes the Table S2 feature glossary.

Run ``scripts/02_train_models.py`` first so the models exist.

Usage
-----
    python scripts/03_shap_and_figures.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peptide_ml.figures import GROUPS, group_panel  # noqa: E402
from peptide_ml.shap_analysis import compute_shap_table, load_bundle, shap_heatmap  # noqa: E402
from peptide_ml.train import TARGETS, short_name  # noqa: E402

DATA = REPO / "data" / "features_with_targets.csv"
MODELS = REPO / "models" / "regression"
SHAP_DIR = REPO / "results" / "shap"
FIG_DIR = REPO / "results" / "figures"


def main():
    df = pd.read_csv(DATA)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    shap_tables = {}
    for target in TARGETS:
        sn = short_name(target)
        target_dir = MODELS / sn
        if not target_dir.exists():
            print(f"[skip] no model for {sn} (run 02_train_models.py first)")
            continue
        print(f"\n=== SHAP :: {sn} ===")
        model, scaler, features = load_bundle(target_dir, sn)
        table = compute_shap_table(df, model, scaler, features, target)
        shap_tables[sn] = table

        table.to_csv(SHAP_DIR / f"shap_{sn}.csv", index=False)
        table.to_excel(SHAP_DIR / f"shap_{sn}.xlsx", index=False)
        shap_heatmap(table, features, sn, FIG_DIR / f"figure3A_shap_heatmap_{sn}.png")
        print(f"  wrote SHAP table + Figure 3A heatmap for {sn}")

    # Figure 3B-D group panels are built from the peak-intensity SHAP table.
    base_table = shap_tables.get("peak_intensity")
    if base_table is not None:
        for group in GROUPS:
            out = FIG_DIR / f"figure{group.panel}_{group.key}.png"
            try:
                sub = group_panel(base_table, group, out)
                sub.to_csv(SHAP_DIR / f"{group.key}.csv", index=False)
                print(f"  Figure {group.panel}: {group.title} -> {len(sub)} peptides")
            except ValueError as e:
                print(f"  [warn] {group.key}: {e}")
    else:
        print("[skip] Figure 3B-D need the peak_intensity model.")

    # Table S2 glossary.
    print("\n=== Table S2 feature glossary ===")
    out_csv = REPO / "results" / "tableS2_feature_glossary.csv"
    subprocess.run(
        [sys.executable, "-m", "peptide_ml.feature_glossary",
         "--data", str(DATA), "--output", str(out_csv), "--xlsx"],
        cwd=str(REPO), check=True,
    )


if __name__ == "__main__":
    main()
