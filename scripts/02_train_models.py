#!/usr/bin/env python3
"""
Train the prevalence-prediction models (regression + classification).

Runs a random hyper-parameter sweep for each of the three targets and saves the
validation-selected best model and its held-out test metrics.

Usage
-----
    python scripts/02_train_models.py                       # full 500-config sweep, both modes
    python scripts/02_train_models.py --mode regression
    python scripts/02_train_models.py --iterations 25       # quick smoke run

Outputs land in ``models/regression/<target>/`` and
``models/classification/<target>/``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peptide_ml.preprocessing import select_feature_columns  # noqa: E402
from peptide_ml.train import TARGETS, run_sweep, save_result, short_name  # noqa: E402

DATA = REPO / "data" / "features_with_targets.csv"


def train_mode(df, features, mode, iterations, out_root):
    out_dir = out_root / mode
    rows = []
    for target in TARGETS:
        if target not in df.columns:
            print(f"  [skip] {target} not in data")
            continue
        print(f"\n=== {mode.upper()} :: {short_name(target)} "
              f"({iterations} configs, selecting on "
              f"{'val Pearson' if mode == 'regression' else 'val ROC-AUC'}) ===")
        result = run_sweep(df, target, mode, features, n_iterations=iterations)
        save_dir = save_result(result, out_dir)
        m = result.best_metrics
        if mode == "regression":
            print(f"  best val r={m['val_pearson']:.4f}  ->  TEST r={m['test_pearson']:.4f}, "
                  f"R2={m['test_r2']:.4f}")
            rows.append((short_name(target), m["val_pearson"], m["test_pearson"], m["test_r2"]))
        else:
            print(f"  best val AUC={m['val_auc']:.4f}  ->  TEST AUC={m['test_auc']:.4f}, "
                  f"F1={m['test_f1']:.4f}, acc={m['test_accuracy']:.4f}")
            rows.append((short_name(target), m["val_auc"], m["test_auc"], m["test_f1"]))
        print(f"  saved -> {save_dir.relative_to(REPO)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--iterations", type=int, default=500, help="configs per target (default 500)")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=REPO / "models")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    features = select_feature_columns(df)
    print(f"Loaded {len(df)} peptides, {len(features)} molecular-descriptor features "
          f"from {args.data.relative_to(REPO)}")

    modes = ["regression", "classification"] if args.mode == "both" else [args.mode]
    summary = {}
    for mode in modes:
        summary[mode] = train_mode(df, features, mode, args.iterations, args.out)

    print("\n" + "=" * 64 + "\nSUMMARY (held-out test set)\n" + "=" * 64)
    for mode, rows in summary.items():
        print(f"\n{mode}:")
        if mode == "regression":
            print(f"  {'target':16s} {'val_r':>8s} {'test_r':>8s} {'test_R2':>8s}")
        else:
            print(f"  {'target':16s} {'val_auc':>8s} {'test_auc':>8s} {'test_f1':>8s}")
        for name, a, b, c in rows:
            print(f"  {name:16s} {a:8.4f} {b:8.4f} {c:8.4f}")


if __name__ == "__main__":
    main()
