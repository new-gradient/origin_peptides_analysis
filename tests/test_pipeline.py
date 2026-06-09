"""
Lightweight sanity checks for the peptide-ML pipeline.

These are fast, data-dependent smoke tests (not exhaustive unit tests): they
confirm the package imports, the canonical table has the expected shape, the
descriptor and target code run, and a tiny training sweep produces sane,
validation-selected metrics. Run with ``pytest`` from the repository root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO / "data" / "features_with_targets.csv"
EXPERIMENTS = REPO / "data" / "raw" / "experiments"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_imports():
    import peptide_ml  # noqa: F401
    from peptide_ml import figures, preprocessing, shap_analysis, targets, train  # noqa: F401


def test_canonical_table_shape(df):
    # 6955 rows in the table; the models use 6954 after dropping the single
    # peptide with a missing target.
    assert len(df) == 6955
    assert {"ID", "sequence"}.issubset(df.columns)
    for target in ["peak_intensity_norm_avg", "auc_norm_avg", "total_increase_norm_avg"]:
        assert target in df.columns
    assert df["peak_intensity_norm_avg"].notna().sum() == 6954


def test_feature_selection_returns_119(df):
    from peptide_ml.preprocessing import select_feature_columns
    features = select_feature_columns(df)
    assert len(features) == 119
    # none of the selected columns are targets/identifiers
    assert "sequence" not in features
    assert not any("norm_avg" in f for f in features)


def test_descriptor_code_runs():
    from peptide_ml.features.non_folding import ChargeFeature, MolecularWeightFeature
    mw = MolecularWeightFeature().calculate_ml_ready("ACDEFGHIK")
    assert mw["molecular_weight"] > 0
    charge = ChargeFeature().calculate_ml_ready("ACDEFGHIK")
    assert isinstance(charge, dict) and len(charge) > 0


def test_target_metrics_run():
    from peptide_ml.targets import compute_raw_metrics
    exp = pd.read_csv(next(EXPERIMENTS.glob("*.csv")))
    metrics = compute_raw_metrics(exp)
    assert "peak_intensity_raw" in metrics.columns
    assert (metrics["peak_intensity_raw"].dropna() >= 0).all()


def test_targets_reproduce_headline(df):
    """Recomputed peak-intensity / AUC targets correlate strongly with shipped."""
    from peptide_ml.targets import compute_targets_from_experiments
    rec = compute_targets_from_experiments(EXPERIMENTS).set_index("sequence")
    ship = df.set_index("sequence")
    common = rec.index.intersection(ship.index)
    for base in ["peak_intensity", "auc"]:
        a = rec.loc[common, f"{base}_norm_avg"]
        b = ship.loc[common, f"{base}_norm_avg"]
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        assert r > 0.99, f"{base} reproduction r={r:.3f}"


@pytest.mark.slow
def test_tiny_regression_sweep(df):
    """A tiny sweep selects on validation and yields a plausible test r."""
    from peptide_ml.preprocessing import select_feature_columns
    from peptide_ml.train import run_sweep
    features = select_feature_columns(df)
    result = run_sweep(df, "peak_intensity_norm_avg", "regression", features, n_iterations=5)
    assert 0.2 < result.best_metrics["test_pearson"] < 0.7
    assert result.best_metrics["n_features"] == 119
