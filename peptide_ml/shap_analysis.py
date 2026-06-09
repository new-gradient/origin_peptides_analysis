"""
SHAP analysis and the Figure 3A heatmap.
========================================

Loads a trained regression model (saved by :mod:`peptide_ml.train`), computes
SHAP values for every peptide, and produces:

* a per-peptide SHAP table (sequence, target, raw features, SHAP values) - the
  data source for the Figure 3B-D group panels (see :mod:`peptide_ml.figures`);
* the Figure 3A heatmap of SHAP contributions across all peptides, sorted by
  target value with features ordered by global importance.

Unlike the original analysis scripts, this module does **not** retrain a model
with hard-coded hyper-parameters; it loads the validation-selected model that
the sweep produced, so the interpretation matches the reported model exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .preprocessing import clean_matrix

# Headless-safe matplotlib backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def load_bundle(target_dir: Path, target_short: str) -> Tuple[xgb.XGBRegressor, object, List[str]]:
    """Load the regressor, feature scaler and feature list for one target."""
    target_dir = Path(target_dir)
    model = xgb.XGBRegressor()
    model.load_model(str(target_dir / f"best_regressor_{target_short}.json"))
    scaler = joblib.load(target_dir / f"feature_scaler_{target_short}.joblib")
    features = (target_dir / f"feature_list_{target_short}.txt").read_text().strip().split("\n")
    return model, scaler, features


def compute_shap_table(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    scaler,
    features: List[str],
    target: str,
) -> pd.DataFrame:
    """Return a per-peptide table of raw features and SHAP values.

    Columns: ``Sequence``, ``Target_Intensity``, the raw features, then the
    SHAP value of each feature prefixed with ``SHAP_``. Rows are sorted by
    target value (low to high), matching the heatmap ordering.
    """
    import shap

    X = clean_matrix(df[features].copy())
    y = df[target].copy()
    valid = ~y.isna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    sequences = (
        df["sequence"][valid].reset_index(drop=True)
        if "sequence" in df.columns
        else pd.Series(range(len(y)))
    )

    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Assemble in one concat to avoid DataFrame fragmentation.
    meta = pd.DataFrame({"Sequence": sequences.values, "Target_Intensity": y.values})
    raw = X[list(features)].reset_index(drop=True)
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{c}" for c in features])
    out = pd.concat([meta, raw, shap_df], axis=1)
    return out.sort_values("Target_Intensity").reset_index(drop=True)


def shap_heatmap(
    shap_table: pd.DataFrame,
    features: List[str],
    target_short: str,
    output_path: Path,
    top_n: int = 50,
) -> None:
    """Render the Figure 3A SHAP-contribution heatmap (features x peptides)."""
    shap_cols = [f"SHAP_{f}" for f in features]
    shap_values = shap_table[shap_cols].to_numpy()

    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    top_features = [features[i] for i in order]
    plot_data = shap_values[:, order].T  # rows = features, cols = peptides

    vmax = np.percentile(np.abs(plot_data), 99)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(
        plot_data, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        yticklabels=top_features, xticklabels=False,
        cbar_kws={"label": "SHAP value (contribution to prediction)"}, ax=ax,
    )
    title = target_short.replace("_", " ").title()
    ax.set_xlabel("Peptides (sorted low → high target value)", fontsize=12)
    ax.set_ylabel("Molecular descriptor (sorted by importance)", fontsize=12)
    ax.set_title(
        f"SHAP feature contributions: {title}\nTop {top_n} features, {len(shap_table)} peptides",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
