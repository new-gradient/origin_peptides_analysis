"""
Feature preprocessing.
======================

Turns the merged feature+target table into model-ready matrices, following the
preprocessing described in the paper:

1. **Feature identification** - drop identifier columns (``ID``, ``sequence``)
   and any target/metric columns, keeping only numeric molecular descriptors.
2. **Variance filtering** - remove near-constant descriptors
   (variance below :data:`VARIANCE_THRESHOLD` after min-max scaling).
3. **Correlation filtering** - of any pair of descriptors with
   ``|Pearson r| >`` :data:`CORRELATION_THRESHOLD`, drop one to remove
   redundancy.
4. **Imputation** - replace infinities/NaNs with the column median.
5. **Standardisation** - z-score using statistics from the *training* split
   only, to avoid leakage into validation/test.

The shipped ``features_with_targets.csv`` already contains the post-reduction
descriptor set used for the paper's models. The variance/correlation helpers are
provided for transparency and so the reduction can be re-derived from a larger
descriptor set; :func:`select_feature_columns` returns that modelling set
directly.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Columns that are never model inputs.
ID_COLUMNS = ["ID", "sequence"]
# Any column containing one of these substrings is a target/metric, not a feature.
METRIC_PATTERNS = [
    "peak_intensity_",
    "auc_",
    "staying_power_",
    "total_increase_",
    "template_score",
]

VARIANCE_THRESHOLD = 0.01      # min-max-scaled variance below this -> drop
CORRELATION_THRESHOLD = 0.98   # |Pearson r| above this -> drop one of the pair

_NUMERIC_KINDS = ("f", "i", "u")  # float, signed int, unsigned int


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the numeric molecular-descriptor columns (the model inputs)."""
    feature_cols = []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if any(pat in col for pat in METRIC_PATTERNS):
            continue
        if df[col].dtype.kind in _NUMERIC_KINDS:
            feature_cols.append(col)
    return feature_cols


def variance_filter(
    df: pd.DataFrame,
    features: Sequence[str],
    threshold: float = VARIANCE_THRESHOLD,
) -> List[str]:
    """Drop descriptors whose min-max-scaled variance is below ``threshold``."""
    X = df[list(features)].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    scaled = MinMaxScaler().fit_transform(X)
    variances = scaled.var(axis=0)
    return [f for f, v in zip(features, variances) if v >= threshold]


def correlation_filter(
    df: pd.DataFrame,
    features: Sequence[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> List[str]:
    """Iteratively drop one member of each highly correlated descriptor pair.

    Features are considered in order; when a feature correlates above
    ``threshold`` with one already kept, it is dropped. Deterministic given a
    fixed input ordering.
    """
    X = df[list(features)].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    corr = np.abs(np.corrcoef(X.values, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)

    kept: List[int] = []
    for i in range(len(features)):
        if all(corr[i, j] <= threshold for j in kept):
            kept.append(i)
    return [features[i] for i in kept]


def clean_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN and impute NaNs with per-column medians."""
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median())


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a clean feature matrix ``X`` and target vector ``y`` for ``target``.

    Rows with a missing target are dropped. Feature NaNs/infs are imputed with
    column medians (computed before any train/test split; this is a global
    median used only for imputation, not scaling).
    """
    X = clean_matrix(df[list(features)].copy())
    y = df[target].copy()
    valid = ~y.isna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    return X, y
