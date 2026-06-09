"""
Prevalence-target computation from mass-spectrometry time-series.
================================================================

For every peptide in every experiment we summarise its abundance trajectory
with four raw metrics, then aggregate across experiments (biological/technical
replicates) into the modelling targets used in the paper.

Raw per-experiment metrics
---------------------------
peak_intensity_raw   : maximum MS intensity observed across the time course.
auc_raw              : area under the intensity-vs-time curve (trapezoidal rule,
                       with a single zero-padding step at the first/last
                       observed time point so that transient peptides are not
                       over-counted).
total_increase_raw   : the final *observed* intensity (intensity at the last
                       time point at which the peptide is detected).  See the
                       note in the README on how this operationalises the
                       paper's "net growth" metric.
staying_power_raw    : a 0-1 persistence score combining mean relative
                       intensity, fraction of time points present, and an
                       early-appearance / sustained-area factor.

Aggregation across experiments
------------------------------
For each metric we report two aggregates per peptide:

``{metric}_avg``        mean of the *raw* metric across all experiments in which
                        the peptide was observed.
``{metric}_norm_avg``   the modelling target.  Within each experiment the raw
                        metric is log1p-transformed and min-max scaled to [0, 1];
                        these per-experiment normalised values are then averaged
                        across experiments.  Normalising *within* each experiment
                        before averaging removes run-to-run scale differences in
                        absolute MS intensity.

``{metric}_count`` / ``{metric}_norm_count`` record how many experiments
contributed to each average.

Running this module on the raw experiment CSVs in ``data/raw/experiments/``
reproduces the normalised peak-intensity and AUC targets of the shipped
``data/features_with_targets.csv`` at Pearson r > 0.99 (see
``scripts/01_compute_targets.py``). The shipped table is the canonical artifact.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RAW_METRIC_NAMES = [
    "peak_intensity_raw",
    "total_increase_raw",
    "auc_raw",
    "staying_power_raw",
]

# Short names used for the aggregated/normalised target columns.
TARGET_BASES = ["peak_intensity", "total_increase", "auc", "staying_power"]


# --------------------------------------------------------------------------- #
# Per-experiment raw metrics
# --------------------------------------------------------------------------- #
def _extract_time_points(columns: List[str]) -> Dict[str, float]:
    """Map each intensity column to its numeric time point.

    Columns are named like ``"Sum of Intensity Standard  T14"``; the integer
    after the final ``T`` is the day.  Falls back to column order if no ``T``
    token is found.
    """
    time_points: Dict[str, float] = {}
    for i, col in enumerate(columns):
        parts = col.split("T", 1) if "T" in col else [col]
        matched = False
        if len(parts) > 1:
            numbers = re.findall(r"\d+\.?\d*", parts[1])
            if numbers:
                try:
                    time_points[col] = float(numbers[0])
                    matched = True
                except ValueError:
                    pass
        if not matched:
            time_points[col] = float(i)
    return time_points


def compute_raw_metrics(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the four raw metrics for every peptide in one experiment.

    Parameters
    ----------
    experiment_df : DataFrame whose first column is the peptide sequence and
        whose remaining columns are intensities at successive time points.

    Returns
    -------
    DataFrame with columns ``sequence`` + :data:`RAW_METRIC_NAMES`.
    """
    if experiment_df.shape[1] < 2:
        raise ValueError("Experiment table needs a sequence column plus >=1 time point.")

    seq_col = experiment_df.columns[0]
    intensity_cols = list(experiment_df.columns[1:])
    times = _extract_time_points(intensity_cols)

    order = sorted(times.items(), key=lambda kv: kv[1])
    intensity_cols = [c for c, _ in order]
    times_array = np.array([t for _, t in order], dtype=float)
    avg_interval = float(np.mean(np.diff(times_array))) if len(times_array) > 1 else 1.0

    rows = []
    for _, row in experiment_df.iterrows():
        seq = row[seq_col]
        try:
            intensities = np.array(
                [
                    float(str(row[c]).replace(",", "").strip())
                    if pd.notna(row[c]) and str(row[c]).strip() not in ("-", "")
                    else 0.0
                    for c in intensity_cols
                ],
                dtype=float,
            )
        except (ValueError, TypeError):
            rows.append({seq_col: seq, **{m: np.nan for m in RAW_METRIC_NAMES}})
            continue

        valid = np.where(intensities > 0)[0]
        if valid.size == 0:
            rows.append({seq_col: seq, **{m: np.nan for m in RAW_METRIC_NAMES}})
            continue

        first_idx, last_idx = valid[0], valid[-1]
        peak_intensity = float(np.max(intensities))
        final_observed = float(intensities[last_idx])

        # --- AUC (trapezoid with single zero-pad before/after the trajectory) ---
        if valid.size == 1:
            peak_val = intensities[first_idx]
            at_boundary = first_idx in (0, len(times_array) - 1)
            base = avg_interval / 2.0 if at_boundary else avg_interval
            auc = 0.5 * base * peak_val
        else:
            padded_t = times_array.copy()
            padded_i = intensities.copy()
            if first_idx > 0:
                padded_t = np.insert(padded_t, first_idx, times_array[first_idx - 1])
                padded_i = np.insert(padded_i, first_idx, 0.0)
            if last_idx < len(times_array) - 1:
                pos = np.where(padded_t == times_array[last_idx])[0][0]
                padded_t = np.insert(padded_t, pos + 1, times_array[last_idx + 1])
                padded_i = np.insert(padded_i, pos + 1, 0.0)
            auc = float(np.trapezoid(padded_i, padded_t))

        # --- Staying power (0-1 persistence score) ---
        n_t = len(times_array)
        if peak_intensity > 0:
            intensity_presence = sum(i / peak_intensity for i in intensities) / n_t
        else:
            intensity_presence = 0.0
        presence_ratio = valid.size / n_t
        if n_t <= 1:
            early_factor = 0.5
        else:
            post_i = intensities[first_idx:]
            post_t = times_array[first_idx:]
            if len(post_i) > 1 and peak_intensity > 0:
                max_area = peak_intensity * (post_t[-1] - post_t[0])
                area_ratio = float(np.trapezoid(post_i, post_t)) / max_area if max_area > 0 else 0.5
            else:
                area_ratio = 0.5
            early_factor = (1.0 - 0.5 * first_idx / max(1, n_t - 1)) * area_ratio
        staying_power = 0.4 * intensity_presence + 0.4 * presence_ratio + 0.2 * early_factor

        rows.append(
            {
                seq_col: seq,
                "peak_intensity_raw": peak_intensity,
                "total_increase_raw": final_observed,
                "auc_raw": auc,
                "staying_power_raw": staying_power,
            }
        )

    out = pd.DataFrame(rows).rename(columns={seq_col: "sequence"})
    return out


# --------------------------------------------------------------------------- #
# Aggregation across experiments
# --------------------------------------------------------------------------- #
def _normalise_within_experiment(series: pd.Series) -> pd.Series:
    """log1p then min-max scale a single experiment's raw metric to [0, 1]."""
    logged = np.log1p(series.astype(float))
    lo, hi = logged.min(), logged.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (logged - lo) / (hi - lo)


def build_targets(raw_metric_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-experiment raw metrics into per-peptide modelling targets.

    Parameters
    ----------
    raw_metric_tables : mapping of experiment name -> raw-metric DataFrame
        (each as returned by :func:`compute_raw_metrics`).

    Returns
    -------
    DataFrame indexed implicitly by ``sequence`` with, for each metric base,
    columns ``{base}_avg``, ``{base}_count``, ``{base}_norm_avg`` and
    ``{base}_norm_count``.
    """
    raw_acc: Dict[str, Dict[str, List[float]]] = {b: {} for b in TARGET_BASES}
    norm_acc: Dict[str, Dict[str, List[float]]] = {b: {} for b in TARGET_BASES}

    for exp_name, table in raw_metric_tables.items():
        if "sequence" not in table.columns:
            warnings.warn(f"Skipping {exp_name}: no sequence column.")
            continue
        for base in TARGET_BASES:
            raw_col = f"{base}_raw"
            if raw_col not in table.columns:
                continue
            raw_vals = table[raw_col]
            norm_vals = _normalise_within_experiment(raw_vals)
            for seq, raw_v, norm_v in zip(table["sequence"], raw_vals, norm_vals):
                if pd.notna(raw_v):
                    raw_acc[base].setdefault(seq, []).append(float(raw_v))
                    norm_acc[base].setdefault(seq, []).append(float(norm_v))

    sequences = sorted({s for base in TARGET_BASES for s in raw_acc[base]})
    records = []
    for seq in sequences:
        rec = {"sequence": seq}
        for base in TARGET_BASES:
            raw_list = raw_acc[base].get(seq, [])
            norm_list = norm_acc[base].get(seq, [])
            rec[f"{base}_avg"] = float(np.mean(raw_list)) if raw_list else np.nan
            rec[f"{base}_count"] = len(raw_list)
            rec[f"{base}_norm_avg"] = float(np.mean(norm_list)) if norm_list else np.nan
            rec[f"{base}_norm_count"] = len(norm_list)
        records.append(rec)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# End-to-end driver
# --------------------------------------------------------------------------- #
def compute_targets_from_experiments(
    experiments_dir: Path,
    pattern: str = "*.csv",
    exclude_substrings: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute aggregated targets directly from a directory of experiment CSVs.

    Each CSV in ``experiments_dir`` matching ``pattern`` is treated as one
    experiment (first column = sequence, remaining columns = timed intensities).
    """
    experiments_dir = Path(experiments_dir)
    files = sorted(experiments_dir.glob(pattern))
    if exclude_substrings:
        files = [f for f in files if not any(s in f.name for s in exclude_substrings)]
    if not files:
        raise FileNotFoundError(f"No experiment CSVs matching {pattern!r} in {experiments_dir}")

    raw_tables = {}
    for f in files:
        df = pd.read_csv(f)
        raw_tables[f.stem] = compute_raw_metrics(df)
    return build_targets(raw_tables)


def merge_features_with_targets(
    features_csv: Path,
    targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join the per-peptide feature table with the targets on ``sequence``."""
    features = pd.read_csv(features_csv)
    merged = features.merge(targets_df, on="sequence", how="inner")
    return merged
