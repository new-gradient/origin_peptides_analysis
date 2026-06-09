"""
Hyper-parameter sweep and model training.
==========================================

Trains XGBoost models for the three prevalence targets, in either ``regression``
or ``classification`` mode, via a random hyper-parameter search.

**Model selection is performed on the validation split** - regression models by
validation Pearson correlation, classification models by validation ROC-AUC.
The 20% test split is held out and only evaluated for the single selected
configuration, so the reported test metrics are an unbiased estimate of
generalisation. (An earlier version of this code selected on the test split;
that is data leakage and has been corrected here.)

Data splits: 68% train / 12% validation / 20% test (the validation fraction is
15% of the non-test data). Classification splits are stratified on the binary
label. Early stopping during training uses the validation split.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .preprocessing import prepare_xy, select_feature_columns

RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15            # fraction of the post-test data -> ~12% overall
EARLY_STOPPING_ROUNDS = 50
QUANTILE_THRESHOLD = 0.75  # classification: top 25% is the positive class

TARGETS = ["peak_intensity_norm_avg", "auc_norm_avg", "total_increase_norm_avg"]

PARAM_SPACE: Dict[str, List] = {
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [500, 1000, 2000],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0, 0.1, 0.5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}


def short_name(target: str) -> str:
    return target.replace("_norm_avg", "").replace("_raw_avg", "")


# --------------------------------------------------------------------------- #
# Splitting and scaling
# --------------------------------------------------------------------------- #
def _split(X: pd.DataFrame, y: pd.Series, stratify: Optional[pd.Series]):
    strat1 = stratify
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strat1
    )
    strat2 = y_tr if stratify is not None else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=strat2
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def _scale(X_tr, X_val, X_te) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_tr),
        scaler.transform(X_val),
        scaler.transform(X_te),
        scaler,
    )


# --------------------------------------------------------------------------- #
# Single fit + evaluate
# --------------------------------------------------------------------------- #
def _fit_regression(df, features, target, params):
    X, y = prepare_xy(df, target, features)
    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X, y, stratify=None)
    X_tr_s, X_val_s, X_te_s, scaler = _scale(X_tr, X_val, X_te)

    model = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

    pred = {s: model.predict(x) for s, x in (("train", X_tr_s), ("val", X_val_s), ("test", X_te_s))}
    truth = {"train": y_tr, "val": y_val, "test": y_te}
    metrics = {"n_samples": len(y), "n_features": len(features), "best_iteration": int(model.best_iteration)}
    for s in ("train", "val", "test"):
        metrics[f"{s}_rmse"] = float(np.sqrt(mean_squared_error(truth[s], pred[s])))
        metrics[f"{s}_r2"] = float(r2_score(truth[s], pred[s]))
        metrics[f"{s}_pearson"] = float(pearsonr(truth[s], pred[s])[0])
    return metrics, model, scaler


def _fit_classification(df, target, features, params):
    X, y_cont = prepare_xy(df, target, features)
    threshold = float(y_cont.quantile(QUANTILE_THRESHOLD))
    y = (y_cont >= threshold).astype(int)

    X_tr, X_val, X_te, y_tr, y_val, y_te = _split(X, y, stratify=y)
    X_tr_s, X_val_s, X_te_s, scaler = _scale(X_tr, X_val, X_te)

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    train_params = {
        **params,
        "scale_pos_weight": (neg / pos) if pos else 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    model = xgb.XGBClassifier(**train_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

    sets = {"train": (X_tr_s, y_tr), "val": (X_val_s, y_val), "test": (X_te_s, y_te)}
    metrics = {
        "n_samples": int(len(y)),
        "n_features": len(features),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "threshold": threshold,
        "best_iteration": int(model.best_iteration),
    }
    for s, (Xs, ys) in sets.items():
        prob = model.predict_proba(Xs)[:, 1]
        pred = model.predict(Xs)
        metrics[f"{s}_accuracy"] = float(accuracy_score(ys, pred))
        metrics[f"{s}_f1"] = float(f1_score(ys, pred))
        metrics[f"{s}_precision"] = float(precision_score(ys, pred, zero_division=0))
        metrics[f"{s}_recall"] = float(recall_score(ys, pred))
        metrics[f"{s}_auc"] = float(roc_auc_score(ys, prob))
    return metrics, model, scaler


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    target: str
    mode: str
    best_params: dict
    best_metrics: dict
    model: object
    scaler: StandardScaler
    features: List[str]
    all_metrics: List[dict] = field(default_factory=list)


def _sample_params(rng: random.Random) -> dict:
    params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
    params["random_state"] = RANDOM_SEED
    return params


def run_sweep(
    df: pd.DataFrame,
    target: str,
    mode: str,
    features: Sequence[str],
    n_iterations: int = 500,
    seed: int = RANDOM_SEED,
) -> SweepResult:
    """Random-search ``n_iterations`` configs, selecting the best on validation.

    ``mode`` is ``"regression"`` (select by val Pearson r) or
    ``"classification"`` (select by val ROC-AUC).
    """
    if mode not in ("regression", "classification"):
        raise ValueError(f"mode must be regression or classification, got {mode!r}")
    rng = random.Random(seed)
    features = list(features)

    selection_key = "val_pearson" if mode == "regression" else "val_auc"
    best: Optional[SweepResult] = None
    all_metrics: List[dict] = []

    for _ in range(n_iterations):
        params = _sample_params(rng)
        if mode == "regression":
            metrics, model, scaler = _fit_regression(df, features, target, params)
        else:
            metrics, model, scaler = _fit_classification(df, target, features, params)
        metrics["params"] = params
        all_metrics.append(metrics)

        if best is None or metrics[selection_key] > best.best_metrics[selection_key]:
            best = SweepResult(
                target=target, mode=mode, best_params=params, best_metrics=metrics,
                model=model, scaler=scaler, features=features,
            )
    best.all_metrics = all_metrics
    return best


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_result(result: SweepResult, output_dir: Path) -> Path:
    """Persist model, scaler, params, feature list and metrics for one target."""
    output_dir = Path(output_dir)
    target_dir = output_dir / short_name(result.target)
    target_dir.mkdir(parents=True, exist_ok=True)
    sn = short_name(result.target)
    kind = "regressor" if result.mode == "regression" else "classifier"

    result.model.save_model(str(target_dir / f"best_{kind}_{sn}.json"))
    joblib.dump(result.scaler, target_dir / f"feature_scaler_{sn}.joblib")
    with open(target_dir / f"best_params_{sn}.json", "w") as f:
        json.dump(_json_safe(result.best_params), f, indent=2)
    with open(target_dir / f"feature_list_{sn}.txt", "w") as f:
        f.write("\n".join(result.features))

    metrics_out = _json_safe({k: v for k, v in result.best_metrics.items() if k != "params"})
    metrics_out["params"] = _json_safe(result.best_params)
    metrics_out["target"] = result.target
    metrics_out["mode"] = result.mode
    metrics_out["selection"] = "val_pearson" if result.mode == "regression" else "val_auc"
    metrics_out["n_sweep_iterations"] = len(result.all_metrics)
    metrics_out["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    if result.mode == "classification":
        metrics_out["quantile_threshold"] = QUANTILE_THRESHOLD
    with open(target_dir / f"results_{sn}.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    return target_dir
