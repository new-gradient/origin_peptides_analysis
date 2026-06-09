# Peptide-amplification machine-learning analysis

Code and data for the machine-learning analysis in **"The Spontaneous Evolution
of Biology"** (ten Have *et al.*). This repository reproduces the modelling and
interpretation behind **Figure 3** and **Table S2**: predicting which peptides
amplify and persist in abiogenesis experiments from their molecular descriptors,
and explaining those predictions with SHAP.

> Scope: this repo covers only the machine-learning analysis of the quantitative
> mass-spectrometry peptide data. The wet-lab, microscopy and energy-measurement
> components of the paper are not part of this codebase.

---

## What the analysis does

Thousands of peptides were quantified by mass spectrometry over a 42-day time
course. For each peptide we summarise its abundance trajectory with three
prevalence metrics used as proxies for templating/amplification success:

| Target | Definition |
|---|---|
| **Peak intensity** | maximum MS intensity over the time course |
| **AUC** | area under the intensity-vs-time curve (cumulative presence) |
| **Total increase** | the final observed intensity (operational "net growth", see note below) |

Each peptide is described by **119 molecular descriptors** (sequence- and
structure-based). We train **XGBoost** models — both regression (predict the
continuous normalised metric) and classification (predict the top-25%) — with a
500-configuration random hyper-parameter search, and interpret them with
**SHAP** to identify which molecular properties drive amplification.

---

## Repository layout

```
peptide_ml/                 importable package
├── targets.py              MS time-series -> prevalence metrics -> normalised targets
├── preprocessing.py        feature selection, variance/correlation filtering, scaling
├── train.py                hyper-parameter sweep, validation-based model selection
├── shap_analysis.py        SHAP values + Figure 3A heatmap
├── figures.py              Figure 3B-D peptide-group panels
├── feature_glossary.py     Table S2 feature definitions
└── features/               molecular-descriptor extraction
    ├── non_folding/         sequence-based descriptors (no structure needed)
    └── folding/             structure-based descriptors (from AlphaFold2/ColabFold)

scripts/
├── 00_build_features.py    (re)compute sequence descriptors from peptide sequences
├── 01_compute_targets.py   recompute targets from raw MS data (provenance check)
├── 02_train_models.py      run the sweeps, save best models + metrics
├── 03_shap_and_figures.py  SHAP, Figure 3 panels, Table S2
└── run_all.sh              end-to-end driver

data/
├── features_with_targets.csv   canonical modelling table (descriptors + targets)
└── raw/experiments/            raw per-experiment MS time-series

models/        trained models, scalers, hyper-parameters, metrics (per target)
results/       SHAP tables, figures, Table S2 glossary
tests/         lightweight pipeline sanity checks
```

---

## Installation

Python 3.9+ recommended. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned to the versions used for the paper
pip install -e .                        # makes `import peptide_ml` available
```

---

## Reproducing the results

```bash
# Full pipeline (targets check -> train -> SHAP/figures/tables)
bash scripts/run_all.sh

# …or step by step:
python scripts/01_compute_targets.py        # reproduce targets from raw MS data
python scripts/02_train_models.py            # 500-config sweep, both modes
python scripts/03_shap_and_figures.py        # SHAP + Figure 3 + Table S2
```

A quick end-to-end check (small sweep) finishes in a couple of minutes:

```bash
python scripts/02_train_models.py --iterations 20
```

The shipped `models/` were produced by the full 500-config sweep; re-running with
the default seed reproduces them.

---

## Methods detail

**Targets.** Raw metrics are computed per experiment, then log1p-transformed and
min-max scaled *within each experiment* before being averaged across experiments
(`peptide_ml/targets.py`). Normalising within each experiment removes run-to-run
differences in absolute MS intensity. For classification, peptides above the 75th
percentile of a metric are the positive class, with inverse-frequency class
weighting for the 3:1 imbalance.

**Features.** 133 descriptors are computed and reduced to 119 by removing
near-constant features (min-max-scaled variance < 0.01) and one of each highly
correlated pair (|Pearson r| > 0.98). Sequence-based descriptors come from the
amino-acid sequence; structure-based descriptors come from AlphaFold2/ColabFold
models. The shipped table already contains the 119-feature set; the reduction
helpers are in `peptide_ml/preprocessing.py`.

**Model training.** 68% train / 12% validation / 20% test split (stratified for
classification). XGBoost with early stopping on the validation set. A random
search over 500 configurations explores tree depth (3–10), learning rate
(0.01–0.1), L1/L2 regularisation and subsampling.

**Model selection (important).** Hyper-parameter configurations are selected on
the **validation** split — regression by validation Pearson *r*, classification
by validation ROC-AUC — and the 20% test split is evaluated only once, for the
selected configuration. The reported test metrics are therefore an unbiased
estimate of generalisation.

**SHAP.** TreeExplainer SHAP values decompose each prediction into per-feature
contributions, enabling the Figure 3A global heatmap and the Figure 3B–D
peptide-group panels.

---

## Results

Held-out test set; models selected on the validation split from a 500-config
random search (`scripts/02_train_models.py`, seed 42).

**Regression** (predicting the continuous normalised metric):

| Target | Validation Pearson r | Test Pearson r | Test R² |
|---|---|---|---|
| Peak intensity | 0.473 | **0.440** | 0.188 |
| AUC | 0.557 | **0.423** | 0.170 |
| Total increase | 0.205 | **0.160** | 0.024 |

**Classification** (predicting the top 25%, Q75 threshold):

| Target | Validation ROC-AUC | Test ROC-AUC | Test F1 | Test accuracy |
|---|---|---|---|---|
| Peak intensity | 0.728 | **0.729** | 0.504 | 0.708 |
| AUC | 0.732 | **0.709** | 0.489 | 0.743 |
| Total increase | 0.597 | **0.557** | 0.215 | 0.695 |

Peak intensity and AUC are the predictable metrics (test r ≈ 0.42–0.44,
ROC-AUC ≈ 0.71–0.73): the 119 molecular descriptors carry a real, if moderate,
signal for peptide amplification. Total increase is close to noise, consistent
with trypsin digestion disrupting growth-trajectory measurement.

> These are validation-selected, leakage-free estimates. They are slightly lower
> than figures from an earlier analysis that selected hyper-parameters on the
> test split; the present numbers are the defensible ones to report.

---

## Notes and caveats

- **`total_increase` definition (code ↔ manuscript).** In code this metric is the
  *final observed intensity* of each peptide — the intensity at the last time
  point at which it is detected — which is what the published targets and models
  were built from (`peptide_ml/targets.py`). The manuscript phrases it as "the
  difference between final and initial intensity values". These coincide under the
  experimental baseline, where peptides are absent at T0 (initial intensity ≈ 0),
  so *final − initial ≈ final intensity*; they differ only for peptides that
  disappear before the final time point. The non-negative (final-intensity) form
  is also what makes the log1p normalisation well defined. **Recommended
  manuscript edit:** describe this metric as "the final observed intensity (net
  accumulation relative to the zero baseline at T0)" so the text matches the
  computed quantity. It is the weakest-predicted of the three targets regardless.
- **Target provenance.** `01_compute_targets.py` reproduces the normalised
  peak-intensity and AUC targets from the raw MS data at r > 0.99; the small
  residual reflects experiment-subset choices in the original analysis. The
  shipped `features_with_targets.csv` is canonical.
- **Structure descriptors** require ColabFold and per-peptide `.pdb` files, which
  are not shipped (they are large and infeasible to regenerate for a reviewer);
  the resulting descriptors are included in the canonical table.

---

## License

Source-available under the **PolyForm Strict License 1.0.0** (see `LICENSE`).
You may view and run this software for noncommercial purposes (e.g. evaluating
the associated publication), but may **not** redistribute it, modify it, or
create derivative works. Confirm the copyright holder named in `LICENSE` before
release.
