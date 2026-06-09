# Methodology

This document expands on the machine-learning methods summarised in the
manuscript and maps each step to the code.

## 1. Targets — from MS trajectories to prevalence metrics

`peptide_ml/targets.py`

Each peptide's abundance trajectory in one experiment is summarised by:

- **peak_intensity_raw** — maximum intensity over the time course.
- **auc_raw** — trapezoidal area under the intensity-vs-time curve, with a single
  zero-padding step immediately before the first and after the last observation
  so that transient peptides are not over-counted.
- **total_increase_raw** — the final observed intensity (see note in the README
  on how this operationalises "net growth").
- **staying_power_raw** — a 0–1 persistence score combining mean relative
  intensity, fraction of time points present, and an early-appearance/sustained
  area factor.

These per-experiment raw metrics are aggregated across experiments
(replicates):

- `{metric}_avg` — mean of the raw metric.
- `{metric}_norm_avg` (**the modelling target**) — the raw metric is
  log1p-transformed and min-max scaled to [0, 1] *within each experiment*, then
  averaged across experiments. Per-experiment normalisation removes run-to-run
  differences in absolute MS intensity, which would otherwise dominate the
  average.

## 2. Features — molecular descriptors

`peptide_ml/features/`

**Sequence-based** (`non_folding/`): chain length and molecular weight; amino-acid
and dipeptide composition (grouped by physicochemistry); isoelectric point and
net charge at pH 2/7/12; local charge statistics and asymmetry; Kyte–Doolittle
and Eisenberg hydrophobicity and hydrophobic moment; amphipathicity; periodic and
alternating sequence patterns; positional terminal properties; specialised
residue fractions (aromatic, oxidation-sensitive, structure-breaking, tiny);
and tryptic/fragmentation descriptors.

**Structure-based** (`folding/`): from AlphaFold2 models predicted with ColabFold —
secondary-structure content (helix/sheet/turn/coil via DSSP), solvent
accessibility and buried/exposed fractions, contact order and short/long-range
contact densities, radius of gyration, asphericity, elongation, molecular volume,
and backbone-dihedral regularity.

## 3. Feature preprocessing — 133 → 119

`peptide_ml/preprocessing.py`

1. Drop identifiers and target/metric columns, keep numeric descriptors.
2. **Variance filter** — remove descriptors whose min-max-scaled variance is
   below 0.01.
3. **Correlation filter** — for any pair with |Pearson r| > 0.98, drop one.
4. **Imputation** — replace inf/NaN with the column median.
5. **Standardisation** — z-score using *training-split* statistics only.

## 4. Model training and selection

`peptide_ml/train.py`

- Split: 68% train / 12% validation / 20% test (stratified on the binary label
  for classification).
- Model: XGBoost (`XGBRegressor` / `XGBClassifier`), early stopping on validation.
- Search: 500 random hyper-parameter configurations per target over depth (3–10),
  learning rate (0.01–0.1), `reg_alpha`, `reg_lambda`, `gamma`, `subsample`,
  `colsample_bytree`, `min_child_weight`.
- **Selection on validation**: regression by validation Pearson *r*,
  classification by validation ROC-AUC. The test split is scored once, for the
  selected configuration — so test metrics estimate generalisation without
  selection bias. (A previous iteration of this analysis selected on the test
  split; selecting on validation is the corrected, leakage-free procedure used
  here.)
- Classification uses `scale_pos_weight = n_neg / n_pos` for the 3:1 imbalance.

## 5. Interpretation — SHAP

`peptide_ml/shap_analysis.py`, `peptide_ml/figures.py`

TreeExplainer SHAP values are computed for every peptide from the
validation-selected regression model. Figure 3A is the global heatmap (features
ordered by mean |SHAP|, peptides ordered by target value). Figures 3B–D are
focused heatmaps over three sequence families, restricted to the descriptors
relevant to each family's theme (hydrophobic–special dipeptides; local charge
asymmetry; rigid–flexible alternation).

## 6. Feature glossary — Table S2

`peptide_ml/feature_glossary.py` emits a plain-language definition for every
descriptor, written to `results/tableS2_feature_glossary.{csv,xlsx}`.
