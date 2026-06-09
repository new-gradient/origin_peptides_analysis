# Data

## `features_with_targets.csv` — canonical modelling table

One row per peptide (6,954 peptides). Columns:

- `ID`, `sequence` — peptide identifier and amino-acid sequence.
- **Molecular descriptors** (119 numeric columns) — the sequence- and
  structure-based features used as model inputs. Structure-based descriptors
  were derived from AlphaFold2/ColabFold models (see below).
- **Targets** — for each prevalence metric (`peak_intensity`, `auc`,
  `total_increase`, `staying_power`):
  - `{metric}_avg` — mean of the raw metric across experiments.
  - `{metric}_norm_avg` — the modelling target: the metric log1p-transformed
    and min-max scaled **within each experiment**, then averaged across
    experiments.
  - `{metric}_count`, `{metric}_norm_count` — number of contributing experiments.
- `template_score` — a derived composite metric (not used by the shipped models).

This is the authoritative table that the models in `models/` were trained on
and that reproduces the paper's reported numbers.

## `raw/experiments/*.csv` — raw mass-spectrometry time-series

One CSV per experiment (condition × replicate). First column is the peptide
sequence; remaining columns are summed MS intensities at successive time points
(`T0`, `T7`, … days). `scripts/01_compute_targets.py` turns these into the
prevalence targets and confirms it reproduces the `*_norm_avg` columns above
(peak intensity and AUC reproduce at r > 0.99).

## Structure-based descriptors / ColabFold

The structure descriptors (helix/sheet/coil content, solvent accessibility,
contact maps, radius of gyration, asphericity, backbone dihedrals, …) require an
AlphaFold2 model per peptide, predicted with
[ColabFold](https://github.com/sokrypton/ColabFold). The predicted `.pdb`
structures (~tens of GB for the full set) are **not** shipped; the resulting
descriptors are included in `features_with_targets.csv`. The extraction code is
in `peptide_ml/features/folding/`.
