"""
peptide_ml
==========

Machine-learning analysis pipeline for the abiogenesis peptide-templating study
("The Spontaneous Evolution of Biology").

The package turns raw mass-spectrometry time-series for thousands of peptides
into three prevalence targets (peak intensity, area-under-curve, total increase),
computes molecular descriptors for every peptide sequence, and trains/interprets
XGBoost regression and classification models with SHAP.

Sub-modules
-----------
features        : molecular-descriptor extraction (sequence + structure)
targets         : compute prevalence metrics from MS time-series and normalise them
preprocessing   : variance/correlation feature filtering, imputation, scaling
train           : hyper-parameter sweep with validation-based model selection
shap_analysis   : SHAP value computation and the Figure 3A heatmap
figures         : peptide-group tables (Figure 3B-D) and the Table S2 glossary
"""

__version__ = "1.0.0"
