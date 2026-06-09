"""
Figure 3B-D peptide-group panels.
=================================

The paper highlights three sequence families whose amplification is driven by
distinct molecular themes.  Each panel is a focused SHAP heatmap over the
peptides in the family, restricted to the descriptors relevant to that theme:

* **Group 1 (Fig 3B)** - dipeptide motifs: hydrophobic residues adjacent to
  polar/special residues (the ``APSTY...`` family).
* **Group 2 (Fig 3C)** - local charge state and charge asymmetry along the
  peptide (the ``ED...`` family).
* **Group 3 (Fig 3D)** - alternation of rigid and flexible residues
  (the ``EG.../EH...`` family).

The peptide membership and the per-group descriptor lists below reproduce the
sets used to build the published figures.  Input is the per-peptide SHAP table
written by :func:`peptide_ml.shap_analysis.compute_shap_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


@dataclass(frozen=True)
class PeptideGroup:
    key: str
    panel: str
    title: str
    description: str
    sequences: List[str]
    shap_features: List[str]


GROUP1 = PeptideGroup(
    key="group1_dipeptide_hydrophobic_special",
    panel="3B",
    title="Hydrophobic–special dipeptide motifs",
    description="Amplification linked to hydrophobic residues adjacent to polar/special residues.",
    sequences=[
        "APSTYGGAKTNGSSK", "APSTYGGEVKSSR", "APSTYGGGLFWNR", "APSTYGGGLVSSSR",
        "APSTYGGGLVSSSSR", "APSTYGNLSVSSR", "APSTYGNLWFNR", "APSTYGQVSSSVR",
        "APSTYNLNFPSK",
    ],
    shap_features=[
        "SHAP_dipeptide_dipep_hydrophobic_special",
        "SHAP_dipeptide_dipep_hydrophobic_hydrophobic",
        "SHAP_dipeptide_dipep_special_special",
        "SHAP_dipeptide_dipep_special_hydrophobic",
        "SHAP_alternating_pattern_rigid_flexible_alternation",
        "SHAP_alternating_pattern_big_fraction",
        "SHAP_alternating_pattern_small_fraction",
    ],
)

GROUP2 = PeptideGroup(
    key="group2_charge_states",
    panel="3C",
    title="Local charge asymmetry",
    description="Amplification linked to local charge state and charge asymmetry along the peptide.",
    sequences=[
        "EDEYYNAK", "EDFVFKPLVEEPQNLLK", "EDHDCGYLEGGK", "EDKRHHKEEGWK",
        "EDLTYLYK", "EDLVLTLSQTDLEMKLESLNEELAYMKK", "EDNLNVVENGEQFLSASK",
        "EDNWRYTLSDQLLAK", "EDVGVTLVFPLTPR", "EDVGVTLVLNLTNR", "EDVGVTLVNLLQSR",
        "EDVGVTLVNLLTNR", "EDVGVTLVQVLTNR",
    ],
    shap_features=[
        "SHAP_charge_charge_ph2_per_residue",
        "SHAP_charge_charge_ph7_per_residue",
        "SHAP_charge_charge_ph12_per_residue",
        "SHAP_local_charge_max_local_charge",
        "SHAP_local_charge_min_local_charge",
        "SHAP_local_charge_charge_variation",
        "SHAP_local_charge_charge_asymmetry",
        "SHAP_isoelectric_isoelectric_point",
    ],
)

GROUP3 = PeptideGroup(
    key="group3_rigidity_flexibility",
    panel="3D",
    title="Rigid–flexible alternation",
    description="Amplification linked to alternation between rigid and flexible residues.",
    sequences=[
        "EGVPRYTK", "EGWYDEEFGFSAR", "EGYDAGYLEGGK", "EHLMLDGLEGGK",
        "EHLVLMEDLDSQADK", "EHLVLMEDLDTNADK", "EHNLEPYFESFLNNLR",
        "EHSKENTELKASLEEAEASLEHEEGK", "EHVWFHFELHMLNLK",
    ],
    shap_features=[
        "SHAP_alternating_pattern_rigid_flexible_alternation",
        "SHAP_alternating_pattern_big_small_alternation",
        "SHAP_alternating_pattern_big_fraction",
        "SHAP_alternating_pattern_small_fraction",
        "SHAP_alternating_pattern_rigid_fraction",
        "SHAP_alternating_pattern_flexible_fraction",
    ],
)

GROUPS = [GROUP1, GROUP2, GROUP3]


def _filter_sequences(shap_table: pd.DataFrame, sequences: List[str]) -> pd.DataFrame:
    mask = shap_table["Sequence"].isin(sequences)
    found = shap_table[mask].copy()
    if len(found) < len(sequences):  # retry case-insensitively
        upper = {s.upper() for s in sequences}
        found = shap_table[shap_table["Sequence"].str.upper().isin(upper)].copy()
    return found.sort_values("Target_Intensity")


def group_panel(
    shap_table: pd.DataFrame,
    group: PeptideGroup,
    output_path: Path,
) -> pd.DataFrame:
    """Render one focused SHAP heatmap (peptides x theme descriptors).

    Returns the filtered per-peptide SHAP sub-table used for the panel.
    """
    sub = _filter_sequences(shap_table, group.sequences)
    if sub.empty:
        raise ValueError(f"No peptides from {group.key} found in SHAP table.")

    feats = [f for f in group.shap_features if f in sub.columns]
    data = sub[feats].to_numpy()  # rows = peptides, cols = descriptors
    nice_labels = [f.replace("SHAP_", "") for f in feats]

    vmax = np.percentile(np.abs(data), 99) if data.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(6, len(feats) * 1.1), max(3, len(sub) * 0.5)))
    sns.heatmap(
        data, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        yticklabels=sub["Sequence"].tolist(), xticklabels=nice_labels,
        cbar_kws={"label": "SHAP value"}, ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_title(f"Figure {group.panel}: {group.title}\n{group.description}",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return sub
