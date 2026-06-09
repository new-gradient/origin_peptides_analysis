#!/usr/bin/env python3
"""
Compute molecular descriptors for a set of peptide sequences.

The full descriptor set in the paper has two parts:

* **Sequence-based descriptors** - computed directly from the amino-acid
  sequence (composition, dipeptide motifs, charge, hydrophobicity, patterns,
  fragmentation). These require no structure and are recomputed here.
* **Structure-based descriptors** - secondary structure, solvent accessibility,
  contact maps, geometry and backbone dihedrals, derived from AlphaFold2 models
  predicted with ColabFold. These require one ``.pdb`` per peptide and the
  ColabFold install; they are *not* recomputed here (the shipped
  ``data/features_with_targets.csv`` already contains them). See
  ``peptide_ml/features/folding/`` and ``peptide_ml.features.peptide_features``
  for the full structure pipeline.

This script computes the sequence-based descriptors for every peptide in the
canonical table and writes them to ``data/processed/sequence_features.csv`` -
a runnable, dependency-light demonstration of the descriptor code (column
naming here is ``ClassName_descriptor``; the shipped table uses the original
``group_descriptor`` convention, but the underlying values match).

Usage
-----
    python scripts/00_build_features.py
    python scripts/00_build_features.py --limit 200      # quick subset
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# The descriptor modules log verbosely at INFO; quieten them for batch runs.
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from peptide_ml.features.non_folding import (  # noqa: E402
    AACompositionFeature, AlternatingPatternFeature, AmphipathicityFeature,
    ChainLengthFeature, ChargeFeature, DipeptideCompositionFeature,
    HydrolysisProneBondFeature, HydrophobicityFeature, InternalCleavageSitesFeature,
    IsoelectricPointFeature, LocalChargeFeature, MolecularWeightFeature,
    PeriodicPatternsFeature, PositionalFeature, PredictedAggregationFeature,
    PredictedSolubilityFeature, ResiduePropertyFeature, SequenceEntropyFeature,
    SpecializedResidueFeature, TrypticTerminalFeature,
)

DATA = REPO / "data" / "features_with_targets.csv"
OUT = REPO / "data" / "processed" / "sequence_features.csv"

# One instance of every sequence-based (non-structural) descriptor family.
SEQUENCE_FEATURES = [
    ChainLengthFeature(), MolecularWeightFeature(), AACompositionFeature(),
    DipeptideCompositionFeature(), SequenceEntropyFeature(), IsoelectricPointFeature(),
    ChargeFeature(), LocalChargeFeature(), HydrophobicityFeature(), AmphipathicityFeature(),
    PredictedSolubilityFeature(), PredictedAggregationFeature(), ResiduePropertyFeature(),
    PeriodicPatternsFeature(), PositionalFeature(), SpecializedResidueFeature(),
    AlternatingPatternFeature(), TrypticTerminalFeature(), InternalCleavageSitesFeature(),
    HydrolysisProneBondFeature(),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N peptides")
    args = ap.parse_args()

    sequences = pd.read_csv(DATA, usecols=["sequence"])["sequence"].tolist()
    if args.limit:
        sequences = sequences[: args.limit]
    print(f"Computing sequence descriptors for {len(sequences)} peptides "
          f"across {len(SEQUENCE_FEATURES)} descriptor families...")

    rows = []
    for seq in tqdm(sequences):
        rec = {"sequence": seq}
        for feat in SEQUENCE_FEATURES:
            prefix = feat.__class__.__name__.replace("Feature", "")
            try:
                for k, v in feat.calculate_ml_ready(seq).items():
                    rec[f"{prefix}_{k}"] = v
            except Exception as e:  # keep going; report once per failure
                rec[f"{prefix}_ERROR"] = str(e)
        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]-1} descriptors -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
