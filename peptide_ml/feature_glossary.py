#!/usr/bin/env python3
"""
Generate Peptide Feature Glossary

Creates a CSV glossary file with scientific definitions for all peptide features
used in the ML model. Intended for biologist collaborators.

Usage:
    python generate_feature_glossary.py --data path/to/features.csv --output glossary.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


# Patterns that indicate metric/target columns (not features)
METRIC_PATTERNS = [
    'peak_intensity_', 'auc_', 'staying_power_',
    'total_increase_', 'template_score'
]

# Non-feature columns
NON_FEATURE_COLS = ['ID', 'sequence']

# Amino acid names
AA_NAMES = {
    'A': 'Alanine',
    'C': 'Cysteine',
    'D': 'Aspartic acid',
    'E': 'Glutamic acid',
    'F': 'Phenylalanine',
    'G': 'Glycine',
    'H': 'Histidine',
    'I': 'Isoleucine',
    'K': 'Lysine',
    'L': 'Leucine',
    'M': 'Methionine',
    'N': 'Asparagine',
    'P': 'Proline',
    'Q': 'Glutamine',
    'R': 'Arginine',
    'S': 'Serine',
    'T': 'Threonine',
    'V': 'Valine',
    'W': 'Tryptophan',
    'Y': 'Tyrosine',
}

# Dipeptide property group descriptions
DIPEPTIDE_GROUPS = {
    'hydrophobic': 'hydrophobic residues (A, V, L, I, F, M, W)',
    'polar': 'polar uncharged residues (S, T, N, Q)',
    'charged_pos': 'positively charged residues (K, R)',
    'charged_neg': 'negatively charged residues (D, E)',
    'special': 'special residues (C, Y, H, G, P)',
}


def get_feature_definitions() -> dict:
    """Return a dictionary mapping feature names to their scientific definitions."""

    definitions = {}

    # ==========================================================================
    # BASIC / STRUCTURAL FEATURES
    # ==========================================================================
    definitions['chain_length_chain_length'] = (
        "Number of amino acid residues in the peptide sequence (sequence length)"
    )
    definitions['chain_length_chain_length_log'] = (
        "Log-transformed sequence length; reduces the influence of very long sequences"
    )
    definitions['molecular_weight_molecular_weight'] = (
        "Total molecular mass of the peptide in Daltons (Da), calculated from amino acid composition"
    )
    definitions['molecular_weight_molecular_weight_per_residue'] = (
        "Average molecular mass per amino acid residue (total MW / sequence length)"
    )

    # ==========================================================================
    # AMINO ACID COMPOSITION (20 features)
    # ==========================================================================
    for aa_code, aa_name in AA_NAMES.items():
        feature_name = f'aa_composition_aa_fraction_{aa_code}'
        definitions[feature_name] = (
            f"Fraction of {aa_name} ({aa_code}) residues in the sequence (0-1 scale)"
        )

    # ==========================================================================
    # DIPEPTIDE COMPOSITION (25 features)
    # ==========================================================================
    # Generate all dipeptide combinations
    groups = ['hydrophobic', 'polar', 'charged_pos', 'charged_neg', 'special']
    for g1 in groups:
        for g2 in groups:
            feature_name = f'dipeptide_dipep_{g1}_{g2}'
            g1_desc = DIPEPTIDE_GROUPS[g1]
            g2_desc = DIPEPTIDE_GROUPS[g2]
            definitions[feature_name] = (
                f"Frequency of dipeptide motifs where {g1_desc} are followed by {g2_desc}"
            )

    # ==========================================================================
    # SEQUENCE ENTROPY
    # ==========================================================================
    definitions['entropy_sequence_entropy'] = (
        "Shannon entropy of amino acid composition; measures sequence diversity "
        "(0 = single amino acid type, ~4.3 bits = uniform distribution of all 20 amino acids)"
    )

    # ==========================================================================
    # CHARGE FEATURES
    # ==========================================================================
    definitions['isoelectric_isoelectric_point'] = (
        "Isoelectric point (pI): the pH at which the peptide has zero net electrical charge (range 0-14)"
    )
    definitions['charge_charge_ph2_per_residue'] = (
        "Net electrical charge per residue at pH 2 (acidic conditions); positive values indicate cationic peptide"
    )
    definitions['charge_charge_ph7_per_residue'] = (
        "Net electrical charge per residue at pH 7 (physiological pH); indicates peptide polarity at neutral pH"
    )
    definitions['charge_charge_ph12_per_residue'] = (
        "Net electrical charge per residue at pH 12 (basic conditions); negative values indicate anionic peptide"
    )
    definitions['local_charge_max_local_charge'] = (
        "Maximum local charge within a sliding window along the sequence; indicates regions of high charge density"
    )
    definitions['local_charge_min_local_charge'] = (
        "Minimum local charge within a sliding window; indicates regions of negative charge accumulation"
    )
    definitions['local_charge_charge_variation'] = (
        "Standard deviation of local charges along the sequence; high values indicate uneven charge distribution"
    )
    definitions['local_charge_charge_asymmetry'] = (
        "Difference in charge between N-terminal and C-terminal regions; measures charge polarity along the peptide"
    )

    # ==========================================================================
    # HYDROPHOBICITY FEATURES
    # ==========================================================================
    definitions['hydrophobicity_kd_mean'] = (
        "Mean hydrophobicity using the Kyte-Doolittle scale; positive values indicate hydrophobic character"
    )
    definitions['hydrophobicity_kd_std'] = (
        "Standard deviation of Kyte-Doolittle hydrophobicity along the sequence; measures hydrophobicity variation"
    )
    definitions['hydrophobicity_eisenberg_mean'] = (
        "Mean hydrophobicity using the Eisenberg consensus scale; alternative measure of overall hydrophobic character"
    )
    definitions['hydrophobicity_eisenberg_std'] = (
        "Standard deviation of Eisenberg hydrophobicity; measures the variation in hydrophobicity along the sequence"
    )
    definitions['hydrophobicity_hydrophobic_moment'] = (
        "Hydrophobic moment: measures amphipathicity (segregation of hydrophobic/hydrophilic faces), "
        "important for membrane interaction potential"
    )

    # ==========================================================================
    # AMPHIPATHICITY FEATURES
    # ==========================================================================
    definitions['amphipathicity_max_amphipathicity'] = (
        "Maximum local amphipathicity score; indicates peak segregation of hydrophobic and hydrophilic residues"
    )
    definitions['amphipathicity_mean_amphipathicity'] = (
        "Average amphipathicity along the sequence; overall measure of hydrophobic/hydrophilic face separation"
    )
    definitions['amphipathicity_amphipathicity_variation'] = (
        "Standard deviation of local amphipathicity; measures consistency of amphipathic character"
    )

    # ==========================================================================
    # PERIODIC PATTERN FEATURES
    # ==========================================================================
    definitions['periodic_patterns_repeats_2_density'] = (
        "Density of 2-residue repeating patterns (e.g., XYXY); indicates sequence periodicity"
    )
    definitions['periodic_patterns_repeats_3_density'] = (
        "Density of 3-residue repeating patterns (e.g., XYZXYZ); common in structural motifs"
    )
    definitions['periodic_patterns_repeats_4_density'] = (
        "Density of 4-residue repeating patterns; may indicate regular secondary structure"
    )
    definitions['periodic_patterns_repeat_ratio_2_3'] = (
        "Ratio of 2-mer to 3-mer repeat densities; characterizes the dominant repeat pattern type"
    )

    # ==========================================================================
    # RESIDUE PROPERTIES
    # ==========================================================================
    definitions['residue_properties_average_mass'] = (
        "Average molecular mass of individual amino acid residues in Daltons"
    )
    definitions['residue_properties_mass_std'] = (
        "Standard deviation of residue masses; indicates diversity in residue sizes"
    )
    definitions['residue_properties_instability_index'] = (
        "Instability index predicting in vivo half-life; values >40 suggest unstable proteins"
    )
    definitions['residue_properties_mass_per_charge'] = (
        "Ratio of total mass to number of charged residues (log-transformed); indicates charge density"
    )

    # ==========================================================================
    # POSITIONAL FEATURES
    # ==========================================================================
    definitions['positional_n_term_charged_frac'] = (
        "Fraction of charged residues (D, E, K, R) in the N-terminal region (first 5 residues)"
    )
    definitions['positional_c_term_charged_frac'] = (
        "Fraction of charged residues (D, E, K, R) in the C-terminal region (last 5 residues)"
    )
    definitions['positional_n_term_hydrophobic_frac'] = (
        "Fraction of hydrophobic residues in the N-terminal region; affects membrane insertion"
    )
    definitions['positional_c_term_hydrophobic_frac'] = (
        "Fraction of hydrophobic residues in the C-terminal region"
    )
    definitions['positional_terminal_charge_bias'] = (
        "Asymmetry in charge distribution between N-terminal and C-terminal regions"
    )

    # ==========================================================================
    # SPECIALIZED RESIDUE FEATURES
    # ==========================================================================
    # Aromatic residues (F, W, Y)
    definitions['specialized_residue_aromatic_fraction'] = (
        "Fraction of aromatic residues (Phe, Trp, Tyr); important for π-stacking and hydrophobic interactions"
    )
    definitions['specialized_residue_aromatic_clustering'] = (
        "Clustering coefficient of aromatic residues; measures tendency to cluster together in the sequence"
    )
    definitions['specialized_residue_aromatic_max_density'] = (
        "Maximum local density of aromatic residues in a sliding window"
    )

    # Oxidation-sensitive residues (C, M, W)
    definitions['specialized_residue_oxidation_sensitive_fraction'] = (
        "Fraction of oxidation-sensitive residues (Cys, Met, Trp); indicates susceptibility to oxidative damage"
    )
    definitions['specialized_residue_oxidation_sensitive_clustering'] = (
        "Clustering of oxidation-sensitive residues; clustered residues may be more vulnerable to oxidation"
    )
    definitions['specialized_residue_oxidation_sensitive_max_density'] = (
        "Maximum local density of oxidation-sensitive residues"
    )

    # Structure-breaking residues (G, P)
    definitions['specialized_residue_structure_breaking_fraction'] = (
        "Fraction of structure-breaking residues (Gly, Pro); these disrupt regular secondary structures"
    )
    definitions['specialized_residue_structure_breaking_clustering'] = (
        "Clustering of Gly and Pro residues; affects flexibility and structural disorder"
    )
    definitions['specialized_residue_structure_breaking_max_density'] = (
        "Maximum local density of structure-breaking residues"
    )

    # Tiny residues (A, G, S)
    definitions['specialized_residue_tiny_fraction'] = (
        "Fraction of tiny residues (Ala, Gly, Ser); small side chains allow tight packing"
    )
    definitions['specialized_residue_tiny_clustering'] = (
        "Clustering coefficient of tiny residues in the sequence"
    )
    definitions['specialized_residue_tiny_max_density'] = (
        "Maximum local density of tiny residues"
    )

    # ==========================================================================
    # ALTERNATING PATTERN FEATURES
    # ==========================================================================
    definitions['alternating_pattern_big_small_alternation'] = (
        "Score measuring alternation between big (F, W, Y, K, R, M) and small (A, G, C, S) residues"
    )
    definitions['alternating_pattern_rigid_flexible_alternation'] = (
        "Score measuring alternation between rigid (W, Y, F) and flexible (R, K, M, S) residues"
    )
    definitions['alternating_pattern_big_fraction'] = (
        "Fraction of bulky residues (Phe, Trp, Tyr, Lys, Arg, Met) in the sequence"
    )
    definitions['alternating_pattern_small_fraction'] = (
        "Fraction of small residues (Ala, Gly, Cys, Ser) in the sequence"
    )
    definitions['alternating_pattern_rigid_fraction'] = (
        "Fraction of conformationally rigid residues (Trp, Tyr, Phe)"
    )
    definitions['alternating_pattern_flexible_fraction'] = (
        "Fraction of conformationally flexible residues (Arg, Lys, Met, Ser)"
    )

    # ==========================================================================
    # SECONDARY STRUCTURE FEATURES (from DSSP)
    # ==========================================================================
    definitions['helix_content_helix_fraction'] = (
        "Fraction of residues in α-helical conformation (predicted from 3D structure using DSSP)"
    )
    definitions['sheet_content_sheet_fraction'] = (
        "Fraction of residues in β-sheet conformation (extended strand structures)"
    )
    definitions['turn_content_turn_fraction'] = (
        "Fraction of residues in turn conformations (reverse direction of the peptide chain)"
    )
    definitions['coil_content_coil_fraction'] = (
        "Fraction of residues in random coil (unstructured/flexible regions)"
    )
    definitions['solvent_accessibility_avg_rsa'] = (
        "Average relative solvent accessibility (0-1); measures how exposed residues are to solvent"
    )
    definitions['solvent_accessibility_std_rsa'] = (
        "Standard deviation of solvent accessibility; indicates variation in exposure along the sequence"
    )

    # ==========================================================================
    # RESIDUE EXPOSURE FEATURES
    # ==========================================================================
    definitions['residue_exposure_buried_fraction'] = (
        "Fraction of residues that are buried (RSA < 0.2); indicates core/interior residues"
    )
    definitions['residue_exposure_exposed_fraction'] = (
        "Fraction of residues that are solvent-exposed (RSA > 0.5); surface-accessible residues"
    )

    # ==========================================================================
    # CONTACT ORDER FEATURES
    # ==========================================================================
    definitions['contact_order_0'] = (
        "Relative contact order: average sequence separation of contacting residues, "
        "normalized by sequence length; higher values indicate more complex folding topology"
    )

    # ==========================================================================
    # LONG-RANGE CONTACT FEATURES
    # ==========================================================================
    definitions['long_range_contact_lr_contact_density'] = (
        "Density of long-range contacts (>5 residues apart); indicates tertiary structure complexity"
    )
    definitions['long_range_contact_lr_contact_distribution'] = (
        "Coefficient of variation in long-range contact distribution; measures contact uniformity"
    )

    # ==========================================================================
    # INTERACTION DENSITY FEATURES
    # ==========================================================================
    definitions['interaction_density_local_density'] = (
        "Density of local contacts (<4 residues apart in sequence); indicates local structural compactness"
    )
    definitions['interaction_density_nonlocal_density'] = (
        "Density of non-local contacts (≥4 residues apart); indicates long-range structural interactions"
    )

    # ==========================================================================
    # CONTACT DENSITY FEATURES
    # ==========================================================================
    definitions['contact_density_short_range_density'] = (
        "Density of short-range atomic contacts (<5 residues apart in sequence)"
    )
    definitions['contact_density_long_range_density'] = (
        "Density of long-range atomic contacts (≥5 residues apart); key indicator of tertiary structure"
    )

    # ==========================================================================
    # GEOMETRIC FEATURES
    # ==========================================================================
    definitions['radius_gyration_rg_norm'] = (
        "Normalized radius of gyration: RMS distance of atoms from center of mass, "
        "indicates overall compactness (normalized by 30 Å)"
    )
    definitions['asphericity_asphericity'] = (
        "Asphericity parameter (0-1): deviation from spherical shape; 0 = perfect sphere, 1 = maximally elongated"
    )
    definitions['elongation_major_minor_ratio'] = (
        "Ratio of major to minor principal axes; values >1 indicate elongated shape"
    )
    definitions['elongation_planarity'] = (
        "Planarity: ratio of intermediate to major axis; indicates whether structure is planar or globular"
    )
    definitions['molecular_volume_volume_norm'] = (
        "Molecular volume (convex hull) normalized by 1000 Å³; overall 3D space occupied by the peptide"
    )

    # ==========================================================================
    # BACKBONE DIHEDRAL FEATURES
    # ==========================================================================
    definitions['backbone_dihedral_phi_psi_correlation'] = (
        "Correlation between φ and ψ backbone torsion angles; indicates regularity of backbone conformation"
    )
    definitions['backbone_dihedral_ramachandran_outliers'] = (
        "Fraction of residues with φ/ψ angles outside allowed Ramachandran regions (transformed); "
        "indicates structural strain or unusual conformations"
    )

    return definitions


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract feature column names, excluding metrics/targets and non-feature columns."""
    feature_cols = []

    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if any(pattern in col for pattern in METRIC_PATTERNS):
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            feature_cols.append(col)

    return feature_cols


def create_glossary(features: list, definitions: dict) -> pd.DataFrame:
    """Create a glossary DataFrame with feature names and definitions."""
    glossary_data = []

    for feature in features:
        if feature in definitions:
            definition = definitions[feature]
        else:
            # Generate a reasonable default definition from the feature name
            definition = generate_default_definition(feature)

        glossary_data.append({
            'Feature Name': feature,
            'Definition': definition
        })

    return pd.DataFrame(glossary_data)


def generate_default_definition(feature_name: str) -> str:
    """Generate a default definition for features not in the definitions dict."""
    # Parse the feature name
    parts = feature_name.split('_')

    # Try to interpret common patterns
    if 'fraction' in feature_name.lower():
        return f"Fraction/proportion measure derived from: {' '.join(parts)}"
    elif 'density' in feature_name.lower():
        return f"Density measure derived from: {' '.join(parts)}"
    elif 'mean' in feature_name.lower() or 'avg' in feature_name.lower():
        return f"Average value of: {' '.join(parts)}"
    elif 'std' in feature_name.lower():
        return f"Standard deviation (variation) of: {' '.join(parts)}"
    elif 'max' in feature_name.lower():
        return f"Maximum value of: {' '.join(parts)}"
    elif 'min' in feature_name.lower():
        return f"Minimum value of: {' '.join(parts)}"
    else:
        return f"Feature derived from: {' '.join(parts)} (see technical documentation for details)"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a glossary of peptide feature definitions."
    )
    parser.add_argument(
        "--data",
        default="data/processed_data/features_ml_with_targets.csv",
        help="Path to features CSV file"
    )
    parser.add_argument(
        "--output",
        default="peptide_feature_definitions.csv",
        help="Output glossary file path"
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        help="Also export as Excel file (.xlsx)"
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)

    # Get feature columns
    features = get_feature_columns(df)
    print(f"Found {len(features)} feature columns")

    # Get definitions
    definitions = get_feature_definitions()
    print(f"Have {len(definitions)} pre-defined definitions")

    # Create glossary
    glossary_df = create_glossary(features, definitions)

    # Check coverage
    defined_count = sum(1 for f in features if f in definitions)
    print(f"Definition coverage: {defined_count}/{len(features)} features ({100*defined_count/len(features):.1f}%)")

    # List any undefined features
    undefined = [f for f in features if f not in definitions]
    if undefined:
        print(f"\nFeatures without explicit definitions ({len(undefined)}):")
        for f in undefined:
            print(f"  - {f}")

    # Save CSV
    glossary_df.to_csv(args.output, index=False)
    print(f"\nSaved glossary to: {args.output}")

    # Optionally save as Excel
    if args.xlsx:
        xlsx_path = args.output.replace('.csv', '.xlsx')
        glossary_df.to_excel(xlsx_path, index=False, sheet_name='Feature Glossary')
        print(f"Saved Excel version to: {xlsx_path}")

    print(f"\nGlossary contains {len(glossary_df)} feature definitions")


if __name__ == "__main__":
    main()
