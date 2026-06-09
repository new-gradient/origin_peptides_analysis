# Peptide Features Calculation System

This directory contains the comprehensive feature calculation system for peptide sequence analysis and structure characterization. The system computes over 100 distinct features from peptide sequences and their predicted 3D structures, organized into two main categories: non-folding (sequence-based) and folding (structure-based) features.

## System Architecture

### Core Components

#### Base Framework (`base_feature.py`)
The foundation of the feature system, providing:
- **Abstract `BaseFeature` class**: Template for all feature implementations
- **Dual-mode calculation**: Raw features and ML-ready transformations
- **Bulk processing**: Efficient parallel computation for multiple sequences
- **Visualization**: Sophisticated plotting with distribution analysis

#### Main Pipeline (`peptide_features.py`)
Orchestrates the entire feature calculation workflow:
- **Parallel processing**: Uses `ProcessPoolExecutor` for multi-core computation
- **Feature organization**: Groups features into logical categories
- **Data persistence**: Save/load features as CSV files
- **Comprehensive visualization**: Distribution plots for all feature types

## Feature Categories

### Non-Folding Features (Sequence-Based)

These features are calculated directly from the amino acid sequence without requiring 3D structure information.

#### 1. Basic Features (`non_folding/basic_features.py`)

**Chain Length**
- Raw: Sequence length (number of residues)
- ML-ready: Length + log-transformed version
- Range: Typically 10-30 for peptides

**Molecular Weight**
- Raw: Total molecular weight in Daltons
- ML-ready: Total weight + weight per residue
- Algorithm: BioPython's molecular weight calculation

**Amino Acid Composition**
- Raw: Count of each of 20 standard amino acids
- ML-ready: Fraction of each amino acid (normalized by length)
- Output: 20 features for counts, 20 for fractions

**Dipeptide Composition**
- Raw: Counts of all 400 possible dipeptides
- ML-ready: Grouped by properties (hydrophobic, polar, charged, special)
- Output: 25 grouped dipeptide features

**Sequence Entropy**
- Raw/ML-ready: Shannon entropy of amino acid distribution
- Formula: -Σ(p_i * log2(p_i))
- Range: 0 (single AA) to 4.32 bits (uniform distribution)

#### 2. Chemical Features (`non_folding/chemical_features.py`)

**Isoelectric Point**
- Raw/ML-ready: pH at which net charge is zero
- Range: 0-14 pH units
- Algorithm: BioPython's pI calculation

**Charge Profile**
- Raw: Net charge at pH 2, 7, and 12
- ML-ready: Charge normalized per residue
- Uses Henderson-Hasselbalch equation with custom pKa values

**Local Charge Analysis**
- Raw: Sliding window charge profile (window size = 5)
- ML-ready features:
  - `max_local_charge`: Maximum local charge
  - `min_local_charge`: Minimum local charge
  - `charge_variation`: Standard deviation of local charges
  - `charge_asymmetry`: N-terminal vs C-terminal charge difference

**Hydrophobicity**
- Raw: Kyte-Doolittle and Eisenberg hydrophobicity profiles
- ML-ready:
  - Mean and standard deviation for both scales
  - Hydrophobic moment (amphipathicity measure)
- Algorithm: Complex exponential calculation for hydrophobic moment

**Amphipathicity**
- Raw: Local amphipathic profiles (window size = 11)
- ML-ready:
  - `max_amphipathicity`: Peak amphipathic value
  - `mean_amphipathicity`: Average amphipathicity
  - `amphipathicity_variation`: Standard deviation

**Predicted Solubility**
- Raw: GRAVY score + absolute charge at pH 7
- ML-ready: Direct values (lower GRAVY + higher charge = better solubility)

**Predicted Aggregation**
- Raw: GRAVY + charge + aromatic content
- ML-ready: Composite score with weights (GRAVY: +0.5, charge: -0.3, aromatic: +0.2)

#### 3. Pattern Features (`non_folding/pattern_features.py`)

**Residue Properties**
- Raw: Mass statistics, instability index
- ML-ready:
  - `average_mass`: Mean residue mass
  - `mass_std`: Mass standard deviation
  - `instability_index`: Protein instability prediction
  - `mass_per_charge`: Log-transformed mass/charge ratio

**Periodic Patterns**
- Raw: Counts of repeating 2-mer, 3-mer, and 4-mer patterns
- ML-ready:
  - Density measures with aggressive log transformations
  - `repeat_ratio_2_3`: Ratio of 2-mer to 3-mer densities
- Transformation: Power-law scaling for very small values

**Positional Features**
- Raw: Analysis of first/last 5 residues
- ML-ready:
  - `n_term_charged_frac`: N-terminal charge fraction
  - `c_term_charged_frac`: C-terminal charge fraction
  - `n_term_hydrophobic_frac`: N-terminal hydrophobic fraction
  - `c_term_hydrophobic_frac`: C-terminal hydrophobic fraction
  - `terminal_charge_bias`: Terminal charge asymmetry

**Specialized Residues**
- Groups analyzed:
  - Aromatic: F, W, Y
  - Oxidation-sensitive: C, M, W
  - Structure-breaking: G, P
  - Tiny: A, G, S
- ML-ready features per group:
  - Fraction of specialized residues
  - Clustering coefficient
  - Maximum local density

**Alternating Patterns**
- Raw: Alternation scores for property pairs
- Property pairs:
  - Big (FWYKRM) vs Small (AGCS)
  - Rigid (WYF) vs Flexible (RKMS)
- ML-ready: Direct alternation scores + group fractions

#### 4. Fragment Features (`non_folding/fragment_features.py`)

**Tryptic Terminal**
- Raw: Boolean for C-terminal K/R (not preceded by P)
- ML-ready: Float conversion (0.0 or 1.0)
- Application: Identifies trypsin-generated fragments

**Internal Cleavage Sites**
- Raw: Count of cleavable (K/R not followed by P) and resistant sites (KP/RP)
- ML-ready: Log-transformed density normalized by sequence length

**Hydrolysis Prone Bonds**
- Raw: Count of Asp-Pro dipeptides
- ML-ready: Log-transformed density
- Significance: DP bonds susceptible to acid hydrolysis

### Folding Features (Structure-Based)

These features require 3D structure information from protein folding predictions (AlphaFold/ColabFold).

#### 1. Secondary Structure Features (`folding/DSSP_features.py`)

**DSSP Implementation**
- Uses DSSP algorithm for secondary structure assignment
- Robust PDB handling with temporary file strategy
- Categories: Helix (H,G,I), Sheet (E), Turn (T), Coil (remainder)

**Helix Content**
- Raw/ML-ready: Fraction of residues in helical conformations
- Range: 0.0 to 1.0

**Sheet Content**
- Raw: Fraction in β-sheet
- ML-ready: 3-tier transformation for zero-inflated distribution:
  - Zero sheet (0) → [0.0-0.2]
  - Minimal sheet (0-10%) → [0.4-0.6]
  - Significant sheet (>10%) → [0.8-1.0]

**Turn and Coil Content**
- Raw/ML-ready: Fraction in turns and unstructured regions
- Coil = 1 - (helix + sheet + turn)

**Solvent Accessibility**
- Raw: DSSP relative solvent accessibility (RSA) values
- ML-ready:
  - `avg_rsa`: Mean RSA
  - `std_rsa`: RSA standard deviation

**Residue Exposure**
- Raw: Buried (<0.2 RSA) and exposed (>0.5 RSA) residue counts
- ML-ready: Sophisticated transformations with weighted scoring

#### 2. Contact Map Features (`folding/contact_map_features.py`)

**Contact Order**
- Raw: Average sequence separation of contacting residues
- ML-ready: Normalized by contacts and sequence length
- Formula: Σ(|i-j|) / (n_contacts × n_residues)

**Long Range Contacts**
- Raw: Statistics for contacts >5 residues apart
- ML-ready:
  - `lr_contact_density`: Scaled density (200× factor)
  - `lr_contact_distribution`: Coefficient of variation

**Contact Clusters**
- Raw: Connected components in contact map
- ML-ready: Multiple threshold attempts (8.0, 7.0, 6.0, 5.0 Å)
  - `cluster_density`: Normalized cluster count
  - `avg_cluster_size_ratio`: Size distribution measure
  - `clusters_per_residue`: Log-scaled density

**Interaction Density**
- Raw: Local (<4 residues) and non-local contact counts
- ML-ready: Density normalized by maximum possible contacts

#### 3. Geometry Features (`folding/geometry_features.py`)

**Radius of Gyration**
- Raw: RMS distance from center of mass
- ML-ready: Normalized by 30.0 Å
- Physical meaning: Overall compactness

**Asphericity**
- Raw: Shape deviation from sphere using gyration tensor
- ML-ready: Direct use (already 0-1 scaled)
- Values: 0 = perfect sphere, 1 = maximally aspherical

**Contact Density**
- Raw: Short-range (<5) and long-range (≥5) residue contacts
- ML-ready:
  - Short-range: Direct density
  - Long-range: Aggressive log transformation (50× scale)

**Elongation**
- Raw: Principal component analysis of coordinates
- ML-ready:
  - `major_minor_ratio`: Shape elongation (≥1.0)
  - `planarity`: Intermediate/major axis ratio (0-1)

**Molecular Volume**
- Raw: Convex hull volume calculation
- ML-ready: Normalized by 1000 ų
- Fallback: Bounding box volume if convex hull fails

#### 4. Dynamics Features (`folding/dynamics_features.py`)

**Backbone Dihedrals**
- Raw: φ/ψ torsion angles
- ML-ready:
  - `phi_psi_correlation`: Angle correlation (-1 to 1)
  - `ramachandran_outliers`: Transformed outlier fraction
- Validation: 4 allowed Ramachandran regions

**Confidence Scores**
- Raw: pLDDT values from AlphaFold
- ML-ready:
  - `mean_plddt`: Normalized confidence (0-1)
  - `plddt_variance`: Confidence variation

**Predicted Aligned Error (PAE)**
- Raw: PAE matrix statistics
- ML-ready: Normalized PAE values
- Application: Domain organization and flexibility

## Usage

### Command Line Interface

**Calculate all features with plots:**
```bash
python -m peptide_ml.features.peptide_features \
  --csv /path/to/data.csv \
  --pdb-dir /path/to/pdb/structures \
  --output-dir feature_plots \
  --save
```

**Load pre-calculated features:**
```bash
python -m peptide_ml.features.peptide_features \
  --csv /path/to/data.csv \
  --pdb-dir /path/to/pdb/structures \
  --output-dir feature_plots \
  --load
```

**Recalculate only ML features:**
```bash
python -m peptide_ml.features.peptide_features \
  --csv /path/to/data.csv \
  --pdb-dir /path/to/pdb/structures \
  --output-dir feature_plots \
  --calc-ml \
  --raw-csv /path/to/features_raw.csv \
  --ml-csv /path/to/features_ml.csv \
  --save
```

### Python API

```python
from peptide_ml.features.peptide_features import PeptideFeatures

# Initialize feature calculator
pf = PeptideFeatures(
    csv_path='data.csv',
    pdb_dir='pdb_structures/',
    output_dir='feature_plots/'
)

# Calculate all features
pf.calculate_all_features(parallel=True)

# Save features
pf.save_features('output_dir/')

# Access feature data
raw_features = pf.raw_data  # DataFrame with raw features
ml_features = pf.ml_data    # DataFrame with ML-ready features

# Plot distributions
pf.plot_all_features()
```

## Feature Transformations

The system employs sophisticated transformations to prepare features for machine learning:

### Transformation Strategies

1. **Log Transformations**: For right-skewed distributions
   - Formula: `log(1 + scale_factor × value)`
   - Scale factors: 50-300× for very small values

2. **Three-Tier Transformations**: For zero-inflated features
   - Zero values → Low range [0.0-0.2]
   - Small values → Mid range [0.4-0.6]
   - Large values → High range [0.8-1.0]

3. **Min-Max Scaling**: For bounded features
   - Formula: `(value - min) / (max - min)`

4. **Power Transformations**: For extreme value spreading
   - Square root for moderate spreading
   - Cubic root for aggressive spreading

5. **Sigmoid Centering**: For balanced distributions
   - Centers transformed values around 0.5

## Output Files

### Feature Data Files
- `features_raw.csv`: Raw calculated features
- `features_ml.csv`: ML-ready transformed features

### Visualization Outputs
- `feature_plots/basic/`: Basic feature distributions
- `feature_plots/chemical/`: Chemical property distributions
- `feature_plots/composition/`: Composition feature distributions
- `feature_plots/pattern/`: Pattern feature distributions
- `feature_plots/secondary_structure/`: Structure feature distributions
- `feature_plots/contact_map/`: Contact-based distributions
- `feature_plots/geometry/`: Geometric feature distributions
- `feature_plots/dynamics/`: Dynamics feature distributions
- `feature_plots/feature_correlations.png`: Feature correlation heatmap

## Dependencies

### Required Python Packages
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **BioPython**: Sequence analysis and structure processing
- **scipy**: Scientific computing (convex hull, stats)
- **matplotlib/seaborn**: Visualization
- **DSSP**: Secondary structure assignment (external binary)

### Required Input Data
1. **CSV file** with columns:
   - `PeptideID`: Unique identifier
   - `Sequence`: Amino acid sequence

2. **PDB structures** organized as:
   - Directory structure: `pdb_dir/peptide_{ID}/`
   - PDB file naming: `peptide_{ID}_relaxed_rank_*.pdb`
   - JSON files: AlphaFold confidence scores (optional)

## Performance Considerations

- **Parallel Processing**: Default uses all available CPU cores
- **Memory Usage**: ~2-4 GB for 1000 peptides
- **Computation Time**: ~1-2 minutes per 100 peptides (with structures)
- **Caching**: Pre-calculated features can be loaded to skip computation

## Troubleshooting

### Common Issues

1. **DSSP not found**: Install DSSP binary and ensure it's in PATH
2. **Missing PDB files**: Check directory structure and file naming
3. **Memory errors**: Reduce parallel workers or process in batches
4. **Invalid sequences**: Check for non-standard amino acids

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Feature Selection Guidelines

For machine learning applications:

1. **High-importance features** (typically selected):
   - Hydrophobicity measures
   - Charge distribution
   - Secondary structure content
   - Contact order
   - Sequence entropy

2. **Redundant features** (consider removing):
   - Highly correlated features (>0.95 correlation)
   - Single-value features for your dataset

3. **Domain-specific selection**:
   - Aggregation studies: Focus on hydrophobicity, aromatic content
   - Stability studies: Emphasize secondary structure, contact features
   - Interaction studies: Prioritize charge, amphipathicity features


