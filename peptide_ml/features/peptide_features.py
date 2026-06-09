from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Callable
import pandas as pd
import numpy as np
import glob
import argparse
import ast
import concurrent.futures
from tqdm import tqdm  # Add tqdm for progress bars

# Import your existing modules
from peptide_ml._tools.data_processor import ExperimentDataProcessor
from peptide_ml.features.non_folding import (
    ChainLengthFeature,
    MolecularWeightFeature,
    AACompositionFeature,
    DipeptideCompositionFeature,
    SequenceEntropyFeature,
    IsoelectricPointFeature,
    ChargeFeature,
    LocalChargeFeature,
    HydrophobicityFeature,
    AmphipathicityFeature,
    PeriodicPatternsFeature,
    ResiduePropertyFeature,
    PositionalFeature,
    SpecializedResidueFeature,
    AlternatingPatternFeature,
    TrypticTerminalFeature,
    InternalCleavageSitesFeature,
    HydrolysisProneBondFeature,
    PredictedSolubilityFeature,     
    PredictedAggregationFeature,  
)
from peptide_ml.features.folding import (
    ContactClusterFeature,
    ContactOrderFeature,
    LongRangeContactFeature,
    InteractionDensityFeature,
    HelixContentFeature,
    SheetContentFeature,
    TurnContentFeature,
    CoilContentFeature,
    SolventAccessibilityFeature,
    ResidueExposureFeature,
    BackboneDihedralFeature,
    RadiusOfGyrationFeature,
    AsphericityFeature,
    ContactDensityFeature,
    ElongationFeature,
    MolecularVolumeFeature,
)

# Import the default base feature implementation.
from peptide_ml.features.base_feature import BaseFeature

# Define valid amino acids
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def clean_sequence(seq: str) -> str:
    """Clean protein sequence to only include valid amino acids."""
    if not isinstance(seq, str):
        return ''
    return ''.join(c for c in seq.upper() if c in VALID_AA)

def convert_if_list(val):
    """
    If the value is a string representing a list (e.g. "[1.8, 1.8, ...]"),
    convert it to a Python list using ast.literal_eval.
    Otherwise, return the value unchanged.
    """
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val

class PeptideFeatures:
    """
    Class to calculate, load, save, and plot peptide features.
    
    Calculation:
      - Computes raw and ML-ready features for each peptide.
    CSV I/O:
      - By default, uses the data/ folder:
          • Input data CSV: data/data.csv
          • Peptide FASTA files in data/peptides
          • Raw features saved to data/peptides/features_raw.csv
          • ML-ready features saved to data/peptides/features_ml.csv
      - Provides options to load pre-calculated features.
    Plotting:
      - Methods to plot individual features, feature groups, and a correlation heatmap.
    Additional:
      - A method calc_ml() that loads the raw features and recalculates the ML-ready features.
        (This assumes each feature object implements calculate_ml_from_raw.)
    """
    
    def __init__(self, csv_path: str, pdb_dir: str, fasta_dir: str = "data/peptides", 
                 max_workers: int = None):
        """
        Initialize the PeptideFeatures class.
        
        Args:
            csv_path: Path to the CSV file with peptide data
            pdb_dir: Directory containing PDB files
            fasta_dir: Directory for FASTA files
            max_workers: Maximum number of worker processes/threads for parallel execution
                         (None = use CPU count)
        """
        self.csv_path = Path(csv_path)
        self.pdb_dir = Path(pdb_dir)
        self.fasta_dir = Path(fasta_dir)
        self.max_workers = max_workers
        
        # Initialize the experimental data processor
        self.processor = ExperimentDataProcessor(str(self.csv_path), low_memory=False)
        
        # Clean peptide sequences
        for peptide in self.processor.peptides:
            peptide.sequence = clean_sequence(peptide.sequence)
        
        # Ensure FASTA and PDB mapping exists
        self.fasta_dir.mkdir(parents=True, exist_ok=True)
        self.fasta_mapping = self.processor.write_peptides_to_fasta(str(self.fasta_dir))
        self.pbd_mapping = self.processor.create_pdb_mapping(str(self.pdb_dir))
        
        # Remove invalid peptides
        self.processor.peptides = self.processor.clean_peptides()
        
        # Initialize feature objects
        self.features = {
            # Basic features
            'chain_length': ChainLengthFeature(),
            'molecular_weight': MolecularWeightFeature(),
            'aa_composition': AACompositionFeature(),
            'dipeptide': DipeptideCompositionFeature(),
            'entropy': SequenceEntropyFeature(),
            
            # Chemical features
            'isoelectric': IsoelectricPointFeature(),
            'charge': ChargeFeature(),
            'local_charge': LocalChargeFeature(),
            'hydrophobicity': HydrophobicityFeature(),
            'amphipathicity': AmphipathicityFeature(),
            'predicted_solubility': PredictedSolubilityFeature(), 
            'predicted_aggregation': PredictedAggregationFeature(), 
            
            # Pattern features
            'periodic_patterns': PeriodicPatternsFeature(),
            'residue_properties': ResiduePropertyFeature(),
            'positional': PositionalFeature(),
            'specialized_residue': SpecializedResidueFeature(),
            'alternating_pattern': AlternatingPatternFeature(),

            # Fragment features
            'tryptic_terminal': TrypticTerminalFeature(),
            'internal_cleavage': InternalCleavageSitesFeature(),
            'hydrolysis_prone': HydrolysisProneBondFeature(),

            # Secondary Structure features
            'helix_content': HelixContentFeature(),
            'sheet_content': SheetContentFeature(),
            'turn_content': TurnContentFeature(),
            'coil_content': CoilContentFeature(),
            'solvent_accessibility': SolventAccessibilityFeature(),
            'residue_exposure': ResidueExposureFeature(),
            
            # Contact Map features
            'contact_cluster': ContactClusterFeature(),
            'contact_order': ContactOrderFeature(),
            'long_range_contact': LongRangeContactFeature(),
            'interaction_density': InteractionDensityFeature(),
            'contact_density': ContactDensityFeature(),
            
            # Geometry features
            'radius_gyration': RadiusOfGyrationFeature(),
            'asphericity': AsphericityFeature(),
            'elongation': ElongationFeature(),
            'molecular_volume': MolecularVolumeFeature(),
            
            # Dynamics features
            'backbone_dihedral': BackboneDihedralFeature(),
        }
        
        self.feature_groups = {
            'basic': ['chain_length', 'molecular_weight', 'entropy'],
            'composition': ['aa_composition', 'dipeptide'],
            'chemical': ['isoelectric', 'charge', 'local_charge', 
                         'hydrophobicity', 'amphipathicity',
                         'predicted_solubility',    
                         'predicted_aggregation'],
            'pattern': ['periodic_patterns', 'residue_properties', 'positional',
                        'specialized_residue', 'alternating_pattern'],
            'secondary_structure': [
                'helix_content', 'sheet_content', 'turn_content', 
                'coil_content', 'solvent_accessibility', 'residue_exposure'
            ],
            'contact_map': [
                'contact_cluster', 'contact_order', 'long_range_contact',
                'interaction_density', 'contact_density'
            ],
            'geometry': [
                'radius_gyration', 'asphericity', 'elongation', 
                'molecular_volume'
            ],
            'dynamics': [
                'backbone_dihedral'
            ],
            'fragment_props': ['tryptic_terminal', 'internal_cleavage', 'hydrolysis_prone']
        }
        
        # Dictionaries to store calculated features (each key holds a DataFrame)
        self.raw_features = {}
        self.ml_features = {}
        
        # Plot style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.raw_color = '#2196F3'
        self.ml_color = '#F44336'

    def _calculate_feature_for_peptide(self, feature_name, feature, peptide, is_pdb_based):
        """Helper function to calculate feature for a single peptide. Used by parallel executor."""
        try:
            if is_pdb_based:
                raw_result = feature.calculate_raw(peptide.pdb_path)
                ml_result = feature.calculate_ml_ready(peptide.pdb_path)
            else:
                raw_result = feature.calculate_raw(peptide.sequence)
                ml_result = feature.calculate_ml_ready(peptide.sequence)
            return {'raw': raw_result, 'ml': ml_result, 'success': True}
        except Exception as e:
            print(f"Error calculating {feature_name} for peptide {peptide.sequence[:10]}...: {e}")
            return {'raw': {}, 'ml': {}, 'success': False}

    def _calculate_feature_for_all_peptides(self, feature_info):
        """
        Helper function to calculate a single feature for all peptides.
        
        Args:
            feature_info: Tuple of (feature_name, feature_class)
        
        Returns:
            Dictionary with feature name, raw results, and ml results
        """
        name, feature_class = feature_info
        
        try:
            print(f"Calculating {name} features...")
            
            # Process all peptides for this feature
            raw_results = []
            ml_results = []
            
            for peptide in tqdm(self.processor.peptides, desc=f"Processing {name}"):
                try:
                    # Create a FRESH instance of the feature class for each peptide
                    # This is CRITICAL to prevent state being shared between peptides
                    feature = feature_class.__class__()
                    
                    # Determine if the feature requires PDB files
                    is_pdb_based = isinstance(feature, (HelixContentFeature, SheetContentFeature, 
                                        TurnContentFeature, CoilContentFeature, 
                                        SolventAccessibilityFeature, ResidueExposureFeature,
                                        ContactClusterFeature, ContactOrderFeature,
                                        LongRangeContactFeature, InteractionDensityFeature,
                                        ContactDensityFeature, RadiusOfGyrationFeature,
                                        AsphericityFeature, ElongationFeature,
                                        MolecularVolumeFeature, BackboneDihedralFeature))
                    
                    if is_pdb_based:
                        raw_result = feature.calculate_raw(peptide.pdb_path)
                        ml_result = feature.calculate_ml_ready(peptide.pdb_path)
                    else:
                        raw_result = feature.calculate_raw(peptide.sequence)
                        ml_result = feature.calculate_ml_ready(peptide.sequence)
                    
                    raw_results.append(raw_result)
                    ml_results.append(ml_result)
                except Exception as e:
                    print(f"Error processing peptide {peptide.sequence[:10]}... for {name}: {e}")
                    raw_results.append({})
                    ml_results.append({})
            
            return {
                'name': name,
                'raw_results': pd.DataFrame(raw_results),
                'ml_results': pd.DataFrame(ml_results)
            }
        except Exception as e:
            print(f"Error processing feature {name}: {e}")
            return {
                'name': name,
                'raw_results': pd.DataFrame(),
                'ml_results': pd.DataFrame()
            }

    def calculate_all_features(self):
        """Calculate raw and ML-ready features for all peptides using parallel processing by feature."""
        # Get a list of all features for parallel processing
        feature_tasks = list(self.features.items())
        
        # Execute feature calculations in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all features for processing
            futures = [executor.submit(self._calculate_feature_for_all_peptides, task) for task in feature_tasks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    feature_name = result['name']
                    self.raw_features[feature_name] = result['raw_results']
                    self.ml_features[feature_name] = result['ml_results']
                    print(f"Completed processing for {feature_name}")
                except Exception as e:
                    print(f"A feature calculation failed with error: {e}")
        
        print("Completed all feature calculations")

    def _prefix_columns(self, features_dict: dict) -> pd.DataFrame:
        """
        Combine the feature DataFrames into one DataFrame with columns prefixed by feature name.
        Also, prepend a DataFrame with "ID" and "sequence" columns (from the processor's peptides).
        """
        prefixed = []
        for feature_name, df in features_dict.items():
            if not df.empty:
                df_prefixed = df.copy()
                df_prefixed.columns = [f"{feature_name}_{col}" for col in df.columns]
                prefixed.append(df_prefixed)
        if prefixed:
            combined_features = pd.concat(prefixed, axis=1)
        else:
            combined_features = pd.DataFrame()
        # Create ID and sequence columns
        id_df = pd.DataFrame({
            "ID": [i + 1 for i in range(len(self.processor.peptides))],
            "sequence": [peptide.sequence for peptide in self.processor.peptides]
        })
        return pd.concat([id_df, combined_features], axis=1)

    def save_raw_features(self, output_csv: Path):
        """
        Save the combined raw features for all peptides to a CSV file.
        (Default location: data/peptides/features_raw.csv)
        """
        print(f"Saving raw features to {output_csv} ...")
        try:
            combined = self._prefix_columns(self.raw_features)
            combined.to_csv(output_csv, index=False)
            print("Raw features saved successfully.")
        except Exception as e:
            print(f"Error saving raw features: {e}")

    def save_ml_features(self, output_csv: Path):
        """
        Save the combined ML-ready features for all peptides to a CSV file.
        (Default location: data/peptides/features_ml.csv)
        """
        print(f"Saving ML-ready features to {output_csv} ...")
        try:
            combined = self._prefix_columns(self.ml_features)
            combined.to_csv(output_csv, index=False)
            print("ML-ready features saved successfully.")
        except Exception as e:
            print(f"Error saving ML-ready features: {e}")

    def load_raw_features(self, input_csv: Path):
        """
        Load combined raw features from an existing CSV file and reassemble them into a dictionary.
        After loading, remove the feature-specific prefix from each column name and convert
        any string representations of lists back into actual lists.
        """
        print(f"Loading raw features from {input_csv} ...")
        try:
            combined = pd.read_csv(input_csv)
            raw_dict = {}
            for feature_name in self.features:
                cols = [col for col in combined.columns if col.startswith(f"{feature_name}_")]
                if cols:
                    df_feature = combined[cols].copy()
                    new_columns = {col: col[len(feature_name) + 1:] for col in df_feature.columns}
                    df_feature.rename(columns=new_columns, inplace=True)
                    for col in df_feature.columns:
                        df_feature[col] = df_feature[col].apply(convert_if_list)
                    raw_dict[feature_name] = df_feature
                else:
                    raw_dict[feature_name] = pd.DataFrame()
            self.raw_features = raw_dict
            print("Raw features loaded successfully.")
        except Exception as e:
            print(f"Error loading raw features: {e}")

    def load_ml_features(self, input_csv: Path):
        """
        Load combined ML-ready features from an existing CSV file and reassemble them into a dictionary.
        After loading, remove the feature-specific prefix from each column name and convert
        any string representations of lists back into actual lists.
        """
        print(f"Loading ML-ready features from {input_csv} ...")
        try:
            combined = pd.read_csv(input_csv)
            ml_dict = {}
            for feature_name in self.features:
                cols = [col for col in combined.columns if col.startswith(f"{feature_name}_")]
                if cols:
                    df_feature = combined[cols].copy()
                    new_columns = {col: col[len(feature_name) + 1:] for col in df_feature.columns}
                    df_feature.rename(columns=new_columns, inplace=True)
                    for col in df_feature.columns:
                        df_feature[col] = df_feature[col].apply(convert_if_list)
                    ml_dict[feature_name] = df_feature
                else:
                    ml_dict[feature_name] = pd.DataFrame()
            self.ml_features = ml_dict
            print("ML-ready features loaded successfully.")
        except Exception as e:
            print(f"Error loading ML-ready features: {e}")

    def _process_feature_ml_for_all_peptides(self, args):
        """Helper function for parallel ML feature recalculation by feature."""
        feature_name, feature_class_instance, raw_data = args
        
        try:
            print(f"Recalculating ML features for {feature_name}...")
            ml_results = []
            
            for idx, raw_row in tqdm(raw_data.iterrows(), 
                                     total=len(raw_data), 
                                     desc=f"Processing {feature_name}"):
                try:
                    # Create a FRESH instance of the feature class for each peptide
                    # This is CRITICAL to prevent state being shared between peptides
                    feature_obj = feature_class_instance.__class__()
                    
                    # Load raw data into feature object
                    if hasattr(feature_obj, 'load_raw_data'):
                        feature_obj.load_raw_data(raw_row)
                    else:
                        feature_obj.raw_data = raw_row.to_dict()
                    
                    # Get corresponding peptide
                    if idx < len(self.processor.peptides):
                        peptide = self.processor.peptides[idx]
                        
                        # Calculate ML features based on feature type
                        is_pdb_based = isinstance(feature_obj, (HelixContentFeature, SheetContentFeature, 
                                             TurnContentFeature, CoilContentFeature, 
                                             SolventAccessibilityFeature, ResidueExposureFeature,
                                             ContactClusterFeature, ContactOrderFeature,
                                             LongRangeContactFeature, InteractionDensityFeature,
                                             ContactDensityFeature, RadiusOfGyrationFeature,
                                             AsphericityFeature, ElongationFeature,
                                             MolecularVolumeFeature, BackboneDihedralFeature))
                        
                        if is_pdb_based:
                            ml_data = feature_obj.calculate_ml_ready(peptide.pdb_path)
                        else:
                            ml_data = feature_obj.calculate_ml_ready(peptide.sequence)
                        
                        ml_results.append(ml_data)
                    else:
                        print(f"Warning: Missing peptide data for index {idx}")
                        ml_results.append({})
                except Exception as e:
                    print(f"  Error processing peptide at index {idx} for feature {feature_name}: {e}")
                    ml_results.append({})
            
            return {
                'name': feature_name, 
                'ml_results': pd.DataFrame(ml_results)
            }
        except Exception as e:
            print(f"Error processing feature {feature_name}: {e}")
            return {
                'name': feature_name, 
                'ml_results': pd.DataFrame()
            }

    def calc_ml(self, raw_csv: Path):
        """Recalculate ML features from raw features using parallel processing by feature."""
        print(f"Loading raw features from {raw_csv} for ML recalculation...")
        try:
            if not self.raw_features:
                self.load_raw_features(raw_csv)
        except Exception as e:
            print(f"Error loading raw features: {e}")
            return
            
        # Dictionary to store recalculated ML features
        new_ml_features = {}
        
        # Prepare tasks for parallel processing - one task per feature
        tasks = []
        for feature_name, feature_class_instance in self.features.items():
            raw_data = self.raw_features.get(feature_name)
            if raw_data is None or raw_data.empty:
                print(f"No raw data found for {feature_name}. Skipping recalculation.")
                new_ml_features[feature_name] = pd.DataFrame()
            else:
                tasks.append((feature_name, feature_class_instance, raw_data))
        
        # Process features in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks and get future objects
            futures = [executor.submit(self._process_feature_ml_for_all_peptides, task) for task in tasks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    feature_name = result['name']
                    new_ml_features[feature_name] = result['ml_results']
                    print(f"Completed ML recalculation for {feature_name}")
                except Exception as e:
                    print(f"A feature recalculation failed with error: {e}")
            
        # Update ML features
        self.ml_features = new_ml_features
        print("Recalculation of ML-ready features complete.")

    def recalculate_specific_features(self, feature_names):
        """
        Recalculate and plot only specific features using parallel processing.
        
        Args:
            feature_names: List of feature names to recalculate
        """
        print(f"Recalculating specific features: {', '.join(feature_names)}")
        
        # Prepare tasks for parallel processing - one task per feature
        tasks = []
        for name in feature_names:
            if name not in self.features:
                print(f"Warning: Feature '{name}' not found. Skipping.")
                continue
                
            feature_obj = self.features[name]
            raw_data = self.raw_features.get(name)
            
            if raw_data is None or raw_data.empty:
                print(f"No raw data for {name}. Skipping.")
                continue
                
            tasks.append((name, feature_obj, raw_data))
        
        # Process features in parallel 
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
                # Submit tasks and get future objects
                futures = [executor.submit(self._process_feature_ml_for_all_peptides, task) for task in tasks]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        feature_name = result['name']
                        self.ml_features[feature_name] = result['ml_results']
                        print(f"Completed ML recalculation for {feature_name}")
                    except Exception as e:
                        print(f"A feature recalculation failed with error: {e}")
        else:
            print("No features to recalculate.")

    def _plot_feature_task(self, args):
        """Helper function for parallel plotting."""
        feature_name, raw_data, ml_data, feature_obj, save_path = args
        try:
            if hasattr(feature_obj, "plot_distributions"):
                fig = feature_obj.plot_distributions(raw_data, ml_data)
            else:
                fig = BaseFeature.plot_distributions(feature_obj, raw_data, ml_data)
            
            plt.suptitle(f'{feature_name} Distributions', y=1.02, size=16)
            
            if save_path:
                save_file = save_path / f"{feature_name}_distributions.png"
                fig.savefig(save_file, bbox_inches='tight', dpi=300)
                plt.close(fig)
                return f"Saved plot to {save_file}"
            else:
                return fig
        except Exception as e:
            return f"Error plotting {feature_name}: {e}"

    def plot_feature(self, feature_name: str, save_path: Optional[Path] = None) -> plt.Figure:
        """Create and optionally save a distribution plot for a specific feature."""
        print(f"Plotting {feature_name}...")
        raw_data = self.raw_features.get(feature_name)
        ml_data = self.ml_features.get(feature_name)
        if raw_data is None or ml_data is None:
            raise ValueError(f"Feature {feature_name} has not been calculated.")
        feature_obj = self.features[feature_name]
        
        # For a single feature, just do it directly (no parallelism needed)
        return self._plot_feature_task((feature_name, raw_data, ml_data, feature_obj, save_path))

    def plot_feature_group(self, group_name: str, save_path: Optional[Path] = None):
        """Plot all features in a specific group using parallel processing."""
        if group_name not in self.feature_groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        
        print(f"\nPlotting feature group: {group_name}")
        feature_names = self.feature_groups[group_name]
        
        # Prepare tasks for parallel processing
        tasks = []
        for feature_name in feature_names:
            raw_data = self.raw_features.get(feature_name)
            ml_data = self.ml_features.get(feature_name)
            if raw_data is not None and ml_data is not None:
                feature_obj = self.features[feature_name]
                tasks.append((feature_name, raw_data, ml_data, feature_obj, save_path))
        
        # Process plots in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for result in executor.map(self._plot_feature_task, tasks):
                if isinstance(result, str):
                    print(result)  # Print any messages (like errors or save confirmations)

    def plot_all_features(self, save_dir: Optional[Path] = None):
        """Create and display (or save) plots for all feature groups using parallel processing."""
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            for group in self.feature_groups:
                (save_dir / group).mkdir(exist_ok=True)
                
        # Could parallelize at the group level, but let's use the group method which is already parallel
        for group_name, features in self.feature_groups.items():
            group_save_path = save_dir / group_name if save_dir else None
            self.plot_feature_group(group_name, group_save_path)

    def plot_feature_correlations(self, save_path: Optional[Path] = None):
        """Plot a correlation heatmap for all ML-ready features."""
        print("Calculating feature correlations...")
        try:
            all_features = pd.concat([df for df in self.ml_features.values()], axis=1)
            corr = all_features.corr()
            plt.figure(figsize=(96, 64))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
            plt.title('Feature Correlations')
            if save_path:
                heatmap_file = save_path / "feature_correlations.png"
                plt.savefig(heatmap_file, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Saved correlation plot to {heatmap_file}")
            else:
                plt.show()
        except Exception as e:
            print(f"Error generating correlation plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate, (re)load, save, and plot peptide features."
    )
    parser.add_argument("--csv", default="data/data.csv", help="Path to peptide CSV")
    parser.add_argument("--pdb-dir", default="data/peptides/pdb", 
                        help="Directory containing PDB files")
    parser.add_argument("--output-dir", default="feature_plots", 
                        help="Directory to save plots")
    parser.add_argument("--calc", action="store_true", 
                        help="Calculate all raw and ML features")
    parser.add_argument("--load", action="store_true", 
                        help="Load existing features")
    parser.add_argument("--calc-ml", action="store_true", 
                        help="Load raw features and recalculate ML features only")
    parser.add_argument("--save", action="store_true", 
                        help="Save features to CSV")
    parser.add_argument("--features", nargs='+', 
                        help="Specific features to recalculate (with --calc-ml)")
    parser.add_argument("--raw-csv", default="data/peptides/features_raw.csv",
                        help="Path to raw features CSV")
    parser.add_argument("--ml-csv", default="data/peptides/features_ml.csv",
                        help="Path to ML-ready features CSV")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes for parallel execution")
    args = parser.parse_args()
    
    # Initialize PeptideFeatures with parallel processing
    pf = PeptideFeatures(
        csv_path=args.csv,
        pdb_dir=args.pdb_dir,
        max_workers=args.num_workers
    )
    
    # Get paths for feature CSVs
    raw_csv_path = Path(args.raw_csv)
    ml_csv_path = Path(args.ml_csv)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each feature group
    for group in pf.feature_groups:
        (output_dir / group).mkdir(exist_ok=True)
    
    # Calculate features or load them from CSV
    if args.calc:
        # Calculate all features from scratch
        print("Calculating all features...")
        pf.calculate_all_features()
        if args.save:
            # Save both raw and ML features
            pf.save_raw_features(raw_csv_path)
            pf.save_ml_features(ml_csv_path)
    
    elif args.load:
        # Load features from existing CSV files
        print("Loading features...")
        pf.load_raw_features(raw_csv_path)
        pf.load_ml_features(ml_csv_path)
    
    elif args.calc_ml:
        # IMPORTANT: Never overwrite raw features with --calc-ml
        print("Loading raw features for ML recalculation...")
        pf.load_raw_features(raw_csv_path)
        pf.load_ml_features(ml_csv_path)
        
        if args.features:
            # Recalculate only specified features
            print(f"Recalculating ML features for: {', '.join(args.features)}")
            pf.recalculate_specific_features(args.features)
            
            if args.save:
                # Save ONLY the ML features
                pf.save_ml_features(ml_csv_path)
                
            # Plot only those specific features
            print("Plotting only recalculated features...")
            for feature in args.features:
                group = next((g for g, features in pf.feature_groups.items() 
                             if feature in features), None)
                if group:
                    group_path = output_dir / group
                    group_path.mkdir(exist_ok=True)
                    try:
                        pf.plot_feature(feature, group_path)
                    except Exception as e:
                        print(f"Error plotting {feature}: {e}")
                else:
                    print(f"Warning: Feature group not found for {feature}")
                    
            # Skip the full plotting below
            print("Processing complete!")
            import sys
            sys.exit(0)
        else:
            # Recalculate all ML features
            print("Recalculating all ML features...")
            pf.calc_ml(raw_csv_path)
            if args.save:
                # Save ONLY the ML features
                pf.save_ml_features(ml_csv_path)
    
    # If no calculation or loading requested, calculate all features
    if not (args.calc or args.load or args.calc_ml):
        print("No action specified, calculating all features...")
        pf.calculate_all_features()
        if args.save:
            pf.save_raw_features(raw_csv_path)
            pf.save_ml_features(ml_csv_path)
    
    # Plot all features (only if we didn't already plot specific features)
    print("Plotting all features...")
    pf.plot_all_features(save_dir=output_dir)
    
    print("Plotting feature correlation heatmap...")
    pf.plot_feature_correlations(save_path=output_dir)
    
    print("Processing complete!")