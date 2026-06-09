from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from ..base_feature import BaseFeature

class ResiduePropertyFeature(BaseFeature):
    """
    Calculate various residue properties with improved transformations.
    
    This feature analyzes amino acid properties like mass, charge distribution,
    and stability to characterize peptide composition.
    
    ML-ready features:
    - average_mass: Mean residue mass in Daltons
    - mass_std: Standard deviation of residue masses
    - instability_index: Measure of protein stability (>40 suggests instability)
    - mass_per_charge_ratio: Transformed ratio of total mass to charged residues
    """

    def __init__(self):
        super().__init__()
        # Residue masses (Da)
        self.residue_masses = {
            'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        }
        self.raw_columns = ['residue_masses', 'average_mass', 'instability_index']
        self.ml_columns = ['average_mass', 'mass_std', 'instability_index', 'mass_per_charge_ratio']
        
        # Transformation parameters for mass_per_charge
        self.mass_per_charge_params = {
            'method': 'log',              # Log transformation for extreme skew
            'default_value': 0.5,         # Default value when no charges (between 0-1)
            'scale_factor': 0.01,         # Scaling factor to bring values to reasonable range
            'min_val': 0.0,               # Will be updated during fitting
            'max_val': 1.0                # Will be updated during fitting
        }

    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """
        Calculate raw residue properties.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with raw residue properties
        """
        masses = [self.residue_masses[aa] for aa in sequence if aa in self.residue_masses]
        
        try:
            protein = ProteinAnalysis(sequence)
            instability_idx = protein.instability_index()
        except:
            # Fallback if ProteinAnalysis fails
            instability_idx = 0
        
        # Count charged residues
        positive_charged = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative_charged = sequence.count('D') + sequence.count('E')
        charged_count = positive_charged + negative_charged
        
        # Calculate total mass
        total_mass = sum(masses)
        
        raw_data = {
            'residue_masses': masses,
            'average_mass': np.mean(masses) if masses else 0,
            'instability_index': instability_idx,
            'total_mass': total_mass,
            'charged_residues': charged_count
        }
        self.raw_data = raw_data
        return raw_data

    def _transform_mass_per_charge(self, 
                                  total_mass: float, 
                                  charged_count: int) -> float:
        """
        Transform mass per charge ratio to a ML-friendly distribution.
        
        Args:
            total_mass: Total mass of the sequence
            charged_count: Number of charged residues
            
        Returns:
            Transformed mass per charge ratio [0-1]
        """
        params = self.mass_per_charge_params
        
        # Handle case with no charged residues
        if charged_count == 0 or total_mass == 0:
            return params['default_value']
        
        # Calculate raw ratio and apply scaling
        raw_ratio = total_mass / charged_count
        scaled_ratio = raw_ratio * params['scale_factor']
        
        # Apply transformation based on method
        method = params['method']
        if method == 'log':
            transformed = np.log1p(scaled_ratio)
        elif method == 'sqrt':
            transformed = np.sqrt(scaled_ratio)
        else:
            transformed = scaled_ratio
            
        # Apply min-max normalization if range is valid
        min_val = params['min_val']
        max_val = params['max_val']
        
        if max_val > min_val:
            result = (transformed - min_val) / (max_val - min_val)
            # Clip to [0, 1] range to handle outliers
            result = np.clip(result, 0, 1)
        else:
            result = transformed
            
        return result

    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready residue properties with transformed features.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with ML-ready properties
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            raw_data = self.raw_data
        else:
            raw_data = self.calculate_raw(sequence)
            
        # Extract values with safe defaults
        masses = raw_data.get('residue_masses', [])
        total_mass = raw_data.get('total_mass', 0)
        charged_count = raw_data.get('charged_residues', 0)
        
        # Calculate transformed mass per charge ratio
        mass_per_charge_ratio = self._transform_mass_per_charge(total_mass, charged_count)
        
        ml_data = {
            'average_mass': np.mean(masses) if masses else 0,
            'mass_std': np.std(masses) if masses else 0,
            'instability_index': raw_data.get('instability_index', 0),
            'mass_per_charge_ratio': mass_per_charge_ratio
        }
        self.ml_data = ml_data
        return ml_data
    
    def fit_mass_per_charge_params(self, mass_values: np.ndarray, charge_counts: np.ndarray):
        """
        Fit transformation parameters for mass_per_charge feature.
        
        Args:
            mass_values: Array of total mass values
            charge_counts: Array of charged residue counts
        """
        params = self.mass_per_charge_params
        method = params['method']
        scale_factor = params['scale_factor']
        
        # Calculate raw ratios where charge count > 0
        valid_indices = charge_counts > 0
        if np.sum(valid_indices) == 0:
            # No valid data points, use defaults
            params['min_val'] = 0.0
            params['max_val'] = 1.0
            return
            
        masses = mass_values[valid_indices]
        charges = charge_counts[valid_indices]
        
        raw_ratios = masses / charges
        scaled_ratios = raw_ratios * scale_factor
        
        # Apply transformation to get distribution
        if method == 'log':
            transformed = np.log1p(scaled_ratios)
        elif method == 'sqrt':
            transformed = np.sqrt(scaled_ratios)
        else:
            transformed = scaled_ratios
            
        # Set min/max values for normalization, handling outliers
        # Use percentiles instead of min/max to be robust to extreme values
        params['min_val'] = np.percentile(transformed, 1)  # 1st percentile
        params['max_val'] = np.percentile(transformed, 99)  # 99th percentile
    
    def inverse_transform_mass_per_charge(self, transformed_value: float) -> float:
        """
        Convert a transformed mass_per_charge_ratio back to its original scale.
        
        Args:
            transformed_value: Transformed value [0-1]
            
        Returns:
            Original mass per charge ratio
        """
        params = self.mass_per_charge_params
        
        # If it's the default value for no charges, return infinity
        if abs(transformed_value - params['default_value']) < 1e-6:
            return float('inf')
            
        # Undo min-max scaling
        min_val = params['min_val']
        max_val = params['max_val']
        
        if max_val > min_val:
            unscaled = transformed_value * (max_val - min_val) + min_val
        else:
            unscaled = transformed_value
            
        # Undo transformation
        method = params['method']
        scale_factor = params['scale_factor']
        
        if method == 'log':
            original = (np.exp(unscaled) - 1) / scale_factor
        elif method == 'sqrt':
            original = (unscaled ** 2) / scale_factor
        else:
            original = unscaled / scale_factor
            
        return original

    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Custom plotting for residue properties with improved visualization.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1) All residue masses distribution
        if "residue_masses" in raw_data.columns:
            raw_masses_flat = []
            for masses in raw_data["residue_masses"].dropna():
                if isinstance(masses, list):
                    raw_masses_flat.extend(masses)
                elif isinstance(masses, str):
                    # Handle string representation of list from CSV
                    try:
                        import ast
                        parsed = ast.literal_eval(masses)
                        if isinstance(parsed, list):
                            raw_masses_flat.extend(parsed)
                    except:
                        pass
            
            if raw_masses_flat:
                sns.histplot(raw_masses_flat, ax=axes[0, 0], bins=50, color='blue')
                axes[0, 0].set_title('All Residue Masses (raw)')
            else:
                axes[0, 0].text(0.5, 0.5, 'No residue mass data', ha='center', va='center')
                axes[0, 0].set_title('All Residue Masses (raw)')
        else:
            axes[0, 0].set_visible(False)

        # 2) Average mass
        if "average_mass" in ml_data.columns:
            sns.histplot(ml_data["average_mass"].dropna(), ax=axes[0, 1], bins=50, color='green')
            axes[0, 1].set_title('Average Residue Mass (ML)')
        else:
            axes[0, 1].set_visible(False)

        # 3) Mass std dev
        if "mass_std" in ml_data.columns:
            sns.histplot(ml_data["mass_std"].dropna(), ax=axes[0, 2], bins=50, color='orange')
            axes[0, 2].set_title('Mass Std Dev (ML)')
        else:
            axes[0, 2].set_visible(False)

        # 4) Instability index
        if "instability_index" in ml_data.columns:
            sns.histplot(ml_data["instability_index"].dropna(), ax=axes[1, 0], bins=50, color='red')
            axes[1, 0].set_title('Instability Index (ML)')
            # Reference line at 40
            axes[1, 0].axvline(x=40, color='black', linestyle='--', label='Stability threshold')
            axes[1, 0].legend()
        else:
            axes[1, 0].set_visible(False)

        # 5) Transformed mass per charge ratio
        if "mass_per_charge_ratio" in ml_data.columns:
            mpc = ml_data["mass_per_charge_ratio"].dropna()
            
            sns.histplot(mpc, ax=axes[1, 1], bins=50, color='purple')
            axes[1, 1].set_title('Mass per Charge Ratio (transformed)')
            
            # Add explanation of transformation
            method = self.mass_per_charge_params.get('method', 'unknown')
            scale = self.mass_per_charge_params.get('scale_factor', 1.0)
            
            axes[1, 1].text(0.95, 0.95, f"Transform: {method}", 
                           transform=axes[1, 1].transAxes, ha='right', va='top', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
            axes[1, 1].text(0.95, 0.87, f"Scale factor: {scale}", 
                           transform=axes[1, 1].transAxes, ha='right', va='top', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        else:
            axes[1, 1].set_visible(False)

        # 6) Empty or additional plot
        axes[1, 2].set_visible(False)

        plt.tight_layout()
        plt.suptitle("Residue Properties Distributions", fontsize=16, y=0.98)
        
        return fig

class PeriodicPatternsFeature(BaseFeature):
    """
    Periodic patterns feature implementation with improved transformations.
    
    This feature measures patterns that repeat within a sequence at different intervals,
    which can indicate functional repeats, structural motifs, or sequence design patterns.
    
    ML-ready features:
    - repeats_2_density: Transformed density of 2-residue repeats
    - repeats_3_density: Transformed density of 3-residue repeats
    - repeats_4_density: Transformed density of 4-residue repeats
    - repeat_ratio_2_3: Transformed ratio of 2-mer to 3-mer repeats
    """
    
    def __init__(self):
        super().__init__()
        self.raw_columns = ['repeats_2', 'repeats_3', 'repeats_4']
        self.ml_columns = ['repeats_2_density', 'repeats_3_density', 'repeats_4_density', 'repeat_ratio_2_3']
        
        # Transformation parameters for each feature
        self.transform_params = {
            'repeats_2_density': {
                'scale_factor': 50.0,  # Scaling factor for small values
                'method': 'log',       # Transformation method
                'min_val': 0.0,        # Will be updated during fitting
                'max_val': 1.0         # Will be updated during fitting
            },
            'repeats_3_density': {
                'scale_factor': 150.0,
                'method': 'log',
                'min_val': 0.0,
                'max_val': 1.0
            },
            'repeats_4_density': {
                'scale_factor': 300.0,
                'method': 'log',
                'min_val': 0.0,
                'max_val': 1.0
            },
            'repeat_ratio_2_3': {
                'scale_factor': 2.0,
                'method': 'sqrt',     # Square root is milder than log
                'min_val': 0.0,
                'max_val': 1.0
            }
        }
    
    def calculate_raw(self, sequence: str) -> Dict[str, int]:
        """
        Calculate raw repeat counts.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with repeat counts for different lengths
        """
        repeats = {}
        for length in range(2, 5):
            count = 0
            for i in range(len(sequence) - length):
                pattern = sequence[i:i + length]
                if sequence.count(pattern) > 1:
                    count += 1
            repeats[f'repeats_{length}'] = count
        return repeats
    
    def _transform_value(self, value: float, feature_name: str) -> float:
        """
        Apply transformation to spread out values near zero.
        
        Args:
            value: Raw density or ratio value
            feature_name: Name of the feature being transformed
            
        Returns:
            Transformed value suitable for ML
        """
        if value <= 0:
            return 0.0
            
        params = self.transform_params.get(feature_name, {})
        if not params:
            return value
            
        scale_factor = params.get('scale_factor', 1.0)
        method = params.get('method', 'none')
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        
        # Apply scaling to spread out small values
        scaled = scale_factor * value
        
        # Apply transformation based on method
        if method == 'log':
            transformed = np.log1p(scaled)
        elif method == 'sqrt':
            transformed = np.sqrt(scaled)
        elif method == 'cbrt':
            transformed = np.cbrt(scaled)
        else:
            transformed = scaled
            
        # Apply min-max normalization if range is valid
        if max_val > min_val:
            result = (transformed - min_val) / (max_val - min_val)
            # Clip to [0, 1] range to handle outliers
            result = np.clip(result, 0, 1)
        else:
            result = transformed
            
        return result
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version with normalized and transformed counts.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with transformed repeat metrics
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            raw_counts = self.raw_data
        else:
            raw_counts = self.calculate_raw(sequence)
            self.raw_data = raw_counts
            
        length = len(sequence)
        
        # Calculate raw density values
        raw_features = {}
        for key, count in raw_counts.items():
            raw_features[f'{key}_density'] = count / length if length > 0 else 0
        
        # Calculate raw ratio
        if raw_counts['repeats_3'] > 0:
            raw_features['repeat_ratio_2_3'] = raw_counts['repeats_2'] / raw_counts['repeats_3']
        else:
            raw_features['repeat_ratio_2_3'] = 0.0
        
        # Apply transformations to each feature
        ml_features = {}
        for feature_name, value in raw_features.items():
            ml_features[feature_name] = self._transform_value(value, feature_name)
        
        self.ml_data = ml_features    
        return ml_features
    
    def fit_transformation_params(self, data: Dict[str, np.ndarray]):
        """
        Fit transformation parameters based on data distribution.
        
        Args:
            data: Dictionary mapping feature names to arrays of values
        """
        for feature_name, values in data.items():
            if feature_name not in self.transform_params:
                continue
                
            params = self.transform_params[feature_name]
            method = params.get('method', 'none')
            scale_factor = params.get('scale_factor', 1.0)
            
            # Filter out zeros for fitting
            positive_values = values[values > 0]
            if len(positive_values) == 0:
                # Default parameters if no positive values
                params['min_val'] = 0.0
                params['max_val'] = 1.0
                continue
                
            # Apply scaling and transformation to get distribution
            if method == 'log':
                transformed = np.log1p(scale_factor * positive_values)
            elif method == 'sqrt':
                transformed = np.sqrt(scale_factor * positive_values)
            elif method == 'cbrt':
                transformed = np.cbrt(scale_factor * positive_values)
            else:
                transformed = scale_factor * positive_values
                
            # Set min/max values for normalization
            params['min_val'] = np.min(transformed)
            params['max_val'] = np.max(transformed)
    
    def inverse_transform(self, transformed_value: float, feature_name: str) -> float:
        """
        Convert a transformed value back to its original scale.
        
        Args:
            transformed_value: Transformed value
            feature_name: Name of the feature
            
        Returns:
            Original value
        """
        if transformed_value <= 0:
            return 0.0
            
        params = self.transform_params.get(feature_name, {})
        if not params:
            return transformed_value
            
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        scale_factor = params.get('scale_factor', 1.0)
        method = params.get('method', 'none')
        
        # Undo min-max scaling
        if max_val > min_val:
            unscaled = transformed_value * (max_val - min_val) + min_val
        else:
            unscaled = transformed_value
            
        # Undo transformation
        if method == 'log':
            original = (np.exp(unscaled) - 1) / scale_factor
        elif method == 'sqrt':
            original = (unscaled ** 2) / scale_factor
        elif method == 'cbrt':
            original = (unscaled ** 3) / scale_factor
        else:
            original = unscaled / scale_factor
            
        return original
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Custom plotting for PeriodicPatternsFeature distributions.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        # Create figure with rows for raw and ML-ready features
        fig = plt.figure(figsize=(16, 10))
        
        # Calculate number of columns based on features
        n_raw_features = len(self.raw_columns)
        n_ml_features = len(self.ml_columns)
        n_cols = max(n_raw_features, n_ml_features)
        
        # Create GridSpec
        gs = plt.GridSpec(2, n_cols, figure=fig, hspace=0.4, wspace=0.3)
        
        # Plot raw features
        for idx, col in enumerate(self.raw_columns):
            ax = fig.add_subplot(gs[0, idx])
            
            if col in raw_data.columns:
                data = raw_data[col].dropna()
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                else:
                    # Use log scale for x-axis if data is highly skewed
                    sns.histplot(data, ax=ax, bins=30, kde=True, color='blue', alpha=0.6)
                    
                    # Check if data is highly skewed
                    if data.max() > 10 * data.median():
                        ax.set_xscale('log')
                        ax.text(0.95, 0.95, "Log scale", transform=ax.transAxes, 
                               ha='right', va='top', fontsize=8,
                               bbox=dict(facecolor='white', alpha=0.7))
                        
                ax.set_title(f"Raw {col}")
            else:
                ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center')
                ax.set_title(f"Raw {col}")
        
        # Plot ML-ready features
        for idx, col in enumerate(self.ml_columns):
            ax = fig.add_subplot(gs[1, idx])
            
            if col in ml_data.columns:
                data = ml_data[col].dropna()
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                else:
                    sns.histplot(data, ax=ax, bins=30, kde=True, color='red', alpha=0.6)
                    
                    # Add annotation about transformation
                    if col in self.transform_params:
                        params = self.transform_params[col]
                        method = params.get('method', 'none')
                        scale = params.get('scale_factor', 1.0)
                        
                        ax.text(0.95, 0.95, f"Transform: {method}", 
                               transform=ax.transAxes, ha='right', va='top', 
                               fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                        
                        ax.text(0.95, 0.87, f"Scale factor: {scale:.1f}", 
                               transform=ax.transAxes, ha='right', va='top', 
                               fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                        
                ax.set_title(f"ML-ready {col}")
            else:
                ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center')
                ax.set_title(f"ML-ready {col}")
        
        plt.suptitle("Periodic Patterns Distributions", fontsize=16, y=0.98)
        plt.tight_layout()
        return fig

class PositionalFeature(BaseFeature):
    """Analyze position-specific patterns in sequence"""
    
    def __init__(self):
        super().__init__()
        self.terminal_length = 5  # Look at first/last 5 residues
        # Define important residue groups for terminals
        self.charged = set('DEKR')
        self.hydrophobic = set('AILMFWV')
        self.raw_columns = ['n_term_seq', 'c_term_seq', 'n_term_charged', 'c_term_charged', 
                           'n_term_hydrophobic', 'c_term_hydrophobic', 'terminal_charge_diff']
        self.ml_columns = ['n_term_charged_frac', 'c_term_charged_frac', 'n_term_hydrophobic_frac',
                          'c_term_hydrophobic_frac', 'terminal_charge_bias']
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate raw positional features"""
        n_term = sequence[:self.terminal_length]
        c_term = sequence[-self.terminal_length:]
        
        return {
            'n_term_seq': n_term,
            'c_term_seq': c_term,
            'n_term_charged': sum(aa in self.charged for aa in n_term),
            'c_term_charged': sum(aa in self.charged for aa in c_term),
            'n_term_hydrophobic': sum(aa in self.hydrophobic for aa in n_term),
            'c_term_hydrophobic': sum(aa in self.hydrophobic for aa in c_term),
            'terminal_charge_diff': (
                sum(aa in self.charged for aa in n_term) - 
                sum(aa in self.charged for aa in c_term)
            )
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready positional features"""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(sequence)
            self.raw_data = raw
            
        ml_data = {
            'n_term_charged_frac': raw['n_term_charged'] / self.terminal_length,
            'c_term_charged_frac': raw['c_term_charged'] / self.terminal_length,
            'n_term_hydrophobic_frac': raw['n_term_hydrophobic'] / self.terminal_length,
            'c_term_hydrophobic_frac': raw['c_term_hydrophobic'] / self.terminal_length,
            'terminal_charge_bias': raw['terminal_charge_diff'] / self.terminal_length
        }
        self.ml_data = ml_data
        return ml_data
    
class SpecializedResidueFeature(BaseFeature):
    """Analyze special residue groups and their patterns"""
    
    def __init__(self):
        super().__init__()
        self.groups = {
            'aromatic': set('FWY'),
            'oxidation_sensitive': set('CMW'),
            'structure_breaking': set('GP'),
            'tiny': set('AGS')
        }
        raw_cols = []
        ml_cols = []
        for name in self.groups:
            raw_cols.extend([f'{name}_count', f'{name}_distribution_mean'])
            ml_cols.extend([f'{name}_fraction', f'{name}_clustering', f'{name}_max_density'])
        self.raw_columns = raw_cols
        self.ml_columns = ml_cols
        
    def _calculate_group_distribution(self, sequence: str, group: set) -> List[float]:
        """Calculate distribution of group members along sequence"""
        window_size = 5
        if len(sequence) < window_size:
            return [0]  # Return a default value if the sequence is too short
        distribution = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            distribution.append(sum(aa in group for aa in window) / window_size)
        return distribution
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        features = {}
        # Count occurrences and compute summary statistics
        for name, group in self.groups.items():
            count = sum(aa in group for aa in sequence)
            distribution = self._calculate_group_distribution(sequence, group)
            features[f'{name}_count'] = count
            # Instead of returning the full list, return the mean of the distribution
            features[f'{name}_distribution_mean'] = np.mean(distribution) if distribution else 0
        return features
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready specialized residue features"""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(sequence)
            self.raw_data = raw
            
        length = len(sequence)
        features = {}
        
        if length == 0:
            # Return zeros for empty sequences
            for name in self.groups:
                features[f'{name}_fraction'] = 0.0
                features[f'{name}_clustering'] = 0.0
                features[f'{name}_max_density'] = 0.0
            self.ml_data = features
            return features
            
        for name, group in self.groups.items():
            distribution = self._calculate_group_distribution(sequence, group)
            count = sum(aa in group for aa in sequence)
            
            # Handle edge cases more gracefully
            features[f'{name}_fraction'] = count / length
            features[f'{name}_clustering'] = np.std(distribution) if distribution and len(distribution) > 1 else 0.0
            features[f'{name}_max_density'] = max(distribution) if distribution else 0.0
        
        self.ml_data = features        
        return features
    
class AlternatingPatternFeature(BaseFeature):
    """Analyze alternating patterns in sequence properties"""
    
    def __init__(self):
        super().__init__()
        self.big = set('FWYKRM')        # Big side chains
        self.small = set('AGCS')         # Small side chains
        self.rigid = set('WYF')          # Rigid side chains
        self.flexible = set('RKMS')      # Flexible side chains
        self.raw_columns = ['big_small_alternation', 'rigid_flexible_alternation', 'big_residues', 
                           'small_residues', 'rigid_residues', 'flexible_residues']
        self.ml_columns = ['big_small_alternation', 'rigid_flexible_alternation', 'big_fraction', 
                          'small_fraction', 'rigid_fraction', 'flexible_fraction']
        
    def _calculate_alternation_score(self, sequence: str, group1: set, group2: set) -> float:
        """Calculate how well properties alternate"""
        score = 0
        for i in range(len(sequence)-1):
            if (sequence[i] in group1 and sequence[i+1] in group2) or \
               (sequence[i] in group2 and sequence[i+1] in group1):
                score += 1
        return score / (len(sequence)-1) if len(sequence) > 1 else 0
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate raw alternating pattern features"""
        return {
            'big_small_alternation': self._calculate_alternation_score(sequence, self.big, self.small),
            'rigid_flexible_alternation': self._calculate_alternation_score(sequence, self.rigid, self.flexible),
            'big_residues': sum(aa in self.big for aa in sequence),
            'small_residues': sum(aa in self.small for aa in sequence),
            'rigid_residues': sum(aa in self.rigid for aa in sequence),
            'flexible_residues': sum(aa in self.flexible for aa in sequence)
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready alternating pattern features"""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(sequence)
            self.raw_data = raw
            
        length = len(sequence)
        
        ml_data = {
            'big_small_alternation': raw['big_small_alternation'],
            'rigid_flexible_alternation': raw['rigid_flexible_alternation'],
            'big_fraction': raw['big_residues'] / length,
            'small_fraction': raw['small_residues'] / length,
            'rigid_fraction': raw['rigid_residues'] / length,
            'flexible_fraction': raw['flexible_residues'] / length
        }
        self.ml_data = ml_data
        return ml_data
