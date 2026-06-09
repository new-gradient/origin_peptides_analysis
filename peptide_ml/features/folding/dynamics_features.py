from typing import Dict, List, Any
import json
import numpy as np
from Bio.PDB import PDBParser, PPBuilder, Structure
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import seaborn as sns
import pandas as pd
from ..base_feature import BaseFeature  # Ensure BaseFeature is available in your package
# import matplotlib
import matplotlib.pyplot as plt
import re 
import math

class FlexibilityFeatureMixin:
    """Mixin for extracting flexibility-related data."""
    def _get_structure(self, pdb_path: str) -> Structure:
        parser = PDBParser(QUIET=True)
        return parser.get_structure("peptide", pdb_path)
        
    def _get_pae_data(self, pdb_path: str) -> Dict:
        """Get PAE data from corresponding JSON file.
        
        The JSON filename is derived from the pdb_path by replacing '.pdb' with '.json'
        and replacing 'relaxed' with 'scores'.
        """
        json_path = pdb_path.replace('.pdb', '.json')
        json_path = json_path.replace('relaxed', 'scores')
        with open(json_path) as f:
            return json.load(f)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
from typing import Dict, Any, List, Tuple
from Bio.PDB import PDBParser, PPBuilder, Structure

class BackboneDihedralFeature(BaseFeature, FlexibilityFeatureMixin):
    """
    Backbone dihedral angle statistics to assess conformational preferences.
    
    The phi/psi angles describe the backbone conformation and indicate:
    - Structural regularity/disorder
    - Conformational strain
    - Potential flexibility hotspots
    
    For templating: More regular/unstrained conformations may be better templates
    as they represent stable, well-defined structures.
    
    ML-ready output columns:
    - phi_psi_correlation: Float [-1,1], correlation between phi/psi angles
      Values near -1 indicate regular secondary structure,
      values near 0 indicate more disordered regions.
    - ramachandran_outliers: Float [0,1], transformed measure of backbone conformational strain
      based on residues outside allowed regions of the Ramachandran plot.
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['phi_angles', 'psi_angles']
        self.ml_columns = ['phi_psi_correlation', 'ramachandran_outliers']
        
        # Define allowed regions in Ramachandran plot (in radians)
        # These regions approximate common secondary structures and are expanded 
        # slightly to capture more of the typical distribution
        
        # Alpha-helix region
        self.alpha_region = {
            'phi_min': -1.3, 'phi_max': -0.5,
            'psi_min': -0.9, 'psi_max': -0.1
        }
        
        # Beta-sheet region
        self.beta_region = {
            'phi_min': -2.3, 'phi_max': -1.6,
            'psi_min': 1.8, 'psi_max': 2.8
        }
        
        # Left-handed helix region
        self.l_alpha_region = {
            'phi_min': 0.6, 'phi_max': 1.4,
            'psi_min': 0.0, 'psi_max': 1.0
        }
        
        # Additional allowed region for turns and loops
        self.turn_region = {
            'phi_min': -2.7, 'phi_max': -1.8,
            'psi_min': -0.6, 'psi_max': 0.6
        }
        
        # Transformation parameters for ramachandran_outliers
        self.outlier_transform_params = {
            'method': 'sqrt',  # Use square root transformation (milder than log)
            'scale_factor': 2.0,  # Moderate scaling factor
            'center_value': 0.5,  # Target center for the distribution
            'spread_factor': 0.3  # Controls the spread of the distribution
        }
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, Any]:
        structure = self._get_structure(pdb_path)
        phi_psi = []
        # Iterate over all models, chains, and polypeptides
        for model in structure:
            for chain in model:
                polypeptides = PPBuilder().build_peptides(chain)
                for poly in polypeptides:
                    phi_psi.extend(poly.get_phi_psi_list())
        # Extract phi and psi angles, skipping None values
        phi = np.array([angle[0] for angle in phi_psi if angle[0] is not None])
        psi = np.array([angle[1] for angle in phi_psi if angle[1] is not None])
        raw_data = {
            'phi_angles': phi,
            'psi_angles': psi
        }
        self.raw_data = raw_data
        return raw_data
    
    def _is_in_allowed_region(self, phi: float, psi: float) -> bool:
        """
        Check if a phi/psi angle pair is in any of the allowed regions.
        
        Args:
            phi: Phi angle in radians
            psi: Psi angle in radians
            
        Returns:
            True if in allowed region, False otherwise
        """
        # Check alpha-helix region
        in_alpha = (self.alpha_region['phi_min'] <= phi <= self.alpha_region['phi_max'] and
                   self.alpha_region['psi_min'] <= psi <= self.alpha_region['psi_max'])
        
        # Check beta-sheet region
        in_beta = (self.beta_region['phi_min'] <= phi <= self.beta_region['phi_max'] and
                  self.beta_region['psi_min'] <= psi <= self.beta_region['psi_max'])
        
        # Check left-handed helix region
        in_l_alpha = (self.l_alpha_region['phi_min'] <= phi <= self.l_alpha_region['phi_max'] and
                     self.l_alpha_region['psi_min'] <= psi <= self.l_alpha_region['psi_max'])
        
        # Check turn region
        in_turn = (self.turn_region['phi_min'] <= phi <= self.turn_region['phi_max'] and
                  self.turn_region['psi_min'] <= psi <= self.turn_region['psi_max'])
        
        # Return True if in any allowed region
        return in_alpha or in_beta or in_l_alpha or in_turn
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        """Calculate ML-ready features from backbone dihedrals."""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        # Check if required keys exist
        if 'phi_angles' not in raw or 'psi_angles' not in raw:
            print(f"Warning: Missing required angles in raw data for {pdb_path}")
            return {'phi_psi_correlation': 0, 'ramachandran_outliers': 0}
        
        # Handle various potential input formats
        phi_data = raw['phi_angles']
        psi_data = raw['psi_angles']
        
        # Parse string representations of numpy arrays from CSV
        try:
            if isinstance(phi_data, str):
                # Handle multi-line string representation of numpy arrays
                import re
                # Remove brackets and clean up whitespace/newlines
                phi_str = phi_data.replace('[', '').replace(']', '').replace('\n', ' ')
                # Extract all floating point numbers
                phi_values = re.findall(r'-?\d+\.\d+', phi_str)
                phi = np.array([float(x) for x in phi_values])
            else:
                phi = np.array(phi_data)
                
            if isinstance(psi_data, str):
                # Same cleaning for psi angles
                import re
                psi_str = psi_data.replace('[', '').replace(']', '').replace('\n', ' ')
                psi_values = re.findall(r'-?\d+\.\d+', psi_str)
                psi = np.array([float(x) for x in psi_values])
            else:
                psi = np.array(psi_data)
                
        except Exception as e:
            print(f"Warning: Cannot parse angle data for {pdb_path}: {e}")
            return {'phi_psi_correlation': 0, 'ramachandran_outliers': 0}
        
        # Ensure we have enough data points
        if phi.size < 2 or psi.size < 2:
            print(f"Warning: Not enough angle data points for {pdb_path}")
            return {'phi_psi_correlation': 0, 'ramachandran_outliers': 0}
        
        # Compute correlation with error handling
        try:
            correlation = np.corrcoef(phi, psi)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        except Exception as e:
            print(f"Warning: Error calculating correlation: {e}")
            correlation = 0
        
        # Calculate outliers based on allowed regions in Ramachandran plot
        try:
            # Match phi and psi arrays by length if needed
            min_len = min(phi.size, psi.size)
            phi = phi[:min_len]
            psi = psi[:min_len]
            
            # Count residues in disallowed regions
            outliers = 0
            for i in range(min_len):
                if not self._is_in_allowed_region(phi[i], psi[i]):
                    outliers += 1
                    
            outlier_fraction = outliers / min_len if min_len > 0 else 0
            
            # Apply balanced transformation to get better distribution
            transformed_outlier = self._transform_outlier_fraction(outlier_fraction)
                
        except Exception as e:
            print(f"Warning: Error calculating Ramachandran outliers: {e}")
            transformed_outlier = 0
        
        ml_data = {
            'phi_psi_correlation': correlation,
            'ramachandran_outliers': transformed_outlier
        }
        self.ml_data = ml_data
        return ml_data
    
    def _transform_outlier_fraction(self, outlier_fraction: float) -> float:
        """
        Apply a balanced transformation to the outlier fraction.
        
        This uses a milder transformation approach to spread out values
        while maintaining a more balanced distribution.
        
        Args:
            outlier_fraction: Raw fraction of residues outside allowed regions
            
        Returns:
            Transformed value suitable for ML
        """
        if outlier_fraction <= 0:
            return 0.0
            
        params = self.outlier_transform_params
        method = params['method']
        scale_factor = params['scale_factor']
        center_value = params['center_value']
        spread = params['spread_factor']
        
        # Apply scaling to spread out small values
        scaled = scale_factor * outlier_fraction
        
        # Apply appropriate transformation based on method
        if method == 'sqrt':
            # Square root transformation (milder than log)
            transformed = np.sqrt(scaled)
        elif method == 'log':
            # Log transformation (more aggressive)
            transformed = np.log1p(scaled)
        elif method == 'cbrt':
            # Cube root transformation (between sqrt and log)
            transformed = np.cbrt(scaled)
        else:
            # Default to sqrt if unknown method
            transformed = np.sqrt(scaled)
            
        # Apply sigmoid-like function to center around the desired value
        # This helps create a more balanced distribution
        sigmoid = 1 / (1 + np.exp(-(transformed - center_value) / spread))
        
        # Scale to [0, 1] range
        result = sigmoid
        
        return result
    
    def inverse_transform_outliers(self, transformed_value: float) -> float:
        """
        Convert a transformed ramachandran_outliers value back to its original scale.
        
        Args:
            transformed_value: The transformed value
            
        Returns:
            Original outlier fraction
        """
        if transformed_value <= 0:
            return 0.0
            
        params = self.outlier_transform_params
        method = params['method']
        scale_factor = params['scale_factor']
        center_value = params['center_value']
        spread = params['spread_factor']
        
        # Inverse of sigmoid-like transformation
        inv_sigmoid = center_value - spread * np.log(1/transformed_value - 1)
        
        # Inverse of the transformation method
        if method == 'sqrt':
            inv_transformed = inv_sigmoid ** 2
        elif method == 'log':
            inv_transformed = np.exp(inv_sigmoid) - 1
        elif method == 'cbrt':
            inv_transformed = inv_sigmoid ** 3
        else:
            inv_transformed = inv_sigmoid ** 2
            
        # Undo scaling
        return inv_transformed / scale_factor

    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Improved custom plotting for BackboneDihedralFeature.
        
        Handles different data formats and creates cleaner visualizations for:
        - Raw phi angles distribution
        - Raw psi angles distribution
        - ML-ready phi/psi correlation
        - ML-ready ramachandran outlier fraction
        """
        # Create figure with better spacing (2x2 grid only)
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Process phi and psi angles with robust parsing for CSV string format
        phi_vals = []
        psi_vals = []
        
        # More robust extraction of phi angles from CSV format
        if 'phi_angles' in raw_data.columns and not raw_data['phi_angles'].empty:
            for cell in raw_data['phi_angles'].dropna():
                try:
                    # Handle various input formats
                    if isinstance(cell, (list, np.ndarray)):
                        # Direct list or array
                        phi_vals.extend([float(x) for x in cell if x is not None and not np.isnan(x)])
                    elif isinstance(cell, str):
                        # Parse string representation of arrays from CSV
                        # Remove brackets and split by spaces
                        clean_str = cell.replace('[', '').replace(']', '')
                        # Split by space or newline and filter empty strings
                        values = [x for x in re.split(r'[\s\n]+', clean_str) if x.strip()]
                        phi_vals.extend([float(x) for x in values if x and not x.isspace()])
                    else:
                        # Handle single numeric values
                        if cell is not None and not np.isnan(float(cell)):
                            phi_vals.append(float(cell))
                except Exception as e:
                    print(f"Error processing phi value: {e}")
        
        # More robust extraction of psi angles from CSV format
        if 'psi_angles' in raw_data.columns and not raw_data['psi_angles'].empty:
            for cell in raw_data['psi_angles'].dropna():
                try:
                    # Handle various input formats
                    if isinstance(cell, (list, np.ndarray)):
                        # Direct list or array
                        psi_vals.extend([float(x) for x in cell if x is not None and not np.isnan(x)])
                    elif isinstance(cell, str):
                        # Parse string representation of arrays from CSV
                        # Remove brackets and split by spaces
                        clean_str = cell.replace('[', '').replace(']', '')
                        # Split by space or newline and filter empty strings
                        values = [x for x in re.split(r'[\s\n]+', clean_str) if x.strip()]
                        psi_vals.extend([float(x) for x in values if x and not x.isspace()])
                    else:
                        # Handle single numeric values
                        if cell is not None and not np.isnan(float(cell)):
                            psi_vals.append(float(cell))
                except Exception as e:
                    print(f"Error processing psi value: {e}")
        
        # Convert to numpy arrays for easier filtering
        phi_vals = np.array(phi_vals) if phi_vals else np.array([])
        psi_vals = np.array(psi_vals) if psi_vals else np.array([])
        
        # Filter out any remaining NaN or invalid values
        if len(phi_vals) > 0:
            phi_vals = phi_vals[~np.isnan(phi_vals)]
        if len(psi_vals) > 0:
            psi_vals = psi_vals[~np.isnan(psi_vals)]
        
        # Get ML data with better error handling
        ml_corr = ml_data['phi_psi_correlation'].dropna() if 'phi_psi_correlation' in ml_data.columns else pd.Series()
        ml_outliers = ml_data['ramachandran_outliers'].dropna() if 'ramachandran_outliers' in ml_data.columns else pd.Series()
        
        # Plot 1: Raw Phi Angles
        ax1 = fig.add_subplot(gs[0, 0])
        if len(phi_vals) > 0:
            sns.histplot(phi_vals, ax=ax1, bins=30, kde=True, color='blue', alpha=0.6)
            ax1.set_title("Raw Phi Angles", fontsize=12, pad=10)
            ax1.set_xlabel("Angle (radians)", fontsize=10)
            ax1.set_ylabel("Count", fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No phi angle data available', 
                    ha='center', va='center', fontsize=12)
            ax1.set_title("Raw Phi Angles", fontsize=12, pad=10)
        
        # Plot 2: Raw Psi Angles
        ax2 = fig.add_subplot(gs[0, 1])
        if len(psi_vals) > 0:
            sns.histplot(psi_vals, ax=ax2, bins=30, kde=True, color='blue', alpha=0.6)
            ax2.set_title("Raw Psi Angles", fontsize=12, pad=10)
            ax2.set_xlabel("Angle (radians)", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No psi angle data available', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title("Raw Psi Angles", fontsize=12, pad=10)
        
        # Plot 3: ML-ready Phi/Psi Correlation
        ax3 = fig.add_subplot(gs[1, 0])
        if not ml_corr.empty:
            sns.histplot(ml_corr, ax=ax3, bins=20, kde=True, color='red', alpha=0.6)
            ax3.set_title("ML-ready Phi/Psi Correlation", fontsize=12, pad=10)
            ax3.set_xlabel("Correlation Coefficient", fontsize=10)
            ax3.set_ylabel("Count", fontsize=10)
            # Set x-axis limits to correlation range
            ax3.set_xlim(-1.05, 1.05)
        else:
            ax3.text(0.5, 0.5, 'No ML correlation data available', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title("ML-ready Phi/Psi Correlation", fontsize=12, pad=10)
        
        # Plot 4: ML-ready Ramachandran Outliers
        ax4 = fig.add_subplot(gs[1, 1])
        if not ml_outliers.empty:
            sns.histplot(ml_outliers, ax=ax4, bins=20, kde=True, color='red', alpha=0.6)
            ax4.set_title("ML-ready Ramachandran Outliers", fontsize=12, pad=10)
            ax4.set_xlabel("Transformed Outlier Score", fontsize=10)
            ax4.set_ylabel("Count", fontsize=10)
            
            # Add explanation of the transformation
            method = self.outlier_transform_params.get('method', 'unknown')
            ax4.text(0.95, 0.95, f"Transform: {method} + sigmoid", 
                   transform=ax4.transAxes, ha='right', va='top', 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
            # Set x-axis limits, avoiding identical limits
            ax4.set_xlim(-0.05, 1.05)  # Fixed range for [0,1]
        else:
            ax4.text(0.5, 0.5, 'No ML outlier data available', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title("ML-ready Ramachandran Outliers", fontsize=12, pad=10)
        
        # Add a descriptive annotation if data appears to be in radians rather than degrees
        if len(phi_vals) > 0 and np.max(np.abs(phi_vals)) < 3.5:  # Likely radians
            fig.text(0.5, 0.02, 
                    "Note: Angles appear to be in radians rather than degrees", 
                    ha='center', fontsize=10, style='italic')
        
        # Add an overall title
        plt.suptitle("Backbone Dihedral Angle Analysis", fontsize=16, y=0.98)
        
        # Add Ramachandran plot if we have both phi and psi values
        if len(phi_vals) > 50 and len(psi_vals) > 50:
            # Create a new figure for the Ramachandran plot
            rama_fig = plt.figure(figsize=(10, 10))
            ax_rama = rama_fig.add_subplot(111)
            
            # Create scatter plot with hexbin for density
            h = ax_rama.hexbin(phi_vals, psi_vals, gridsize=50, cmap='viridis', mincnt=1)
            plt.colorbar(h, ax=ax_rama, label='Count')
            
            # Draw allowed regions
            # Alpha helix region
            ax_rama.add_patch(plt.Rectangle(
                (self.alpha_region['phi_min'], self.alpha_region['psi_min']),
                self.alpha_region['phi_max'] - self.alpha_region['phi_min'],
                self.alpha_region['psi_max'] - self.alpha_region['psi_min'],
                linewidth=2, edgecolor='r', facecolor='none', label='α-helix'
            ))
            
            # Beta sheet region
            ax_rama.add_patch(plt.Rectangle(
                (self.beta_region['phi_min'], self.beta_region['psi_min']),
                self.beta_region['phi_max'] - self.beta_region['phi_min'],
                self.beta_region['psi_max'] - self.beta_region['psi_min'],
                linewidth=2, edgecolor='g', facecolor='none', label='β-sheet'
            ))
            
            # Left-handed helix region
            ax_rama.add_patch(plt.Rectangle(
                (self.l_alpha_region['phi_min'], self.l_alpha_region['psi_min']),
                self.l_alpha_region['phi_max'] - self.l_alpha_region['phi_min'],
                self.l_alpha_region['psi_max'] - self.l_alpha_region['psi_min'],
                linewidth=2, edgecolor='b', facecolor='none', label='L-α-helix'
            ))
            
            # Turn region
            ax_rama.add_patch(plt.Rectangle(
                (self.turn_region['phi_min'], self.turn_region['psi_min']),
                self.turn_region['phi_max'] - self.turn_region['phi_min'],
                self.turn_region['psi_max'] - self.turn_region['psi_min'],
                linewidth=2, edgecolor='purple', facecolor='none', label='Turns'
            ))
            
            ax_rama.set_xlabel('Phi (radians)')
            ax_rama.set_ylabel('Psi (radians)')
            ax_rama.set_xlim(-3.2, 3.2)
            ax_rama.set_ylim(-3.2, 3.2)
            ax_rama.set_title('Ramachandran Plot with Allowed Regions')
            ax_rama.legend(loc='upper right')
            ax_rama.grid(alpha=0.3)
            
            # Save this as a separate plot
            rama_fig.tight_layout()
            # Close to avoid displaying in notebook
            plt.close(rama_fig)
        
        plt.tight_layout()
        return fig
    
class ConfidenceScoreFeature(BaseFeature, FlexibilityFeatureMixin):
    """
    AlphaFold2 confidence metrics indicating prediction reliability.
    
    pLDDT (predicted local-distance difference test) scores reflect:
    - Local structure accuracy
    - Conformational stability
    - Potential for alternative conformations
    
    ML-ready output columns:
    - mean_plddt: Float [0,1], normalized average pLDDT score
    - plddt_variance: Float [0,1], squared normalized standard deviation of pLDDT
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['plddt_values', 'mean_plddt', 'plddt_std']
        self.ml_columns = ['mean_plddt', 'plddt_variance']
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, Any]:
        scores = self._get_pae_data(pdb_path)
        plddt = np.array(scores['plddt'])
        raw_data = {
            'plddt_values': plddt.tolist(),
            'mean_plddt': np.mean(plddt),
            'plddt_std': np.std(plddt)
        }
        self.raw_data = raw_data
        return raw_data
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        ml_data = {
            'mean_plddt': raw['mean_plddt'] / 100.0,  # Normalize to [0,1]
            'plddt_variance': (raw['plddt_std'] / 100.0) ** 2
        }
        self.ml_data = ml_data
        return ml_data

    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Custom plotting for ConfidenceScoreFeature.
        
        Plots a histogram of the flattened raw pLDDT values, along with histograms
        for the ML-ready mean pLDDT and pLDDT variance.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Flatten raw pLDDT values
        plddt_vals = []
        if 'plddt_values' in raw_data.columns:
            for cell in raw_data['plddt_values'].dropna():
                if isinstance(cell, list):
                    plddt_vals.extend(cell)
                else:
                    plddt_vals.append(cell)
        if plddt_vals:
            sns.histplot(plddt_vals, ax=axs[0], color='blue', alpha=0.6)
            axs[0].set_title("Raw pLDDT values")
        else:
            axs[0].text(0.5, 0.5, 'No raw pLDDT data', ha='center', va='center')
        if 'mean_plddt' in ml_data.columns:
            sns.histplot(ml_data['mean_plddt'].dropna(), ax=axs[1], color='red', alpha=0.6)
            axs[1].set_title("ML-ready Mean pLDDT")
        else:
            axs[1].text(0.5, 0.5, 'No ML mean pLDDT data', ha='center', va='center')
        if 'plddt_variance' in ml_data.columns:
            sns.histplot(ml_data['plddt_variance'].dropna(), ax=axs[2], color='green', alpha=0.6)
            axs[2].set_title("ML-ready pLDDT Variance")
        else:
            axs[2].text(0.5, 0.5, 'No ML pLDDT variance data', ha='center', va='center')
        plt.tight_layout()
        return fig

class PAEBasedFeature(BaseFeature, FlexibilityFeatureMixin):
    """
    Features derived from predicted aligned error (PAE) between residues.
    
    PAE captures:
    - Global structure confidence
    - Domain organization
    - Potential flexibility between regions
    
    ML-ready output columns:
    - mean_pae: Float [0,1], normalized mean PAE (lower is better)
    - pae_variance: Float [0,1], squared normalized standard deviation of PAE
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['mean_pae', 'max_pae', 'pae_std']
        self.ml_columns = ['mean_pae', 'pae_variance']
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        scores = self._get_pae_data(pdb_path)
        pae = np.array(scores['pae'])
        raw_data = {
            'mean_pae': float(np.mean(pae)),
            'max_pae': float(np.max(pae)),
            'pae_std': float(np.std(pae))
        }
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        ml_data = {
            'mean_pae': raw['mean_pae'] / raw['max_pae'],
            'pae_variance': (raw['pae_std'] / raw['max_pae']) ** 2
        }
        self.ml_data = ml_data
        return ml_data
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Custom plotting for PAEBasedFeature.
        
        Plots a histogram of raw mean PAE and a histogram of ML-ready PAE variance.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        if 'mean_pae' in raw_data.columns:
            sns.histplot(raw_data['mean_pae'].dropna(), ax=axs[0], color='blue', alpha=0.6)
            axs[0].set_title("Raw Mean PAE")
        else:
            axs[0].text(0.5, 0.5, 'No raw mean PAE data', ha='center', va='center')
        if 'pae_variance' in ml_data.columns:
            sns.histplot(ml_data['pae_variance'].dropna(), ax=axs[1], color='red', alpha=0.6)
            axs[1].set_title("ML-ready PAE Variance")
        else:
            axs[1].text(0.5, 0.5, 'No ML PAE variance data', ha='center', va='center')
        plt.tight_layout()
        return fig