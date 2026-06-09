
from typing import Dict, List, Any, Optional # Added Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, DSSP, PDBIO, Structure # Added PDBIO, Structure
import pandas as pd
from pathlib import Path
import tempfile # Added tempfile
import logging # Added logging
import shutil  # To find executable path
import subprocess # For direct test run
import math # For DSSPFeatureMixin compatibility if needed by other classes
import warnings # For DSSP transformation warnings

# Assuming BaseFeature is correctly imported from the package structure
# If running standalone, you might need to uncomment the dummy definition below
from ..base_feature import BaseFeature
# class BaseFeature: # Dummy definition for standalone testing
#     def __init__(self): self.name = self.__class__.__name__
#     def calculate_raw(self, *args, **kwargs): pass
#     def calculate_ml_ready(self, *args, **kwargs): pass
#     def plot_distributions(self, *args, **kwargs): pass

# Configure a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DSSP Mixin with Temporary File Fix ---
class DSSPFeatureMixin:
    """
    Mixin to extract DSSP annotations from a folded PDB file.
    Uses a temporary file strategy to handle potential PDB format issues with mkdssp.
    """
    def _get_dssp_data(self, pdb_path: str) -> List[tuple]:
        logger.info(f"DSSP Attempt: Processing PDB file: {pdb_path}")
        pdb_file = Path(pdb_path)
        if not pdb_file.exists():
            logger.error(f"DSSP Error: PDB file does not exist at {pdb_path}")
            return []

        # Find DSSP executable
        dssp_executable = shutil.which('mkdssp') or shutil.which('dssp')
        if not dssp_executable:
            logger.error("DSSP Error: Could not find 'mkdssp' or 'dssp' executable in PATH.")
            return []
        logger.info(f"DSSP Found: Using executable at {dssp_executable}")

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("peptide", pdb_path)
            if not structure:
                logger.error(f"DSSP Error: BioPython PDBParser returned empty structure for {pdb_path}")
                return []
            model = structure[0] # Assume single model
            if not list(model.get_chains()): # Check if model actually has chains
                 logger.error(f"DSSP Error: No chains found in the parsed model for {pdb_path}")
                 return []
            chain_id = list(model.get_chains())[0].get_id() # Get chain ID after confirming chains exist
        except Exception as parse_e:
             logger.error(f"DSSP Error: Failed to parse PDB file {pdb_path} with BioPython: {parse_e}")
             return []

        # --- Create a temporary, cleaned PDB file ---
        temp_pdb_file = None
        try:
            # Create a temporary file to write the cleaned PDB
            # delete=False is important on some systems for subprocess access
            with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as temp_pdb:
                temp_pdb_file = temp_pdb.name
                io = PDBIO()
                io.set_structure(structure) # Use the parsed structure
                io.save(temp_pdb.name)
            logger.info(f"DSSP Debug: Saved cleaned structure to temporary file: {temp_pdb_file}")

            # --- Call DSSP on the TEMPORARY file ---
            # Pass model and the temp file path. Explicitly pass executable path.
            dssp = DSSP(model, temp_pdb_file, dssp=dssp_executable)

            # Filter annotations for the first chain.
            # Handle potential KeyError if chain_id is not in dssp keys
            dssp_data = [value for key, value in dssp.property_dict.items() if key[0] == chain_id]

            if not dssp_data:
                logger.warning(f"DSSP Warning: No DSSP data found for chain '{chain_id}' in {temp_pdb_file}.")
                # It's possible DSSP ran but found no assignable residues for the chain

            logger.info(f"DSSP Success: Extracted data for {len(dssp_data)} residues using temp file {temp_pdb_file}")
            return dssp_data

        except Exception as e:
            # --- Enhanced Error Reporting ---
            logger.error(f"DSSP Error: DSSP execution failed, even with temp file.")
            logger.error(f"  Input PDB: {pdb_path}")
            logger.error(f"  Temp PDB used: {temp_pdb_file}")
            logger.error(f"  Underlying Exception: {type(e).__name__}: {e}")
            # Optionally run direct subprocess call on the *temp* file for debugging
            if temp_pdb_file and Path(temp_pdb_file).exists():
                 logger.info(f"DSSP Debug: Attempting direct subprocess call on temp file: {dssp_executable} {temp_pdb_file}")
                 try:
                     process = subprocess.run([dssp_executable, temp_pdb_file], capture_output=True, text=True, check=False, timeout=30)
                     logger.error(f"DSSP Debug: Direct call exit code: {process.returncode}")
                     if process.stdout: logger.error(f"DSSP Debug: Direct call STDOUT:\n{process.stdout[-500:]}")
                     if process.stderr: logger.error(f"DSSP Debug: Direct call STDERR:\n{process.stderr[-500:]}")
                 except Exception as sub_e:
                     logger.error(f"DSSP Debug: Error during direct call on temp file: {sub_e}")
            return [] # Return empty list on failure
        finally:
            # --- Clean up the temporary file ---
            if temp_pdb_file and Path(temp_pdb_file).exists():
                try:
                    Path(temp_pdb_file).unlink()
                    logger.info(f"DSSP Debug: Removed temporary file: {temp_pdb_file}")
                except Exception as clean_e:
                    logger.warning(f"DSSP Debug: Could not remove temporary file {temp_pdb_file}: {clean_e}")


class HelixContentFeature(BaseFeature, DSSPFeatureMixin):
    """
    Computes the fraction of residues adopting a helical conformation.
    DSSP codes considered: 'H' (alpha helix), 'G' (3-10 helix), and 'I' (pi helix).
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['helix_fraction']
        self.ml_columns = ['helix_fraction']
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'helix_fraction': 0.0}
        helix_codes = {'H', 'G', 'I'}
        helix_count = sum(1 for entry in dssp_data if entry[2] in helix_codes)
        fraction = helix_count / len(dssp_data)
        raw_data = {'helix_fraction': fraction}
        self.raw_data = raw_data
        return raw_data
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if self.raw_data:
            ml_data =  self.raw_data
            self.ml_data = ml_data
            return ml_data
        else:
            ml_data = self.calculate_raw(pdb_path)
            self.ml_data = ml_data
            return ml_data
            
    
class SheetContentFeature(BaseFeature, DSSPFeatureMixin):
    """
    Computes the fraction of residues in beta-sheet conformations.
    DSSP code considered: 'E'.
    
    ML-ready feature:
    - sheet_fraction: Drastically transformed measure of beta-sheet content
      that distinguishes between peptides with no sheet content and those with
      even a small amount of sheet structure.
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['sheet_fraction']
        self.ml_columns = ['sheet_fraction']
        
        # Thresholds for the 3-tier transformation
        self.threshold_params = {
            'has_sheet_threshold': 0.001,  # Threshold to distinguish zero from non-zero (very sensitive)
            'significant_sheet_threshold': 0.1,  # Threshold for significant sheet content
            
            # Output value ranges for the three tiers
            'zero_sheet_range': (0.0, 0.2),       # Range for zero sheet content
            'minimal_sheet_range': (0.4, 0.6),    # Range for minimal sheet content
            'significant_sheet_range': (0.8, 1.0), # Range for significant sheet content
            
            # Spread within each range (higher = more spread)
            'minimal_spread_factor': 5.0,
            'significant_spread_factor': 3.0
        }
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'sheet_fraction': 0.0}
            
        sheet_count = sum(1 for entry in dssp_data if entry[2] == 'E')
        fraction = sheet_count / len(dssp_data)
        
        raw_data = {'sheet_fraction': fraction}
        self.raw_data = raw_data
        return raw_data
            
    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw_data = self.raw_data
        else:
            raw_data = self.calculate_raw(pdb_path)
            
        # Get raw sheet fraction
        sheet_fraction = raw_data['sheet_fraction']
        
        # Apply drastic transformation
        transformed_value = self._transform_sheet_fraction(sheet_fraction)
        
        ml_data = {'sheet_fraction': transformed_value}
        self.ml_data = ml_data
        return ml_data
    
    def _transform_sheet_fraction(self, value: float) -> float:
        """
        Apply a 3-tier transformation to sheet fraction:
        1. Zero sheet content (0) → mapped to lower range (e.g., 0.0-0.2)
        2. Minimal sheet content (>0 but below threshold) → mapped to middle range (e.g., 0.4-0.6)
        3. Significant sheet content (≥threshold) → mapped to upper range (e.g., 0.8-1.0)
        
        Within each tier, values are spread according to their magnitude.
        
        Args:
            value: Raw sheet fraction (between 0 and 1)
            
        Returns:
            Transformed value with 3-tier distribution
        """
        params = self.threshold_params
        
        # Get threshold values
        has_sheet = params['has_sheet_threshold']
        significant_sheet = params['significant_sheet_threshold']
        
        # Get range values
        zero_min, zero_max = params['zero_sheet_range']
        minimal_min, minimal_max = params['minimal_sheet_range']
        significant_min, significant_max = params['significant_sheet_range']
        
        # Get spread factors
        minimal_spread = params['minimal_spread_factor']
        significant_spread = params['significant_spread_factor']
        
        # Apply tier-based transformation
        if value <= 0.0:
            # Tier 1: Zero sheet content
            return zero_min
        elif value < has_sheet:
            # Still essentially zero (numerical noise)
            # Map to the bottom of the zero range
            return zero_min + (zero_max - zero_min) * (value / has_sheet)
        elif value < significant_sheet:
            # Tier 2: Minimal sheet content
            # Apply aggressive transformation within this range
            
            # Normalize to [0,1] within the minimal range
            normalized = (value - has_sheet) / (significant_sheet - has_sheet)
            
            # Apply power transformation to spread small values
            import numpy as np
            spread = np.power(normalized, 1.0 / minimal_spread)
            
            # Map to the minimal sheet range
            return minimal_min + (minimal_max - minimal_min) * spread
        else:
            # Tier 3: Significant sheet content
            # Apply less aggressive transformation
            
            # Normalize to [0,1] within the significant range (capped at 1.0)
            normalized = min(1.0, (value - significant_sheet) / (1.0 - significant_sheet))
            
            # Apply milder power transformation
            import numpy as np
            spread = np.power(normalized, 1.0 / significant_spread)
            
            # Map to the significant sheet range
            return significant_min + (significant_max - significant_min) * spread
    
    def inverse_transform(self, transformed_value: float) -> float:
        """
        Convert a transformed sheet_fraction back to its original scale.
        
        Args:
            transformed_value: The transformed value
            
        Returns:
            Original sheet fraction
        """
        params = self.threshold_params
        
        # Get threshold values
        has_sheet = params['has_sheet_threshold']
        significant_sheet = params['significant_sheet_threshold']
        
        # Get range values
        zero_min, zero_max = params['zero_sheet_range']
        minimal_min, minimal_max = params['minimal_sheet_range']
        significant_min, significant_max = params['significant_sheet_range']
        
        # Get spread factors
        minimal_spread = params['minimal_spread_factor']
        significant_spread = params['significant_spread_factor']
        
        # Determine which tier the transformed value falls into
        if transformed_value <= zero_max:
            # Tier 1: Zero sheet content
            if transformed_value <= zero_min:
                return 0.0
            else:
                # Reverse the mapping within zero range
                normalized = (transformed_value - zero_min) / (zero_max - zero_min)
                return normalized * has_sheet
                
        elif transformed_value <= minimal_max:
            # Tier 2: Minimal sheet content
            # Normalize within minimal range
            normalized = (transformed_value - minimal_min) / (minimal_max - minimal_min)
            
            # Reverse the power transformation
            import numpy as np
            unspread = np.power(normalized, minimal_spread)
            
            # Map back to original scale
            return has_sheet + unspread * (significant_sheet - has_sheet)
            
        else:
            # Tier 3: Significant sheet content
            # Normalize within significant range
            normalized = (transformed_value - significant_min) / (significant_max - significant_min)
            
            # Reverse the power transformation
            import numpy as np
            unspread = np.power(normalized, significant_spread)
            
            # Map back to original scale
            return significant_sheet + unspread * (1.0 - significant_sheet)
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Create custom plots for SheetContentFeature distributions with clear visualization
        of the 3-tier transformation.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
        
        # Plot raw data
        ax1 = fig.add_subplot(gs[0, 0])
        if 'sheet_fraction' in raw_data.columns:
            data = raw_data['sheet_fraction'].dropna()
            
            if data.empty:
                ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                # Plot primary histogram for all data
                sns.histplot(data=data, ax=ax1, bins=30, kde=True, color='blue', alpha=0.6)
                
                # If many zeros, add text annotation
                zero_percent = (data == 0).mean() * 100
                if zero_percent > 50:
                    ax1.text(0.5, 0.9, f"{zero_percent:.1f}% of values are zero", 
                            transform=ax1.transAxes, ha='center', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))
                    
                # Add inset axis for non-zero values
                non_zero = data[data > 0]
                if len(non_zero) > 0:
                    ax_inset = ax1.inset_axes([0.5, 0.5, 0.45, 0.45])
                    sns.histplot(data=non_zero, ax=ax_inset, bins=20, kde=True, 
                                color='green', alpha=0.6)
                    ax_inset.set_title("Non-zero values only", fontsize=10)
                    ax_inset.tick_params(labelsize=8)
                    
            ax1.set_title("Raw sheet_fraction", fontsize=12, pad=10)
            ax1.set_xlabel("Sheet Fraction", fontsize=10)
            ax1.set_ylabel("Count", fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No sheet_fraction data available', 
                    ha='center', va='center', fontsize=12)
        
        # Plot ML-ready data
        ax2 = fig.add_subplot(gs[0, 1])
        if 'sheet_fraction' in ml_data.columns:
            data = ml_data['sheet_fraction'].dropna()
            
            if data.empty:
                ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                sns.histplot(data=data, ax=ax2, bins=30, kde=True, color='red', alpha=0.6)
                
                # Add annotations for the 3 tiers
                params = self.threshold_params
                zero_min, zero_max = params['zero_sheet_range']
                minimal_min, minimal_max = params['minimal_sheet_range']
                significant_min, significant_max = params['significant_sheet_range']
                
                # Add shaded regions to indicate tiers
                ax2.axvspan(zero_min, zero_max, alpha=0.2, color='blue', label='Zero sheet')
                ax2.axvspan(minimal_min, minimal_max, alpha=0.2, color='green', label='Minimal sheet')
                ax2.axvspan(significant_min, significant_max, alpha=0.2, color='red', label='Significant sheet')
                ax2.legend(loc='upper right', fontsize=9)
                
                # Add annotation about transformation
                ax2.text(0.95, 0.95, "3-tier transformation", 
                       transform=ax2.transAxes, ha='right', va='top', 
                       fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
                
            ax2.set_title("ML-ready sheet_fraction", fontsize=12, pad=10)
            ax2.set_xlabel("Transformed Sheet Fraction", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
            ax2.set_xlim(-0.05, 1.05)  # Fixed range
        else:
            ax2.text(0.5, 0.5, 'No ML sheet_fraction data available', 
                    ha='center', va='center', fontsize=12)
        
        # Add transformation visualization diagram
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_transformation_diagram(ax3)
        
        plt.suptitle("Sheet Content Distributions with 3-Tier Transformation", fontsize=14)
        plt.tight_layout()
        return fig
    
    def _plot_transformation_diagram(self, ax):
        """Add a diagram illustrating the 3-tier transformation"""
        params = self.threshold_params
        
        # Get threshold values and ranges
        has_sheet = params['has_sheet_threshold']
        significant_sheet = params['significant_sheet_threshold']
        
        zero_min, zero_max = params['zero_sheet_range']
        minimal_min, minimal_max = params['minimal_sheet_range']
        significant_min, significant_max = params['significant_sheet_range']
        
        # Create x values for input domain (original scale)
        import numpy as np
        x_values = np.concatenate([
            np.linspace(0, has_sheet, 50),
            np.linspace(has_sheet, significant_sheet, 100),
            np.linspace(significant_sheet, 1.0, 50)
        ])
        
        # Calculate transformed values
        y_values = [self._transform_sheet_fraction(x) for x in x_values]
        
        # Plot the transformation function
        ax.plot(x_values, y_values, 'b-', linewidth=2)
        
        # Add threshold lines
        ax.axvline(x=has_sheet, color='g', linestyle='--', alpha=0.7, 
                  label=f'Has sheet threshold ({has_sheet:.4f})')
        ax.axvline(x=significant_sheet, color='r', linestyle='--', alpha=0.7,
                  label=f'Significant sheet threshold ({significant_sheet:.2f})')
        
        # Add range lines
        ax.axhline(y=zero_max, color='blue', linestyle=':', alpha=0.5)
        ax.axhline(y=minimal_min, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=minimal_max, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=significant_min, color='red', linestyle=':', alpha=0.5)
        
        # Shade the regions
        ax.fill_between(x_values, 0, y_values, alpha=0.1, color='blue')
        
        # Add labels
        ax.set_xlabel('Original sheet fraction', fontsize=10)
        ax.set_ylabel('Transformed value', fontsize=10)
        ax.set_title('Transformation Function', fontsize=12)
        ax.legend(loc='upper center', fontsize=9)
        
        # Set limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Add grid
        ax.grid(alpha=0.3)
            
    
class TurnContentFeature(BaseFeature, DSSPFeatureMixin):
    """
    Computes the fraction of residues in turn conformations.
    DSSP code considered: 'T'.
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['turn_fraction']
        self.ml_columns = ['turn_fraction']
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'turn_fraction': 0.0}
        turn_count = sum(1 for entry in dssp_data if entry[2] == 'T')
        fraction = turn_count / len(dssp_data)
        raw_data = {'turn_fraction': fraction}
        self.raw_data = raw_data
        return raw_data
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if self.raw_data:
            ml_data = self.raw_data
            self.ml_data = ml_data
            return ml_data
          
        else:
            ml_data = self.calculate_raw(pdb_path)
            self.ml_data = ml_data
            return ml_data
            
    
class CoilContentFeature(BaseFeature, DSSPFeatureMixin):
    """
    Computes the fraction of residues in coil (or loop) regions.
    Defined as residues not assigned to helix ('H','G','I'), sheet ('E'), or turn ('T').
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['coil_fraction']
        self.ml_columns = ['coil_fraction']
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'coil_fraction': 0.0}
        structured_codes = {'H', 'G', 'I', 'E', 'T'}
        structured_count = sum(1 for entry in dssp_data if entry[2] in structured_codes)
        coil_count = len(dssp_data) - structured_count
        fraction = coil_count / len(dssp_data)
        raw_data = {'coil_fraction': fraction}
        self.raw_data = raw_data
        return raw_data
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if self.raw_data:
            ml_data = self.raw_data
            self.ml_data = ml_data
            return ml_data
            
        else:
            ml_data = self.calculate_raw(pdb_path)
            self.ml_data = ml_data
            return ml_data
           
    
class SolventAccessibilityFeature(BaseFeature, DSSPFeatureMixin):
    """
    Computes solvent accessibility metrics:
      - Average relative solvent accessibility (RSA)
      - Standard deviation of RSA
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['avg_rsa', 'std_rsa']
        self.ml_columns = ['avg_rsa', 'std_rsa']
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'avg_rsa': 0.0, 'std_rsa': 0.0}
        rsa_values = [entry[3] for entry in dssp_data if entry[3] is not None]
        if not rsa_values:
            return {'avg_rsa': 0.0, 'std_rsa': 0.0}
        avg_rsa = np.mean(rsa_values)
        std_rsa = np.std(rsa_values)
        raw_data = {'avg_rsa': avg_rsa, 'std_rsa': std_rsa}
        self.raw_data = raw_data
        return raw_data
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if self.raw_data:
            ml_data = self.raw_data
            self.ml_data = ml_data
            return ml_data
        else:
            ml_data = self.calculate_raw(pdb_path)
            self.ml_data = ml_data
            return ml_data
            

class ResidueExposureFeature(BaseFeature, DSSPFeatureMixin):
    """
    Calculates and transforms residue exposure states.
    
    This feature measures the solvent exposure of residues, dividing them
    into buried (highly inaccessible) and exposed (highly accessible) categories.
    
    ML-ready features:
    - buried_fraction_transformed: Drastically transformed fraction of buried residues
      using a multi-tier approach to distinguish zero from non-zero values
    - exposed_fraction_transformed: Transformed measure of exposed residues
      that spreads out the extreme values for better ML performance
    """
    def __init__(self):
        super().__init__()
        # Thresholds for classifying residues
        self.buried_threshold = 0.2
        self.exposed_threshold = 0.5
        
        self.raw_columns = ['buried_count', 'exposed_count', 'total_residues']
        self.ml_columns = ['buried_fraction_transformed', 'exposed_fraction_transformed']
        
        # Multi-tier transformation parameters for buried_fraction
        self.buried_transform_params = {
            'has_buried_threshold': 0.001,  # Threshold to distinguish zero from non-zero
            'significant_buried_threshold': 0.1,  # Threshold for significant buried content
            
            # Output value ranges for the three tiers
            'zero_buried_range': (0.0, 0.2),          # Range for zero buried content
            'minimal_buried_range': (0.4, 0.6),       # Range for minimal buried content
            'significant_buried_range': (0.8, 1.0),   # Range for significant buried content
            
            # Spread factors within each range
            'minimal_spread_factor': 3.0,
            'significant_spread_factor': 2.0
        }
        
        # Special transformation for exposed_fraction which tends to be at extremes
        self.exposed_transform_params = {
            # Sigmoid transformation parameters
            'center': 0.7,      # Center point of the sigmoid (where output = 0.5)
            'steepness': 10.0,  # Controls how sharp the transition is
            
            # Apply reverse sigmoid for values above threshold
            'high_threshold': 0.8,
            'reverse_high': True,
            
            # For values near 1.0, map to upper range
            'extreme_threshold': 0.95,
            'extreme_range': (0.8, 1.0)
        }
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        """
        Calculate raw counts of buried and exposed residues.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Dictionary with raw counts
        """
        dssp_data = self._get_dssp_data(pdb_path)
        if not dssp_data:
            return {'buried_count': 0, 'exposed_count': 0, 'total_residues': 0}
            
        rsa_values = [entry[3] for entry in dssp_data if entry[3] is not None]
        if not rsa_values:
            return {'buried_count': 0, 'exposed_count': 0, 'total_residues': 0}
            
        buried = sum(1 for rsa in rsa_values if rsa < self.buried_threshold)
        exposed = sum(1 for rsa in rsa_values if rsa > self.exposed_threshold)
        raw_data = {
            'buried_count': buried, 
            'exposed_count': exposed, 
            'total_residues': len(rsa_values)
        }
        self.raw_data = raw_data
        return raw_data
    
    def _transform_buried_fraction(self, value: float) -> float:
        """
        Apply a 3-tier transformation to buried fraction:
        1. Zero buried content (0) → mapped to lower range (e.g., 0.0-0.2)
        2. Minimal buried content (>0 but below threshold) → mapped to middle range (e.g., 0.4-0.6)
        3. Significant buried content (≥threshold) → mapped to upper range (e.g., 0.8-1.0)
        
        Within each tier, values are spread according to their magnitude.
        
        Args:
            value: Raw buried fraction (between 0 and 1)
            
        Returns:
            Transformed value with 3-tier distribution
        """
        params = self.buried_transform_params
        
        # Get threshold values
        has_buried = params['has_buried_threshold']
        significant_buried = params['significant_buried_threshold']
        
        # Get range values
        zero_min, zero_max = params['zero_buried_range']
        minimal_min, minimal_max = params['minimal_buried_range']
        significant_min, significant_max = params['significant_buried_range']
        
        # Get spread factors
        minimal_spread = params['minimal_spread_factor']
        significant_spread = params['significant_spread_factor']
        
        # Apply tier-based transformation
        if value <= 0.0:
            # Tier 1: Zero buried content
            return zero_min
        elif value < has_buried:
            # Still essentially zero (numerical noise)
            # Map to the bottom of the zero range
            return zero_min + (zero_max - zero_min) * (value / has_buried)
        elif value < significant_buried:
            # Tier 2: Minimal buried content
            # Apply aggressive transformation within this range
            
            # Normalize to [0,1] within the minimal range
            normalized = (value - has_buried) / (significant_buried - has_buried)
            
            # Apply power transformation to spread small values
            import numpy as np
            spread = np.power(normalized, 1.0 / minimal_spread)
            
            # Map to the minimal buried range
            return minimal_min + (minimal_max - minimal_min) * spread
        else:
            # Tier 3: Significant buried content
            # Apply less aggressive transformation
            
            # Normalize to [0,1] within the significant range (capped at 1.0)
            normalized = min(1.0, (value - significant_buried) / (1.0 - significant_buried))
            
            # Apply milder power transformation
            import numpy as np
            spread = np.power(normalized, 1.0 / significant_spread)
            
            # Map to the significant buried range
            return significant_min + (significant_max - significant_min) * spread
    
    def _transform_exposed_fraction(self, value: float) -> float:
        """
        Apply special transformation for exposed fraction.
        
        This handles the bimodal distribution with concentrations at 0 and 1 by:
        1. Using sigmoid transformation for most values to create a more even spread
        2. Special handling for values near 1.0 to highlight this important region
        
        Args:
            value: Raw exposed fraction (between 0 and 1)
            
        Returns:
            Transformed value with better distribution
        """
        import numpy as np
        params = self.exposed_transform_params
        
        # Handle zeros separately
        if value <= 0.0:
            return 0.0
            
        # Special handling for extreme high values
        if value >= params['extreme_threshold']:
            # Map the range [extreme_threshold, 1.0] to [extreme_range_min, extreme_range_max]
            extreme_min, extreme_max = params['extreme_range']
            normalized = (value - params['extreme_threshold']) / (1.0 - params['extreme_threshold'])
            return extreme_min + normalized * (extreme_max - extreme_min)
            
        # For high values, use reverse sigmoid if specified
        if params['reverse_high'] and value >= params['high_threshold']:
            # Normalize to [0,1] from high_threshold to extreme_threshold
            normalized = (value - params['high_threshold']) / (params['extreme_threshold'] - params['high_threshold'])
            
            # Apply reverse sigmoid
            center = 0.5  # Center of normalized range
            steepness = params['steepness']
            sigmoid = 1.0 - (1.0 / (1.0 + np.exp(-steepness * (normalized - center))))
            
            # Map to upper-middle range
            return 0.6 + sigmoid * 0.2
            
        # Standard sigmoid for the rest
        center = params['center']
        steepness = params['steepness']
        
        # Apply sigmoid transformation: 1/(1+exp(-steepness*(x-center)))
        sigmoid = 1.0 / (1.0 + np.exp(-steepness * (value - center)))
        
        # Scale to range [0.0, 0.6] for mid-to-low values
        return sigmoid * 0.6
            
    def _calculate_weighted_rsa_score_for_exposedness(self, rsa_values: List[float]) -> float:
        """
        Calculates a weighted score based on RSA bins for 'exposedness'.
        This score will be the new value for the 'exposed_fraction_transformed' ML feature.
        """
        if not rsa_values: # Handle case with no RSA values (e.g., DSSP failed)
            return 0.0

        # Define RSA bins and weights specifically for "exposedness"
        # Bins: (lower_bound, exclusive_upper_bound, weight)
        # Last bin's upper bound will be treated as inclusive of 1.0.
        rsa_bins_weights = [
            (0.0, 0.05, 0.0),    # Very Buried (effectively weight 0 for exposedness)
            (0.05, 0.2, 0.1),   # Buried (small contribution to being exposed)
            (0.2, 0.5, 0.4),    # Intermediate/Partially Exposed
            (0.5, 0.8, 0.8),    # Exposed
            (0.8, 1.01, 1.0)    # Highly Exposed (using 1.01 to ensure RSA=1.0 is captured by < high)
        ]

        binned_counts = np.zeros(len(rsa_bins_weights), dtype=int)
        valid_rsa_count = 0

        for rsa_val in rsa_values:
            if rsa_val is None:  # Skip if DSSP couldn't assign RSA
                continue
            valid_rsa_count += 1
            assigned_to_bin = False
            for i, (low, high, _) in enumerate(rsa_bins_weights):
                if low <= rsa_val < high:
                    binned_counts[i] += 1
                    assigned_to_bin = True
                    break
            # This check ensures RSA = 1.0 (if bins are like [0.8, 1.0) ) is caught by the last bin.
            # The provided bins [0.8, 1.01) handle this correctly with "< high".
            # If rsa_val == 1.0 and last bin is [0.8, 1.0), it wouldn't be caught.
            # The current [0.8, 1.01) is fine.

        if valid_rsa_count == 0:
            return 0.0

        weighted_score = 0.0
        for i, (_, _, weight) in enumerate(rsa_bins_weights):
            fraction_in_bin = binned_counts[i] / valid_rsa_count
            weighted_score += fraction_in_bin * weight
        
        # Ensure score is robustly within [0,1] despite any floating point nuances
        return np.clip(weighted_score, 0.0, 1.0)

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        """
        Calculate ML-ready features.
        'exposed_fraction_transformed' uses the new weighted RSA score.
        'buried_fraction_transformed' uses its original logic with raw counts.
        """
        # Step 1: Get raw counts for 'buried_fraction_transformed'
        # self.calculate_raw() populates self.raw_data which contains 'buried_count' and 'total_residues'
        # based on the original thresholds (self.buried_threshold = 0.2).
        if hasattr(self, 'raw_data') and self.raw_data:
            raw_counts_data = self.raw_data
        else:
            raw_counts_data = self.calculate_raw(pdb_path) 
            
        total_residues_for_buried_calc = raw_counts_data['total_residues']

        if total_residues_for_buried_calc == 0:
            raw_buried_fraction = 0.0
        else:
            raw_buried_fraction = raw_counts_data['buried_count'] / total_residues_for_buried_calc
        
        ml_buried_transformed = self._transform_buried_fraction(raw_buried_fraction)

        # Step 2: Get RSA values for the new 'exposed_fraction_transformed'
        # This requires calling _get_dssp_data again.
        # Ideally, _get_dssp_data could be cached or rsa_values stored in self.raw_data by calculate_raw.
        # For this change, we'll re-call it to keep the logic for the new feature clear.
        dssp_output = self._get_dssp_data(pdb_path) # from DSSPFeatureMixin
        rsa_values_list = [entry[3] for entry in dssp_output if entry[3] is not None]
        
        ml_exposed_transformed = self._calculate_weighted_rsa_score_for_exposedness(rsa_values_list)
            
        ml_data = {
            'buried_fraction_transformed': ml_buried_transformed,      # Original calculation method
            'exposed_fraction_transformed': ml_exposed_transformed    # New robust calculation method
        }
        self.ml_data = ml_data
        return ml_data
    
    def inverse_transform_buried(self, transformed_value: float) -> float:
        """
        Convert a transformed buried_fraction back to its original scale.
        
        Args:
            transformed_value: The transformed value
            
        Returns:
            Original buried fraction
        """
        params = self.buried_transform_params
        
        # Get threshold values
        has_buried = params['has_buried_threshold']
        significant_buried = params['significant_buried_threshold']
        
        # Get range values
        zero_min, zero_max = params['zero_buried_range']
        minimal_min, minimal_max = params['minimal_buried_range']
        significant_min, significant_max = params['significant_buried_range']
        
        # Get spread factors
        minimal_spread = params['minimal_spread_factor']
        significant_spread = params['significant_spread_factor']
        
        # Determine which tier the transformed value falls into
        if transformed_value <= zero_max:
            # Tier 1: Zero buried content
            if transformed_value <= zero_min:
                return 0.0
            else:
                # Reverse the mapping within zero range
                normalized = (transformed_value - zero_min) / (zero_max - zero_min)
                return normalized * has_buried
                
        elif transformed_value <= minimal_max:
            # Tier 2: Minimal buried content
            # Normalize within minimal range
            normalized = (transformed_value - minimal_min) / (minimal_max - minimal_min)
            
            # Reverse the power transformation
            import numpy as np
            unspread = np.power(normalized, minimal_spread)
            
            # Map back to original scale
            return has_buried + unspread * (significant_buried - has_buried)
            
        else:
            # Tier 3: Significant buried content
            # Normalize within significant range
            normalized = (transformed_value - significant_min) / (significant_max - significant_min)
            
            # Reverse the power transformation
            import numpy as np
            unspread = np.power(normalized, significant_spread)
            
            # Map back to original scale
            return significant_buried + unspread * (1.0 - significant_buried)
    
    def inverse_transform_exposed(self, transformed_value: float) -> float:
        """
        Convert a transformed exposed_fraction back to its original scale.
        
        Args:
            transformed_value: The transformed value
            
        Returns:
            Original exposed fraction
        """
        import numpy as np
        params = self.exposed_transform_params
        
        # Handle zero value
        if transformed_value <= 0.0:
            return 0.0
            
        extreme_min, extreme_max = params['extreme_range']
        
        # Handle extreme high values region
        if transformed_value >= extreme_min:
            # Normalize within extreme range
            normalized = (transformed_value - extreme_min) / (extreme_max - extreme_min)
            # Map back to original scale
            return params['extreme_threshold'] + normalized * (1.0 - params['extreme_threshold'])
            
        # Handle high values that used reverse sigmoid
        if transformed_value > 0.6 and transformed_value < extreme_min:
            # Normalize within upper-middle range
            normalized = (transformed_value - 0.6) / 0.2
            
            # Reverse the reverse sigmoid
            center = 0.5
            steepness = params['steepness']
            
            # Solve for x in: normalized = 1 - 1/(1+exp(-steepness*(x-center)))
            unrev_sigmoid = center - np.log(1/(1-normalized) - 1) / steepness
            
            # Map back to original high range
            return params['high_threshold'] + unrev_sigmoid * (params['extreme_threshold'] - params['high_threshold'])
            
        # Handle standard sigmoid region (values <= 0.6)
        if transformed_value <= 0.6:
            # Normalize to [0,1]
            normalized = transformed_value / 0.6
            
            # Reverse sigmoid: solve for x in normalized = 1/(1+exp(-steepness*(x-center)))
            center = params['center']
            steepness = params['steepness']
            
            return center - np.log(1/normalized - 1) / steepness
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Create custom plots for ResidueExposureFeature distributions.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        # Create a figure with multiple subplots arranged in a 2x3 grid
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Plot raw counts
        for idx, col in enumerate(['buried_count', 'exposed_count', 'total_residues']):
            row = idx // 2
            col_idx = idx % 2
            ax = fig.add_subplot(gs[row, col_idx])
            
            if col in raw_data.columns:
                data = raw_data[col].dropna()
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                else:
                    sns.histplot(data, ax=ax, bins=30, kde=True, color='blue', alpha=0.6)
                    
                ax.set_title(f"Raw {col}")
            else:
                ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center')
                ax.set_title(f"Raw {col}")
        
        # Plot raw fractions (for comparison)
        if all(col in raw_data.columns for col in ['buried_count', 'exposed_count', 'total_residues']):
            # Calculate raw fractions
            total = raw_data['total_residues']
            valid_rows = (total > 0)
            buried_fraction = raw_data.loc[valid_rows, 'buried_count'] / total[valid_rows]
            exposed_fraction = raw_data.loc[valid_rows, 'exposed_count'] / total[valid_rows]
            
            # Plot buried fraction
            ax_raw_buried = fig.add_subplot(gs[1, 0])
            if not buried_fraction.empty:
                sns.histplot(buried_fraction, ax=ax_raw_buried, bins=30, 
                           kde=True, color='lightblue', alpha=0.6)
                ax_raw_buried.set_title("Raw buried_fraction")
                
                # Add annotation about zero values
                zero_percent = (buried_fraction == 0).mean() * 100
                if zero_percent > 50:
                    ax_raw_buried.text(0.5, 0.9, f"{zero_percent:.1f}% of values are zero", 
                                      transform=ax_raw_buried.transAxes, ha='center', fontsize=10,
                                      bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax_raw_buried.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_raw_buried.set_title("Raw buried_fraction")
            
            # Plot exposed fraction
            ax_raw_exposed = fig.add_subplot(gs[1, 1])
            if not exposed_fraction.empty:
                sns.histplot(exposed_fraction, ax=ax_raw_exposed, bins=30, 
                           kde=True, color='lightblue', alpha=0.6)
                ax_raw_exposed.set_title("Raw exposed_fraction")
                
                # Add annotation about 100% values
                high_percent = (exposed_fraction > 0.95).mean() * 100
                if high_percent > 20:
                    ax_raw_exposed.text(0.5, 0.9, f"{high_percent:.1f}% of values above 0.95", 
                                       transform=ax_raw_exposed.transAxes, ha='center', fontsize=10,
                                       bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax_raw_exposed.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_raw_exposed.set_title("Raw exposed_fraction")
        
        # Plot ML-ready features
        # Plot transformed buried fraction
        if 'buried_fraction_transformed' in ml_data.columns:
            ax_ml_buried = fig.add_subplot(gs[2, 0])
            
            ml_buried = ml_data['buried_fraction_transformed'].dropna()
            
            if not ml_buried.empty:
                sns.histplot(ml_buried, ax=ax_ml_buried, bins=30, kde=True, color='red', alpha=0.6)
                ax_ml_buried.set_title("ML-ready buried_fraction_transformed")
                
                # Add annotations for the 3 tiers
                params = self.buried_transform_params
                zero_min, zero_max = params['zero_buried_range']
                minimal_min, minimal_max = params['minimal_buried_range']
                significant_min, significant_max = params['significant_buried_range']
                
                # Add shaded regions to indicate tiers
                ax_ml_buried.axvspan(zero_min, zero_max, alpha=0.2, color='blue', label='Zero buried')
                ax_ml_buried.axvspan(minimal_min, minimal_max, alpha=0.2, color='green', label='Minimal buried')
                ax_ml_buried.axvspan(significant_min, significant_max, alpha=0.2, color='red', label='Significant buried')
                ax_ml_buried.legend(loc='upper right', fontsize=9)
                ax_ml_buried.set_xlim(-0.05, 1.05)
            else:
                ax_ml_buried.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_ml_buried.set_title("ML-ready buried_fraction_transformed")
        
        # Plot transformed exposed fraction
        if 'exposed_fraction_transformed' in ml_data.columns:
            ax_ml_exposed = fig.add_subplot(gs[2, 1])
            
            ml_exposed = ml_data['exposed_fraction_transformed'].dropna()
            
            if not ml_exposed.empty:
                sns.histplot(ml_exposed, ax=ax_ml_exposed, bins=30, kde=True, color='red', alpha=0.6)
                ax_ml_exposed.set_title("ML-ready exposed_fraction_transformed")
                
                # Add shaded regions for the different transformation zones
                extreme_min, extreme_max = self.exposed_transform_params['extreme_range']
                ax_ml_exposed.axvspan(0.0, 0.6, alpha=0.2, color='blue', label='Sigmoid zone')
                ax_ml_exposed.axvspan(0.6, extreme_min, alpha=0.2, color='green', label='High zone (reverse sigmoid)')
                ax_ml_exposed.axvspan(extreme_min, extreme_max, alpha=0.2, color='red', label='Extreme high zone')
                ax_ml_exposed.legend(loc='upper right', fontsize=9)
                ax_ml_exposed.set_xlim(-0.05, 1.05)
            else:
                ax_ml_exposed.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax_ml_exposed.set_title("ML-ready exposed_fraction_transformed")
        
        plt.suptitle("Residue Exposure Distributions with Specialized Transformations", fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
        

if __name__ == '__main__':
    # Pass the path to a valid AlphaFold2/ColabFold .pdb file as the first argument.
    import sys
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else None
    if pdb_path is None:
        raise SystemExit("Usage: python DSSP_features.py <path/to/peptide.pdb>")
    
    features_to_test = [
        ("HelixContent", HelixContentFeature()),
        ("SheetContent", SheetContentFeature()),
        ("TurnContent", TurnContentFeature()),
        ("CoilContent", CoilContentFeature()),
        ("SolventAccessibility", SolventAccessibilityFeature()),
        ("ResidueExposure", ResidueExposureFeature()),
    ]
    
    for feat_name, feat in features_to_test:
        print(f"\nTesting {feat_name} Feature:")
        raw = feat.calculate_raw(pdb_path)
        ml_ready = feat.calculate_ml_ready(pdb_path)
        print("Raw output:", raw)
        print("ML-ready output:", ml_ready)