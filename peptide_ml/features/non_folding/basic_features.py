from typing import Dict, List, Any
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import logging
import traceback # Import traceback

from ..base_feature import BaseFeature

# Get logger instance. If Streamlit is running, it might use its own config.
# This ensures we have *a* logger.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChainLengthFeature(BaseFeature):
    """Chain length feature implementation"""
    
    def __init__(self):
        super().__init__()
        self.raw_columns = ['chain_length']
        self.ml_columns = ['chain_length', 'chain_length_log']
    
    def calculate_raw(self, sequence: str) -> Dict[str, int]:
        """
        Calculate raw chain length.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with raw chain length
        """
        return {'chain_length': len(sequence)}
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version with log transform.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with normal and log-transformed length
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            length = self.raw_data['chain_length']
        else:
            raw_data = self.calculate_raw(sequence)
            length = raw_data['chain_length']
            self.raw_data = raw_data
            
        ml_data = {
            'chain_length': float(length),
            'chain_length_log': float(np.log1p(length))
        }
        self.ml_data = ml_data
        return ml_data
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """Custom plotting for chain length with detailed binning"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Raw length distribution with smaller bins
        sns.histplot(data=raw_data['chain_length'], ax=ax1, 
                    bins=range(0, int(raw_data['chain_length'].max()) + 2, 1),
                    color='blue')
        ax1.set_title('Raw Chain Length (1-unit bins)')
        
        # ML-ready length
        sns.histplot(data=ml_data['chain_length'], ax=ax2,
                    bins=range(0, int(ml_data['chain_length'].max()) + 2, 1),
                    color='red')
        ax2.set_title('ML-ready Chain Length (1-unit bins)')
        
        # Log-transformed length
        sns.histplot(data=ml_data['chain_length_log'], ax=ax3, 
                    bins=50, color='red')
        ax3.set_title('Log-transformed Chain Length')
        
        # Add actual value counts as text
        
        
        plt.tight_layout()
        return fig

class MolecularWeightFeature(BaseFeature):
    """Molecular weight feature implementation"""
    
    def __init__(self):
        super().__init__()
        self.raw_columns = ['molecular_weight']
        self.ml_columns = ['molecular_weight', 'molecular_weight_per_residue']
    
    def calculate_raw(self, sequence: str) -> Dict[str, float]:
        """
        Calculate raw molecular weight.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with raw molecular weight
        """
        protein = ProteinAnalysis(sequence)
        return {'molecular_weight': protein.molecular_weight()}
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version with scaling.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with scaled molecular weight
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            mw = self.raw_data['molecular_weight']
        else:
            raw_data = self.calculate_raw(sequence)
            mw = raw_data['molecular_weight']
            self.raw_data = raw_data
            
        ml_data = {
            'molecular_weight': mw,
            'molecular_weight_per_residue': mw / len(sequence)
        }
        self.ml_data = ml_data
        return ml_data

class AACompositionFeature(BaseFeature):
    """
    Amino acid composition feature implementation.
    Raw returns counts, ML returns fractions.
    Includes detailed debugging.
    """

    def __init__(self):
        super().__init__()
        self.aa_letters = 'ACDEFGHIKLMNPQRSTVWY'
        # Raw columns will now be counts
        self.raw_columns = [f'aa_count_{aa}' for aa in self.aa_letters]
        # ML columns are fractions
        self.ml_columns = [f'aa_fraction_{aa}' for aa in self.aa_letters]
        self.raw_data = None
        self.ml_data = None

    def calculate_raw(self, sequence: str) -> Dict[str, int]:
        """
        Calculate raw amino acid counts with extensive debugging.
        """
        logger.info(f"AA_Comp Raw START: Input sequence='{sequence}' (Type: {type(sequence)}, Length: {len(sequence)})")
        # Default dictionary with all counts as 0
        aa_counts_result = {f'aa_count_{aa}': 0 for aa in self.aa_letters}

        if not sequence or not isinstance(sequence, str):
            logger.error("AA_Comp Raw ERROR: Invalid sequence input (empty or not string). Returning zero counts.")
            self.raw_data = aa_counts_result
            return aa_counts_result

        # Clean sequence just in case (ensure only valid AAs)
        cleaned_sequence = "".join(c for c in sequence.upper() if c in self.aa_letters)
        if len(cleaned_sequence) != len(sequence):
             logger.warning(f"AA_Comp Raw WARNING: Sequence contained invalid characters. Original len={len(sequence)}, Cleaned len={len(cleaned_sequence)}")
             if not cleaned_sequence:
                  logger.error("AA_Comp Raw ERROR: Sequence empty after cleaning. Returning zero counts.")
                  self.raw_data = aa_counts_result
                  return aa_counts_result
             # Use the cleaned sequence for analysis
             sequence_to_analyze = cleaned_sequence
        else:
             sequence_to_analyze = sequence

        logger.info(f"AA_Comp Raw: Analyzing sequence: '{sequence_to_analyze}'")

        try:
            # Ensure it's a string for ProteinAnalysis
            protein = ProteinAnalysis(str(sequence_to_analyze))

            # --- DEBUG: Call methods individually ---
            logger.info("AA_Comp Raw DEBUG: Calling protein.count_amino_acids()...")
            counts_dict = protein.count_amino_acids() # Should return raw counts
            logger.info(f"AA_Comp Raw DEBUG: protein.count_amino_acids() returned: {counts_dict} (Type: {type(counts_dict)})")

            logger.info("AA_Comp Raw DEBUG: Calling protein.get_amino_acids_percent()...")
            percent_dict = protein.get_amino_acids_percent() # Returns percentages
            logger.info(f"AA_Comp Raw DEBUG: protein.get_amino_acids_percent() returned: {percent_dict} (Type: {type(percent_dict)})")
            # --- END DEBUG ---

            # Use the COUNTS dictionary, not the percentages
            if not isinstance(counts_dict, dict):
                 logger.error(f"AA_Comp Raw ERROR: count_amino_acids did not return a dict. Got {type(counts_dict)}. Returning zeros.")
                 self.raw_data = {f'aa_count_{aa}': 0 for aa in self.aa_letters}
                 return self.raw_data

            # Update the default dictionary with actual counts
            for aa, count in counts_dict.items():
                if aa in self.aa_letters: # Ensure we only use standard AAs
                    aa_counts_result[f'aa_count_{aa}'] = int(count) # Ensure integer count
                else:
                    logger.warning(f"AA_Comp Raw WARNING: Got unexpected character '{aa}' from count_amino_acids.")

            logger.info(f"AA_Comp Raw SUCCESS: Calculated counts: {aa_counts_result}")
            self.raw_data = aa_counts_result # Store results
            return aa_counts_result

        except Exception as e:
            logger.error(f"AA_Comp Raw ERROR: Exception during ProteinAnalysis for '{sequence_to_analyze}': {e}")
            logger.error(traceback.format_exc()) # Log full traceback
            # Return zero counts on error
            self.raw_data = {f'aa_count_{aa}': 0 for aa in self.aa_letters}
            return self.raw_data

    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version with fractions.
        """
        logger.info(f"AA_Comp ML START: Input sequence='{sequence[:10]}...' (Len: {len(sequence)})")

        # Use stored raw_data if available, otherwise calculate it
        if hasattr(self, 'raw_data') and self.raw_data is not None:
            aa_counts_raw = self.raw_data
            logger.info("AA_Comp ML: Using pre-calculated raw data.")
        else:
            logger.warning("AA_Comp ML: Raw data not pre-calculated, calling calculate_raw. This might indicate an issue.")
            aa_counts_raw = self.calculate_raw(sequence) # This also sets self.raw_data

        logger.info(f"AA_Comp ML: Input raw counts dict: {aa_counts_raw}")
        length = len(sequence) # Use original sequence length for fraction calculation
        ml_data = {}

        if length > 0:
            for aa in self.aa_letters:
                count_key = f'aa_count_{aa}'
                # Get count, default to 0 if key somehow missing
                count = aa_counts_raw.get(count_key, 0)

                # Check if count is valid (numeric, not NaN)
                if pd.isna(count) or not isinstance(count, (int, float, np.number)):
                    logger.error(f"AA_Comp ML ERROR: Invalid count found for {count_key}: {count}. Setting fraction to NaN.")
                    ml_data[f'aa_fraction_{aa}'] = np.nan # Set to NaN if count is bad
                else:
                    ml_data[f'aa_fraction_{aa}'] = float(count) / length # Calculate fraction

            # Check if ALL fractions are NaN (indicates total failure)
            if all(pd.isna(v) for v in ml_data.values()):
                 logger.error("AA_Comp ML ERROR: All calculated fractions are NaN.")

        else:
            logger.error("AA_Comp ML ERROR: Zero length sequence received. Setting all fractions to NaN.")
            # Return NaNs because fractions are undefined for zero length
            ml_data = {f'aa_fraction_{aa}': np.nan for aa in self.aa_letters}

        logger.info(f"AA_Comp ML FINISHED: Final ml_data (first 5): { {k: ml_data[k] for k in list(ml_data)[:5]} }")
        self.ml_data = ml_data
        return ml_data

    # --- Keep plot_distributions as is, it should adapt based on new column names ---
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """Custom plotting for amino acid composition"""
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1])

        # Plot raw counts in top panel
        ax1 = plt.subplot(gs[0])
        raw_count_cols = [f'aa_count_{aa}' for aa in self.aa_letters if f'aa_count_{aa}' in raw_data.columns]
        if raw_count_cols:
            raw_data[raw_count_cols].boxplot(ax=ax1)
            ax1.set_title('Raw Amino Acid Counts')
            ax1.set_ylabel('Count')
        else:
             ax1.text(0.5, 0.5, "No raw count data", ha='center', va='center')
             ax1.set_title('Raw Amino Acid Counts')
        ax1.tick_params(axis='x', rotation=45)

        # Plot ML fractions in bottom panel
        ax2 = plt.subplot(gs[1])
        ml_frac_cols = [f'aa_fraction_{aa}' for aa in self.aa_letters if f'aa_fraction_{aa}' in ml_data.columns]
        if ml_frac_cols:
            ml_data[ml_frac_cols].boxplot(ax=ax2)
            ax2.set_title('ML-Ready Amino Acid Fractions')
            ax2.set_ylabel('Fraction')
        else:
             ax2.text(0.5, 0.5, "No ML fraction data", ha='center', va='center')
             ax2.set_title('ML-Ready Amino Acid Fractions')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

class DipeptideCompositionFeature(BaseFeature):
    """Dipeptide composition feature implementation"""
    
    def __init__(self):
        super().__init__()
        self.aa_groups = {
            'hydrophobic': set('AILMFWV'),
            'polar': set('STNQ'),
            'charged_pos': set('KR'),
            'charged_neg': set('DE'),
            'special': set('CYHGP')
        }
        self.dipeptides = [''.join(aa) for aa in product('ACDEFGHIKLMNPQRSTVWY', repeat=2)]
        self.raw_columns = self.dipeptides
        
        # Dynamically generate ML column names based on group combinations
        self.ml_columns = []
        for g1 in self.aa_groups:
            for g2 in self.aa_groups:
                self.ml_columns.append(f'dipep_{g1}_{g2}')
    
    def calculate_raw(self, sequence: str) -> Dict[str, int]:
        """
        Calculate raw dipeptide counts.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with counts of each dipeptide
        """
        dipeptides = [''.join(aa) for aa in product('ACDEFGHIKLMNPQRSTVWY', repeat=2)]
        return {dip: sequence.count(dip) for dip in dipeptides}
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version with grouped properties.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with fractions of grouped dipeptide types
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            # We don't actually need the raw data for this calculation
            pass
        else:
            raw_counts = self.calculate_raw(sequence)
            self.raw_data = raw_counts
            
        features = {}
        
        # Calculate group-based dipeptide frequencies
        for i in range(len(sequence)-1):
            aa1, aa2 = sequence[i:i+2]
            
            # Get groups for each amino acid
            aa1_groups = {group for group, aas in self.aa_groups.items() if aa1 in aas}
            aa2_groups = {group for group, aas in self.aa_groups.items() if aa2 in aas}
            
            # Update counts for each group combination
            for g1 in aa1_groups:
                for g2 in aa2_groups:
                    key = f'dipep_{g1}_{g2}'
                    features[key] = features.get(key, 0) + 1
        
        # Normalize by total dipeptides
        total = len(sequence) - 1
        if total > 0:
            ml_data = {k: v/total for k, v in features.items()}
        else:
            ml_data = {k: 0.0 for k in self.ml_columns}
        
        self.ml_data = ml_data
        return ml_data
        
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """Custom plotting for dipeptide composition"""
        fig = plt.figure(figsize=(24, 16))
        
        # Create separate subplots for raw and ML-ready features
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        
        # Plot raw features as a heatmap
        raw_means = raw_data.mean()
        raw_matrix = raw_means.values.reshape(20, 20)
        sns.heatmap(raw_matrix, ax=ax1, cmap='YlOrRd', 
                   xticklabels=list('ACDEFGHIKLMNPQRSTVWY'),
                   yticklabels=list('ACDEFGHIKLMNPQRSTVWY'))
        ax1.set_title('Raw Dipeptide Frequencies')
        
        # Plot ML features (grouped by properties)
        sns.barplot(x=ml_data.columns, y=ml_data.mean(), ax=ax2)
        ax2.set_title('ML-Ready Dipeptide Group Frequencies')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

class SequenceEntropyFeature(BaseFeature):
    """Sequence entropy feature implementation"""
    
    def __init__(self):
        super().__init__()
        self.raw_columns = ['sequence_entropy']
        self.ml_columns = ['sequence_entropy']
    
    def calculate_raw(self, sequence: str) -> Dict[str, float]:
        """
        Calculate raw sequence entropy.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with entropy value
        """
        protein = ProteinAnalysis(sequence)
        aa_frequencies = np.array(list(protein.get_amino_acids_percent().values())) / 100
        entropy = -np.sum(aa_frequencies * np.log2(aa_frequencies + 1e-10))
        return {'sequence_entropy': entropy}
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """
        Calculate ML-ready version (already scaled 0-4.32).
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with entropy value
        """
        if hasattr(self, 'raw_data') and self.raw_data:
            ml_data = self.raw_data
        else:
            ml_data = self.calculate_raw(sequence)
            self.raw_data = ml_data
            
        self.ml_data = ml_data
        return ml_data  # Already ML-ready

    def plot_distribution(self, sequences: List[str]) -> plt.Figure:
        """
        Create custom plot for entropy distribution with theoretical max line.
        
        Args:
            sequences: List of sequences to analyze
            
        Returns:
            matplotlib Figure object showing entropy distribution
        """
        data = self.calculate_bulk(sequences)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(data=data['sequence_entropy'], ax=ax)
        
        # Add theoretical maximum line
        ax.axvline(np.log2(20), color='r', linestyle='--',
                label='Theoretical max')
        
        # Set labels and title
        ax.set_title("Distribution of Sequence Entropy")
        ax.set_xlabel("Entropy (bits)")
        ax.set_ylabel("Count")
        ax.legend()
        
        return fig
