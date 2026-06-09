from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from ..base_feature import BaseFeature

class IsoelectricPointFeature(BaseFeature):
    """Calculate isoelectric point of peptide"""
    
    def __init__(self):
        super().__init__()
        self.raw_columns = ['isoelectric_point']
        self.ml_columns = ['isoelectric_point']
    
    def calculate_raw(self, sequence: str) -> Dict[str, float]:
        """
        Calculate isoelectric point using ProtParam.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with isoelectric point
        """
        protein = ProteinAnalysis(sequence)
        return {'isoelectric_point': protein.isoelectric_point()}
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Already well-scaled between 0-14"""
        if hasattr(self, 'raw_data') and self.raw_data:
            ml_data = self.raw_data
        else:
            ml_data = self.calculate_raw(sequence)
            self.raw_data = ml_data
        
        self.ml_data = ml_data
        return ml_data

class ChargeFeature(BaseFeature):
    """Calculate net charge at different pH values"""
    
    def __init__(self):
        super().__init__()
        # pKa values for amino acids
        self.pka_values = {
            'K': 10.5,  # Lysine
            'R': 12.5,  # Arginine
            'H': 6.0,   # Histidine
            'D': 3.9,   # Aspartic acid
            'E': 4.3,   # Glutamic acid
            'C': 8.3,   # Cysteine
            'Y': 10.1,  # Tyrosine
            'N_term': 8.0,  # N-terminus
            'C_term': 3.1   # C-terminus
        }
        self.raw_columns = ['charge_ph2', 'charge_ph7', 'charge_ph12']
        self.ml_columns = ['charge_ph2_per_residue', 'charge_ph7_per_residue', 'charge_ph12_per_residue']
    
    def _calculate_charge_at_ph(self, sequence: str, ph: float) -> float:
        """Calculate net charge at given pH"""
        charge = 0.0
        
        # Count charged residues
        for aa, pka in self.pka_values.items():
            if aa in ('N_term', 'C_term'):
                continue
            count = sequence.count(aa)
            if pka > ph:  # Basic residues
                charge += count
            else:  # Acidic residues
                charge -= count
        
        # Add terminal charges
        if self.pka_values['N_term'] > ph:
            charge += 1
        if self.pka_values['C_term'] > ph:
            charge -= 1
            
        return charge
    
    def calculate_raw(self, sequence: str) -> Dict[str, float]:
        """Calculate charge at pH 2, 7, and 12"""
        return {
            'charge_ph2': self._calculate_charge_at_ph(sequence, 2.0),
            'charge_ph7': self._calculate_charge_at_ph(sequence, 7.0),
            'charge_ph12': self._calculate_charge_at_ph(sequence, 12.0)
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Normalize charges by sequence length"""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw_charges = self.raw_data
        else:
            raw_charges = self.calculate_raw(sequence)
            self.raw_data = raw_charges
            
        length = len(sequence)
        ml_data = {
            f'{k}_per_residue': v/length 
            for k, v in raw_charges.items()
        }
        self.ml_data = ml_data
        return ml_data

class LocalChargeFeature(BaseFeature):
    """Calculate local charge distribution"""
    
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.charged_pos = set('KR')
        self.charged_neg = set('DE')
        self.raw_columns = ['charge_profile', 'max_local_charge', 'min_local_charge', 'charge_variation']
        self.ml_columns = ['max_local_charge', 'min_local_charge', 'charge_variation', 'charge_asymmetry']
    
    def _calculate_charge_profile(self, sequence: str) -> List[float]:
        """Calculate charge in sliding window"""
        profile = []
        half_window = self.window_size // 2
        
        for i in range(len(sequence)):
            start = max(0, i - half_window)
            end = min(len(sequence), i + half_window + 1)
            window = sequence[start:end]
            
            pos_count = sum(1 for aa in window if aa in self.charged_pos)
            neg_count = sum(1 for aa in window if aa in self.charged_neg)
            local_charge = (pos_count - neg_count) / len(window)
            profile.append(local_charge)
            
        return profile
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate charge profile"""
        profile = self._calculate_charge_profile(sequence)
        return {
            'charge_profile': profile,
            'max_local_charge': max(profile),
            'min_local_charge': min(profile),
            'charge_variation': np.std(profile)
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Extract ML-ready features from profile"""
        if hasattr(self, 'raw_data') and self.raw_data:
            profile = self.raw_data['charge_profile']
        else:
            raw_data = self.calculate_raw(sequence)
            profile = raw_data['charge_profile']
            self.raw_data = raw_data
        
        ml_data = {
            'max_local_charge': max(profile),
            'min_local_charge': min(profile),
            'charge_variation': np.std(profile),
            'charge_asymmetry': np.mean(profile[:len(profile)//2]) - 
                              np.mean(profile[len(profile)//2:])
        }
        self.ml_data = ml_data
        return ml_data
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Custom plotting for local charge features.
        
        Args:
            raw_data: DataFrame with raw charge profiles and metrics
            ml_data: DataFrame with ML-ready charge metrics
        
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot max local charge distribution
        sns.histplot(data=ml_data['max_local_charge'], ax=ax1, color='blue')
        ax1.set_title('Maximum Local Charge')
        ax1.set_xlabel('Charge')
        
        # Plot min local charge distribution
        sns.histplot(data=ml_data['min_local_charge'], ax=ax2, color='red')
        ax2.set_title('Minimum Local Charge')
        ax2.set_xlabel('Charge')
        
        # Plot charge variation
        sns.histplot(data=ml_data['charge_variation'], ax=ax3, color='green')
        ax3.set_title('Charge Variation')
        ax3.set_xlabel('Standard Deviation')
        
        # Plot charge asymmetry
        sns.histplot(data=ml_data['charge_asymmetry'], ax=ax4, color='purple')
        ax4.set_title('Charge Asymmetry')
        ax4.set_xlabel('N-term vs C-term Difference')
        
        plt.tight_layout()
        return fig

class HydrophobicityFeature(BaseFeature):
    """Calculate hydrophobicity using multiple scales"""
    
    def __init__(self):
        super().__init__()
        # Kyte-Doolittle scale
        self.kd_scale = {
            'A':  1.8, 'C':  2.5, 'D': -3.5, 'E': -3.5, 'F':  2.8,
            'G': -0.4, 'H': -3.2, 'I':  4.5, 'K': -3.9, 'L':  3.8,
            'M':  1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V':  4.2, 'W': -0.9, 'Y': -1.3
        }
        # Eisenberg scale
        self.eisenberg_scale = {
            'A':  0.62, 'C':  0.29, 'D': -0.90, 'E': -0.74, 'F':  1.19,
            'G':  0.48, 'H': -0.40, 'I':  1.38, 'K': -1.50, 'L':  1.06,
            'M':  0.64, 'N': -0.78, 'P':  0.12, 'Q': -0.85, 'R': -2.53,
            'S': -0.18, 'T': -0.05, 'V':  1.08, 'W':  0.81, 'Y':  0.26
        }
        self.raw_columns = ['kd_profile', 'eisenberg_profile', 'kd_mean', 'eisenberg_mean']
        self.ml_columns = ['kd_mean', 'kd_std', 'eisenberg_mean', 'eisenberg_std', 'hydrophobic_moment']
    
    def _calculate_hydrophobicity(self, sequence: str, scale: Dict[str, float]) -> List[float]:
        """Calculate hydrophobicity profile using given scale"""
        return [scale[aa] for aa in sequence]
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate hydrophobicity using multiple scales"""
        kd_profile = self._calculate_hydrophobicity(sequence, self.kd_scale)
        eisenberg_profile = self._calculate_hydrophobicity(sequence, self.eisenberg_scale)
        
        return {
            'kd_profile': kd_profile,
            'eisenberg_profile': eisenberg_profile,
            'kd_mean': np.mean(kd_profile),
            'eisenberg_mean': np.mean(eisenberg_profile)
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready hydrophobicity features"""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
            kd_profile = raw['kd_profile']
            eisenberg_profile = raw['eisenberg_profile']
        else:
            raw = self.calculate_raw(sequence)
            kd_profile = raw['kd_profile']
            eisenberg_profile = raw['eisenberg_profile']
            self.raw_data = raw
        
        ml_data = {
            'kd_mean': np.mean(kd_profile),
            'kd_std': np.std(kd_profile),
            'eisenberg_mean': np.mean(eisenberg_profile),
            'eisenberg_std': np.std(eisenberg_profile),
            'hydrophobic_moment': self._calculate_hydrophobic_moment(eisenberg_profile)
        }
        self.ml_data = ml_data
        return ml_data
    
    def _calculate_hydrophobic_moment(self, hydrophobicity_profile: List[float]) -> float:
        """Calculate hydrophobic moment (measure of amphipathicity)"""
        moment = 0
        for i, h in enumerate(hydrophobicity_profile):
            angle = i * 100  # Assuming alpha-helix (100° per residue)
            moment += h * np.exp(complex(0, np.radians(angle)))
        return abs(moment) / len(hydrophobicity_profile)

    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """Plot hydrophobicity feature distributions."""
        # Validate input data
        required_cols = ['kd_profile', 'eisenberg_profile']
        if not all(col in raw_data.columns for col in required_cols):
            warnings.warn("Missing required columns in raw_data")
            return plt.figure()

        # Prepare flattened profile data
        kd_flat = raw_data["kd_profile"].explode().dropna().astype(float)
        eis_flat = raw_data["eisenberg_profile"].explode().dropna().astype(float)
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot distributions
        # Row 1: Raw distributions
        sns.histplot(kd_flat, ax=axes[0, 0], color='blue', alpha=0.7)
        axes[0, 0].set_title("All Kyte-Doolittle values")
        
        sns.histplot(eis_flat, ax=axes[0, 1], color='orange', alpha=0.7)
        axes[0, 1].set_title("All Eisenberg values")
        
        if "kd_mean" in raw_data.columns:
            sns.histplot(raw_data["kd_mean"].dropna(), ax=axes[0, 2], color='blue')
        axes[0, 2].set_title("KD Mean (raw)")
        
        # Row 2: ML-ready features
        if "kd_mean" in ml_data.columns:
            sns.histplot(ml_data["kd_mean"].dropna(), ax=axes[1, 0], color='blue')
        axes[1, 0].set_title("KD Mean (ML)")
        
        if "kd_std" in ml_data.columns:
            sns.histplot(ml_data["kd_std"].dropna(), ax=axes[1, 1], color='green')
        axes[1, 1].set_title("KD Std Dev (ML)")
        
        if "eisenberg_mean" in ml_data.columns:
            sns.histplot(ml_data["eisenberg_mean"].dropna(), ax=axes[1, 2], color='orange')
        axes[1, 2].set_title("Eisenberg Mean (ML)")
        
        fig.set_constrained_layout(True)
        return fig
    
class AmphipathicityFeature(BaseFeature):
    """Calculate various measures of amphipathicity"""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.hydrophobic = set('AILMFWV')
        self.polar = set('STNQY')
        self.raw_columns = ['amphipathic_profile', 'max_amphipathicity', 'mean_amphipathicity']
        self.ml_columns = ['max_amphipathicity', 'mean_amphipathicity', 'amphipathicity_variation']
        
    def _calculate_amphipathic_profile(self, sequence: str) -> List[float]:
        """Calculate amphipathicity in sliding window"""
        profile = []
        half_window = self.window_size // 2
        
        for i in range(len(sequence)):
            start = max(0, i - half_window)
            end = min(len(sequence), i + half_window + 1)
            window = sequence[start:end]
            
            hydrophobic_count = sum(1 for aa in window if aa in self.hydrophobic)
            polar_count = sum(1 for aa in window if aa in self.polar)
            
            # Measure of segregation between hydrophobic and polar residues
            amphipathicity = abs(hydrophobic_count - polar_count) / len(window)
            profile.append(amphipathicity)
            
        return profile
    
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate amphipathicity profile"""
        profile = self._calculate_amphipathic_profile(sequence)
        return {
            'amphipathic_profile': profile,
            'max_amphipathicity': max(profile),
            'mean_amphipathicity': np.mean(profile)
        }
    
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready amphipathicity features"""
        if hasattr(self, 'raw_data') and self.raw_data:
            profile = self.raw_data['amphipathic_profile']
        else:
            raw_data = self.calculate_raw(sequence)
            profile = raw_data['amphipathic_profile']
            self.raw_data = raw_data
        
        ml_data = {
            'max_amphipathicity': max(profile),
            'mean_amphipathicity': np.mean(profile),
            'amphipathicity_variation': np.std(profile)
        }
        self.ml_data = ml_data
        return ml_data
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        # Flatten the list column
        profile_flat = raw_data["amphipathic_profile"].explode().dropna()
        profile_flat = profile_flat.astype(float)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Hist of all local amphipathicity values
        sns.histplot(profile_flat, ax=axes[0, 0], color='blue')
        axes[0, 0].set_title("All Local Amphipathicity Values")

        # Hist of raw max
        sns.histplot(raw_data["max_amphipathicity"].dropna(), ax=axes[0, 1], color='orange')
        axes[0, 1].set_title("Max Amphipathicity (raw)")

        # Hist of ML mean
        sns.histplot(ml_data["mean_amphipathicity"].dropna(), ax=axes[1, 0], color='green')
        axes[1, 0].set_title("Mean Amphipathicity (ML)")

        # Hist of ML variation
        sns.histplot(ml_data["amphipathicity_variation"].dropna(), ax=axes[1, 1], color='purple')
        axes[1, 1].set_title("Amphipathicity Variation (ML)")

        plt.tight_layout()
        return fig

class PredictedSolubilityFeature(BaseFeature):
    """
    Estimates solubility potential based on hydrophobicity (GRAVY) and charge.
    Lower GRAVY and higher absolute charge generally correlate with better solubility.
    """

    def __init__(self):
        super().__init__()
        # pKa values needed for charge calculation (copied from ChargeFeature)
        self.pka_values = {
            'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3,
            'C': 8.3, 'Y': 10.1, 'N_term': 8.0, 'C_term': 3.1
        }
        self.raw_columns = ['solubility_proxy_gravy', 'solubility_proxy_abs_charge']
        self.ml_columns = ['solubility_gravy', 'solubility_abs_charge_norm']

    def _calculate_charge_at_ph7(self, sequence: str) -> float:
        """Helper to calculate charge at pH 7."""
        charge = 0.0
        ph = 7.0
        # Side chain contributions
        charge += sequence.count('K') * (1 / (1 + 10**(ph - self.pka_values['K'])))
        charge += sequence.count('R') * (1 / (1 + 10**(ph - self.pka_values['R'])))
        charge += sequence.count('H') * (1 / (1 + 10**(ph - self.pka_values['H'])))
        charge -= sequence.count('D') * (1 / (1 + 10**(self.pka_values['D'] - ph)))
        charge -= sequence.count('E') * (1 / (1 + 10**(self.pka_values['E'] - ph)))
        charge -= sequence.count('C') * (1 / (1 + 10**(self.pka_values['C'] - ph)))
        charge -= sequence.count('Y') * (1 / (1 + 10**(self.pka_values['Y'] - ph)))
        # Terminal contributions
        charge += (1 / (1 + 10**(ph - self.pka_values['N_term']))) # N-term (+)
        charge -= (1 / (1 + 10**(self.pka_values['C_term'] - ph))) # C-term (-)
        return charge

    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate raw GRAVY score and absolute charge at pH 7 as solubility proxies."""
        if not sequence: # Handle empty sequence
             return {'solubility_proxy_gravy': 0.0, 'solubility_proxy_abs_charge': 0.0}

        try:
            protein = ProteinAnalysis(sequence)
            gravy = protein.gravy()
        except Exception as e:
            warnings.warn(f"ProtParam analysis failed for sequence (len {len(sequence)}): {e}. Setting GRAVY to 0.")
            gravy = 0.0

        abs_charge = abs(self._calculate_charge_at_ph7(sequence))

        raw_data = {
            'solubility_proxy_gravy': gravy,
            'solubility_proxy_abs_charge': abs_charge
        }
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Provides GRAVY directly and normalizes absolute charge by length."""
        if hasattr(self, 'raw_data') and self.raw_data is not None:
             raw = self.raw_data
        else:
             raw = self.calculate_raw(sequence)

        length = len(sequence)
        abs_charge_norm = (raw['solubility_proxy_abs_charge'] / length) if length > 0 else 0.0

        ml_data = {
            'solubility_gravy': raw['solubility_proxy_gravy'], # GRAVY is often used directly
            'solubility_abs_charge_norm': abs_charge_norm
        }
        self.ml_data = ml_data
        return ml_data

class PredictedAggregationFeature(BaseFeature):
    """
    Estimates aggregation propensity based on hydrophobicity, charge,
    and aromatic content. Higher hydrophobicity, lower charge, and
    higher aromatic content can contribute to aggregation.
    """
    def __init__(self):
        super().__init__()
        self.aromatic_aas = set('FWY')
        # Use GRAVY score from PredictedSolubilityFeature raw calculation
        self.raw_columns = ['aggregation_proxy_gravy', 'aggregation_proxy_abs_charge', 'aggregation_proxy_aromatic_frac']
        # Include individual components and a simple composite score
        self.ml_columns = ['aggregation_gravy', 'aggregation_abs_charge_norm', 'aggregation_aromatic_frac', 'aggregation_composite_score']

    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate raw components related to aggregation."""
        if not sequence:
            return {'aggregation_proxy_gravy': 0.0, 'aggregation_proxy_abs_charge': 0.0, 'aggregation_proxy_aromatic_frac': 0.0}

        # Re-use calculation logic from PredictedSolubilityFeature for consistency
        # This avoids recalculating the same things multiple times if both features are run
        solubility_calc = PredictedSolubilityFeature() # Temporary instance
        sol_raw = solubility_calc.calculate_raw(sequence)

        gravy = sol_raw['solubility_proxy_gravy']
        abs_charge = sol_raw['solubility_proxy_abs_charge']

        aromatic_count = sum(1 for aa in sequence if aa in self.aromatic_aas)
        aromatic_frac = aromatic_count / len(sequence) if len(sequence) > 0 else 0.0

        raw_data = {
            'aggregation_proxy_gravy': gravy,
            'aggregation_proxy_abs_charge': abs_charge,
            'aggregation_proxy_aromatic_frac': aromatic_frac
        }
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculates ML-ready features including a simple composite score."""
        if hasattr(self, 'raw_data') and self.raw_data is not None:
             raw = self.raw_data
        else:
             raw = self.calculate_raw(sequence)

        length = len(sequence)
        abs_charge_norm = (raw['aggregation_proxy_abs_charge'] / length) if length > 0 else 0.0
        gravy = raw['aggregation_proxy_gravy']
        aromatic_frac = raw['aggregation_proxy_aromatic_frac']

        # Simple heuristic composite score (weights are arbitrary, for demonstration)
        # Higher score indicates higher predicted aggregation propensity
        # Higher GRAVY -> more aggregation (+)
        # Higher Absolute Charge -> less aggregation (-)
        # Higher Aromatic Fraction -> more aggregation (+)
        # We need to scale these roughly first, e.g., to ~[0,1] or [-1,1]
        # Simple scaling (adjust based on observed ranges if needed):
        scaled_gravy = (gravy + 4.5) / 9.0 # Approx scale GRAVY from [-4.5, 4.5] to [0, 1]
        scaled_charge = abs_charge_norm / 0.5 # Assume max charge norm around 0.5, scale to ~[0, 1+]
        # Aromatic frac is already [0, 1]

        # Example weights:
        w_gravy = 0.5
        w_charge = -0.3 # Negative weight for charge
        w_aromatic = 0.2

        composite_score = (w_gravy * scaled_gravy +
                           w_charge * scaled_charge +
                           w_aromatic * aromatic_frac)
        # Clip score to a reasonable range if desired, e.g., [0, 1] after scaling
        composite_score = np.clip(composite_score, -1, 1) # Clip to [-1, 1] as charge term is negative

        ml_data = {
            'aggregation_gravy': gravy,
            'aggregation_abs_charge_norm': abs_charge_norm,
            'aggregation_aromatic_frac': aromatic_frac,
            'aggregation_composite_score': composite_score
        }
        self.ml_data = ml_data
        return ml_data