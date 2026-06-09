from typing import Dict, List, Any
import numpy as np
from Bio.PDB import *
from Bio.PDB.Atom import Atom
from ..base_feature import BaseFeature
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GeometryFeatureMixin:
    """Mixin for structure coordinate calculations"""
    
    def _get_coords(self, pdb_path: str) -> np.ndarray:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", pdb_path)
        model = structure[0]
        # Get CA atoms for backbone trace
        coords = np.array([atom.get_coord() 
                          for atom in model.get_atoms() 
                          if atom.get_name() == 'CA'])
        return coords

    def _get_all_atoms(self, pdb_path: str) -> List['Atom']:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", pdb_path)
        return list(structure.get_atoms())

class RadiusOfGyrationFeature(BaseFeature, GeometryFeatureMixin):
    """
    Radius of gyration measures the overall spatial distribution of the structure.
    
    Indicates:
    - Overall size and compactness
    - Potential for efficient packing
    - Globularity vs extension
    
    For templating: More compact structures (lower Rg) might template more effectively
    due to stable self-interactions.
    
    ML-ready output columns:
    - rg_norm: Float [0-1], normalized radius of gyration
      Lower values indicate more compact structure
    """
    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        self.raw_columns = ['radius_gyration']
        self.ml_columns = ['rg_norm']
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        coords = self._get_coords(pdb_path)
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
        raw_data = {'radius_gyration': rg}
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
        
        # Normalize by typical max RG for small peptides
        ml_data = {'rg_norm': raw['radius_gyration'] / 30.0}
        self.ml_data = ml_data
        return ml_data

class AsphericityFeature(BaseFeature, GeometryFeatureMixin):
    """
    Measures deviation from perfect spherical shape.
    
    Indicates:
    - Shape anisotropy
    - Structural asymmetry
    - Potential interaction surfaces
    
    For templating: Moderate asphericity might indicate useful interaction surfaces
    while maintaining overall stability.
    
    ML-ready output columns:
    - asphericity: Float [0-1], deviation from spherical shape
      0 = perfect sphere, 1 = maximally aspherical
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['asphericity']
        self.ml_columns = ['asphericity']
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        coords = self._get_coords(pdb_path)
        # Calculate gyration tensor
        center = np.mean(coords, axis=0)
        centered_coords = coords - center
        tensor = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                tensor[i,j] = np.mean(centered_coords[:,i] * centered_coords[:,j])
        
        # Get eigenvalues
        evals = np.linalg.eigvals(tensor)
        evals.sort()
        
        # Calculate asphericity
        asphericity = ((evals[2] - evals[0])**2 + 
                      (evals[2] - evals[1])**2 + 
                      (evals[1] - evals[0])**2) / (2 * np.sum(evals)**2)
        raw_data = {'asphericity': asphericity}
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            ml_data = self.raw_data
        else:
            ml_data = self.calculate_raw(pdb_path)
            
        self.ml_data = ml_data
        return ml_data

class ContactDensityFeature(BaseFeature, GeometryFeatureMixin):
    """
    Measures the density of atomic contacts within different sequence ranges.
    
    Indicates:
    - Local and non-local structural organization
    - Packing efficiency at different scales
    - Balance between local and long-range interactions
    
    For templating: High contact density often indicates stable, well-packed
    structures that might serve as effective templates. Balance between short
    and long-range contacts suggests robust folding.
    
    ML-ready output columns:
    - short_range_density: Float [0-1], density of contacts within 5 residues
      Higher values indicate tight local packing
    - long_range_density: Float [0-1], transformed density of contacts beyond 5 residues
      Higher values suggest stable tertiary structure
    """
    
    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        self.cutoff = cutoff
        self.raw_columns = ['short_range_contacts', 'long_range_contacts', 'total_residues']
        self.ml_columns = ['short_range_density', 'long_range_density']
        
        # Constants for transformation (can be tuned if needed)
        self.long_range_scale_factor = 50.0  # Scaling factor for more aggressive transformation
        self.long_range_min = 0.0  # Will be updated during fitting
        self.long_range_max = 1.0  # Will be updated during fitting
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        coords = self._get_coords(pdb_path)
        n_res = len(coords)
        
        # Return early if no residues
        if n_res == 0:
            return {
                'short_range_contacts': 0,
                'long_range_contacts': 0,
                'total_residues': 0
            }
        
        # Calculate all pairwise distances
        contacts = np.zeros((n_res, n_res))
        for i in range(n_res):
            for j in range(i+1, n_res):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= self.cutoff:
                    contacts[i,j] = contacts[j,i] = 1
                    
        # Count short/long range contacts
        short_range = 0
        long_range = 0
        for i in range(n_res):
            for j in range(i+1, n_res):
                if contacts[i,j]:
                    if j-i < 5:
                        short_range += 1
                    else:
                        long_range += 1
                        
        raw_data = {
            'short_range_contacts': short_range,
            'long_range_contacts': long_range,
            'total_residues': n_res
        }
        self.raw_data = raw_data
        return raw_data
    
    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        n_res = raw['total_residues']
        
        # Handle edge cases
        if n_res < 2:  # Need at least 2 residues for contacts
            ml_data = {
                'short_range_density': 0.0,
                'long_range_density': 0.0
            }
            self.ml_data = ml_data
            return ml_data
            
        # Normalize by maximum possible contacts
        max_short = 4 * n_res  # approximate max short-range
        max_long = (n_res * (n_res-1))//2 - max_short  # remaining as long-range
        
        # Calculate raw density values
        short_range_density = raw['short_range_contacts'] / max_short if max_short > 0 else 0.0
        long_range_density = raw['long_range_contacts'] / max_long if max_long > 0 else 0.0
        
        # Apply transformations
        # short_range_density is already well-distributed, no transformation needed
        
        # More aggressive transformation for long_range_density
        # 1. Scale by a factor to spread out small values
        # 2. Apply log transform 
        # 3. Apply min-max scaling to get back to [0-1] range
        if long_range_density > 0:
            # Apply aggressive log transform with scaling
            transformed_long = np.log1p(self.long_range_scale_factor * long_range_density)
            
            # Min-max scaling to [0-1] range (using stored min/max values)
            if self.long_range_max > self.long_range_min:
                transformed_long = (transformed_long - self.long_range_min) / (self.long_range_max - self.long_range_min)
            else:
                transformed_long = 0.0
        else:
            transformed_long = 0.0
            
        ml_data = {
            'short_range_density': short_range_density,
            'long_range_density': transformed_long
        }
        self.ml_data = ml_data
        return ml_data
    
    def fit_transformation_params(self, long_range_densities: np.ndarray):
        """
        Fit transformation parameters based on actual data.
        
        Args:
            long_range_densities: Array of raw long_range_density values
        """
        # Filter out zeros and negative values
        positive_values = long_range_densities[long_range_densities > 0]
        
        if len(positive_values) > 0:
            # Calculate scaled and log-transformed values
            transformed_values = np.log1p(self.long_range_scale_factor * positive_values)
            
            # Store min and max for normalization
            self.long_range_min = np.min(transformed_values)
            self.long_range_max = np.max(transformed_values)
        else:
            # Default values if no positive densities
            self.long_range_min = 0.0
            self.long_range_max = 1.0
    
    def inverse_transform_long_range(self, transformed_value: float) -> float:
        """
        Convert a transformed long_range_density back to its original scale.
        
        Args:
            transformed_value: The transformed value
            
        Returns:
            The original density value
        """
        if transformed_value <= 0:
            return 0.0
            
        # Undo min-max scaling
        log_value = transformed_value * (self.long_range_max - self.long_range_min) + self.long_range_min
        
        # Undo log transform and scaling
        return (np.exp(log_value) - 1) / self.long_range_scale_factor
    
    # Custom plotting method
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Create custom plots for ContactDensityFeature distributions.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Plot raw features
        for idx, col in enumerate(raw_data.columns):
            ax = fig.add_subplot(gs[0, idx])
            data = raw_data[col].dropna()
            
            if data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                sns.histplot(data=data, ax=ax, bins=30, kde=True, color='blue', alpha=0.6)
                ax.set_title(f"Raw {col}")
        
        # Plot ML-ready features
        for idx, col in enumerate(ml_data.columns):
            ax = fig.add_subplot(gs[1, idx])
            data = ml_data[col].dropna()
            
            if data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                sns.histplot(data=data, ax=ax, bins=30, kde=True, color='red', alpha=0.6)
                ax.set_title(f"ML-ready {col}")
                
                # Add annotation for long_range_density transformation
                if col == 'long_range_density':
                    ax.text(0.95, 0.95, f"Transform: scaled log", 
                           transform=ax.transAxes, ha='right', va='top', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle('Contact Density Distributions')
        plt.tight_layout()
        return fig

class ElongationFeature(BaseFeature, GeometryFeatureMixin):
    """
    Measures shape elongation using principal components.
    
    Indicates:
    - Overall shape characteristics
    - Major conformational axes
    - Potential interaction geometries
    
    For templating: Shape characteristics might influence
    templating efficiency and specificity.
    
    ML-ready output columns:
    - major_minor_ratio: Float [1-inf], shape elongation
      Higher values indicate more elongated structures
    - planarity: Float [0-1], flatness measure
      Higher values indicate more planar structures
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['major_axis', 'intermediate_axis', 'minor_axis']
        self.ml_columns = ['major_minor_ratio', 'planarity']
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        coords = self._get_coords(pdb_path)
        # Center coordinates
        centered = coords - np.mean(coords, axis=0)
        # Calculate covariance matrix
        cov = np.cov(centered.T)
        # Get eigenvalues (principal axes)
        evals = np.linalg.eigvals(cov)
        evals.sort()
        
        raw_data = {
            'major_axis': np.sqrt(evals[2]),
            'intermediate_axis': np.sqrt(evals[1]),
            'minor_axis': np.sqrt(evals[0])
        }
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        ml_data = {
            'major_minor_ratio': raw['major_axis'] / raw['minor_axis'],
            'planarity': raw['intermediate_axis'] / raw['major_axis']
        }
        self.ml_data = ml_data
        return ml_data

class MolecularVolumeFeature(BaseFeature, GeometryFeatureMixin):
    """
    Estimates molecular volume using convex hull approximation.
    
    Indicates:
    - Space occupied by structure
    - Packing efficiency
    - Potential cavity presence
    
    For templating: Volume efficiency might correlate with stable
    self-interactions needed for templating.
    
    ML-ready output columns:
    - volume_norm: Float [0-1], normalized molecular volume
      Lower values suggest more efficient packing
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['volume']
        self.ml_columns = ['volume_norm']
        
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        coords = self._get_coords(pdb_path)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            raw_data = {'volume': hull.volume}
        except:
            # Fallback to rough estimate using bounding box
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
            raw_data = {'volume': np.prod(maxs - mins)}
            
        self.raw_data = raw_data
        return raw_data

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        # Normalize by typical max volume for small peptides
        ml_data = {'volume_norm': raw['volume'] / 1000.0}  # Arbitrary normalization
        self.ml_data = ml_data
        return ml_data