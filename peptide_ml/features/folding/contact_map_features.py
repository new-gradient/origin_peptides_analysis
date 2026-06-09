from typing import Dict, List, Any
import numpy as np
from Bio.PDB import *
from ..base_feature import BaseFeature
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.sparse.csgraph import connected_components
from typing import Dict, Tuple, Any, Optional
from scipy import stats
import warnings 


class ContactFeatureMixin:
    """Mixin for contact map calculations"""
    def _get_contact_map(self, pdb_path: str, cutoff: float = 8.0) -> np.ndarray:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", pdb_path)
        model = structure[0]
        
        # Get CA atoms
        ca_atoms = [atom for atom in model.get_atoms() 
                   if atom.get_name() == 'CA']
        n_res = len(ca_atoms)
        
        # Calculate contact map
        contact_map = np.zeros((n_res, n_res))
        for i in range(n_res):
            for j in range(i+1, n_res):
                dist = ca_atoms[i] - ca_atoms[j]
                if dist <= cutoff:
                    contact_map[i,j] = contact_map[j,i] = 1
        return contact_map
    def _get_atom_coordinates(self, pdb_path: str) -> np.ndarray:
        """Get coordinates of CA atoms from a PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", pdb_path)
        model = structure[0]
        
        # Get CA atoms
        ca_atoms = [atom for atom in model.get_atoms() 
                if atom.get_name() == 'CA']
        
        # Extract coordinates
        coordinates = np.array([atom.get_coord() for atom in ca_atoms])
        
        return coordinates

class ContactOrderFeature(BaseFeature, ContactFeatureMixin):
    # ... (docstring and __init__ remain the same) ...

    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        print(f"  [ContactOrderFeature] Calculating raw for PDB: {pdb_path}") # Debug print
        contact_map = self._get_contact_map(pdb_path)
        n_res = len(contact_map)
        print(f"  [ContactOrderFeature] n_res = {n_res}") # Debug print

        total_separation = 0
        n_contacts = 0
        for i in range(n_res):
            for j in range(i+1, n_res):
                if contact_map[i,j]:
                    total_separation += abs(i - j)
                    n_contacts += 1

        print(f"  [ContactOrderFeature] total_separation = {total_separation}, n_contacts = {n_contacts}") # Debug print

        if n_contacts == 0 or n_res == 0: # Explicitly check n_res too
            print("  [ContactOrderFeature] No contacts or residues found, returning 0.0") # Debug print
            raw_data_dict = {'contact_order': 0.0}
            self.raw_data = raw_data_dict # Store the dict
            return raw_data_dict

        # Check for division by zero before calculating
        denominator = (n_contacts * n_res)
        if denominator == 0:
             print("  [ContactOrderFeature] Denominator is zero, returning 0.0") # Debug print
             raw_data_dict = {'contact_order': 0.0}
        else:
             co = total_separation / denominator
             print(f"  [ContactOrderFeature] Calculated contact_order = {co}") # Debug print
             raw_data_dict = {'contact_order': co}

        # <<< FIX: Store the dictionary in self.raw_data >>>
        self.raw_data = raw_data_dict
        return raw_data_dict

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        print("  [ContactOrderFeature] Calculating ML-ready...") # Debug print
        # <<< FIX: Check self.raw_data correctly and get value >>>
        # Check if raw_data (the dict) exists and has the key
        if hasattr(self, 'raw_data') and self.raw_data is not None and 'contact_order' in self.raw_data:
            print("  [ContactOrderFeature] Using stored raw_data.") # Debug print
            raw_value = self.raw_data['contact_order']
            raw_dict_for_ml = self.raw_data # Keep the dict structure for consistency if needed elsewhere
        else:
            print("  [ContactOrderFeature] Calculating raw data first.") # Debug print
            raw_dict_for_ml = self.calculate_raw(pdb_path) # calculate_raw now correctly sets self.raw_data
            raw_value = raw_dict_for_ml['contact_order']
        # <<< END FIX >>>

        # The ML value is currently the same as the raw value for this feature
        ml_data = {'contact_order': raw_value}
        self.ml_data = ml_data # Store the ML dictionary
        print(f"  [ContactOrderFeature] Final ML data: {ml_data}") # Debug print
        # Check for NaN before returning
        if pd.isna(ml_data['contact_order']):
            print("  [ContactOrderFeature] !!! WARNING: ML-ready contact_order is NaN !!!") # Debug print
        return ml_data

class LongRangeContactFeature(BaseFeature, ContactFeatureMixin):
    """
    Statistics of contacts between residues far apart in sequence.
    
    Long-range contacts reflect:
    - Tertiary structure stability
    - Domain organization
    - Potential folding nuclei
    
    For templating: High density of well-distributed long-range contacts may
    indicate stable structures that could serve as good templates.
    
    ML-ready output columns:
    - lr_contact_density: Float, transformed fraction of possible long-range contacts
      Higher values indicate more stable tertiary structure
    - lr_contact_distribution: Float, transformed evenness of contact distribution
      Higher values suggest more uniform distribution of stabilizing contacts
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['n_contacts', 'mean_separation', 'std_separation']
        self.ml_columns = ['lr_contact_density', 'lr_contact_distribution']
        
        # Transformation parameters - can be tuned during fitting
        self.transformation_params = {
            'lr_contact_density': {
                'scale_factor': 200.0,  # Aggressive scaling for very small values
                'min_val': 0.0,  # Will be updated during fitting
                'max_val': 1.0   # Will be updated during fitting
            },
            'lr_contact_distribution': {
                'scale_factor': 150.0,  # Aggressive scaling
                'min_val': 0.0,  # Will be updated during fitting
                'max_val': 1.0   # Will be updated during fitting
            }
        }
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        contact_map = self._get_contact_map(pdb_path)
        n_res = len(contact_map)
        
        # Count long-range contacts (>5 residues apart)
        lr_contacts = []
        for i in range(n_res):
            for j in range(i+5, n_res):
                if contact_map[i,j]:
                    lr_contacts.append(j-i)
                    
        if not lr_contacts:
            raw_data = {
                'n_contacts': 0,
                'mean_separation': 0,
                'std_separation': 0
            }
            self.raw_data = raw_data
            return raw_data
            
        raw_data = {
            'n_contacts': len(lr_contacts),
            'mean_separation': np.mean(lr_contacts) if lr_contacts else 0,
            'std_separation': np.std(lr_contacts) if lr_contacts else 0
        }
        self.raw_data = raw_data
        return raw_data
    
    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
            
        n_res = len(self._get_contact_map(pdb_path))
        max_lr_contacts = (n_res * (n_res-11))//2  # Maximum possible long-range contacts
        
        # Calculate raw values
        lr_contact_density = raw['n_contacts']/max_lr_contacts if max_lr_contacts > 0 else 0
        lr_contact_distribution = raw['std_separation']/raw['mean_separation'] if raw['mean_separation'] > 0 else 0
        
        # Apply transformations
        density_params = self.transformation_params['lr_contact_density']
        distribution_params = self.transformation_params['lr_contact_distribution']
        
        # Transform lr_contact_density
        transformed_density = self._transform_value(
            lr_contact_density, 
            density_params['scale_factor'],
            density_params['min_val'],
            density_params['max_val']
        )
        
        # Transform lr_contact_distribution
        transformed_distribution = self._transform_value(
            lr_contact_distribution,
            distribution_params['scale_factor'],
            distribution_params['min_val'],
            distribution_params['max_val']
        )
        
        ml_data = {
            'lr_contact_density': transformed_density,
            'lr_contact_distribution': transformed_distribution
        }
        self.ml_data = ml_data
        return ml_data
    
    def _transform_value(self, value: float, scale_factor: float, min_val: float, max_val: float) -> float:
        """
        Apply transformation to a value:
        1. Scale by a factor
        2. Apply log1p
        3. Apply min-max normalization
        
        Args:
            value: The value to transform
            scale_factor: Scaling factor to apply before log
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            
        Returns:
            Transformed value
        """
        if value <= 0:
            return 0.0
            
        # Apply aggressive log transform with scaling
        log_val = np.log1p(scale_factor * value)
        
        # Apply min-max scaling if range is valid
        if max_val > min_val:
            return (log_val - min_val) / (max_val - min_val)
        return log_val  # Fallback if no valid range
    
    def inverse_transform(self, transformed_value: float, feature_name: str) -> float:
        """
        Convert a transformed value back to its original scale.
        
        Args:
            transformed_value: The transformed value
            feature_name: Name of the feature (lr_contact_density or lr_contact_distribution)
            
        Returns:
            Original value
        """
        if transformed_value <= 0:
            return 0.0
            
        params = self.transformation_params.get(feature_name)
        if not params:
            return transformed_value
            
        # Undo min-max scaling
        log_val = transformed_value * (params['max_val'] - params['min_val']) + params['min_val']
        
        # Undo log transform and scaling
        return (np.exp(log_val) - 1) / params['scale_factor']
    
    def fit_transformation_params(self, data: Dict[str, np.ndarray]):
        """
        Fit transformation parameters based on actual data.
        
        Args:
            data: Dictionary with keys 'lr_contact_density' and 'lr_contact_distribution'
                 containing arrays of raw values
        """
        for feature_name, values in data.items():
            if feature_name not in self.transformation_params:
                continue
                
            params = self.transformation_params[feature_name]
            scale_factor = params['scale_factor']
            
            # Filter out zeros and calculate log-transformed values
            positive_values = values[values > 0]
            if len(positive_values) > 0:
                # Apply scaled log transform
                log_vals = np.log1p(scale_factor * positive_values)
                
                # Update min/max values for normalization
                params['min_val'] = np.min(log_vals)
                params['max_val'] = np.max(log_vals)
            else:
                # Default values if no positive values
                params['min_val'] = 0.0
                params['max_val'] = 1.0
    
    # Custom plotting method for better visualization
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Create custom plots for LongRangeContactFeature distributions.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Plot raw features
        for idx, col in enumerate(raw_data.columns):
            ax = fig.add_subplot(gs[0, idx])
            data = raw_data[col].dropna()
            
            if data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                # For extremely skewed data, use log scale on x-axis
                if col in ['n_contacts']:
                    sns.histplot(data=data, ax=ax, bins=30, kde=True, color='blue', alpha=0.6)
                    if data.max() > 10 * data.median():  # If highly skewed
                        ax.set_xscale('log')
                        ax.text(0.95, 0.95, "Log scale", transform=ax.transAxes, 
                               ha='right', va='top', fontsize=8,
                               bbox=dict(facecolor='white', alpha=0.7))
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
                
                # Add annotation about transformation
                ax.text(0.95, 0.95, "Transform: scaled log + normalization", 
                       transform=ax.transAxes, ha='right', va='top', 
                       fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
                
                # If feature has a scale factor, show it
                if col in self.transformation_params:
                    scale = self.transformation_params[col]['scale_factor']
                    ax.text(0.95, 0.87, f"Scale factor: {scale:.1f}", 
                           transform=ax.transAxes, ha='right', va='top', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle('Long Range Contact Distributions')
        plt.tight_layout()
        return fig
        
class ContactClusterFeature(BaseFeature, ContactFeatureMixin):
    """
    Measures how contacts are grouped into clusters in the structure.
    
    Contact clusters indicate:
    - Cooperative folding units
    - Structural domains
    - Potential nucleation sites
    
    For templating: Well-defined clusters might represent stable substructures
    that could serve as independent templating units.
    
    ML-ready output columns:
    - cluster_density: Float [0-1], density of contact clusters
      Higher values indicate more distinct cooperative units
    - avg_cluster_size_ratio: Float [0-1], ratio of average to max cluster size
      Measures uniformity of cluster sizes (replaced avg_cluster_size)
    - clusters_per_residue: Float [0-1], ratio of clusters to residues
      Indicates structural granularity (replaced n_clusters_norm)
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['n_clusters', 'avg_cluster_size', 'max_cluster_size', 'total_residues', 'cluster_sizes']
        self.ml_columns = ['cluster_density', 'avg_cluster_size_ratio', 'clusters_per_residue']
        
        # Contact threshold - adjust this value to get more meaningful clusters
        # Lower values produce more clusters
        self._contact_threshold = 8.0  # Angstroms, default is often 8Å
    
    def _get_contact_map(self, pdb_path: str, threshold: Optional[float] = None) -> np.ndarray:
        """
        Override to use adjustable contact threshold.
        """
        # Use the provided threshold or the instance attribute
        contact_threshold = threshold if threshold is not None else self._contact_threshold
        
        # Get contact map using parent method but with custom threshold
        # This is a simplified example - you'll need to adapt to your actual implementation
        coords = self._get_atom_coordinates(pdb_path)
        n_res = len(coords)
        
        # Calculate pairwise distances
        contact_map = np.zeros((n_res, n_res))
        for i in range(n_res):
            for j in range(i+1, n_res):
                # Calculate Euclidean distance between CA atoms
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= contact_threshold:
                    contact_map[i, j] = contact_map[j, i] = 1
        
        return contact_map
    
    def calculate_raw(self, pdb_path: str) -> Dict[str, Any]:
        """Calculate raw contact cluster features with adjustable threshold."""
        # Try multiple thresholds if the default gives a single cluster
        thresholds = [8.0, 7.0, 6.0, 5.0]
        
        for threshold in thresholds:
            contact_map = self._get_contact_map(pdb_path, threshold)
            
            # Find contact clusters using connected components
            n_clusters, labels = connected_components(contact_map)
            
            # If we have more than one cluster, we can stop
            if n_clusters > 1:
                break
        
        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = counts.tolist()
        
        raw_data = {
            'n_clusters': n_clusters,
            'avg_cluster_size': np.mean(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'total_residues': len(contact_map),
            'cluster_sizes': cluster_sizes  # Store the actual sizes for better features
        }
        
        # Add a warning if we still have only one cluster
        if n_clusters == 1:
            warnings.warn(f"Only one contact cluster found for {pdb_path} even with reduced threshold")
        
        self.raw_data = raw_data
        return raw_data
    
    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        """Calculate ML-ready features with more informative metrics."""
        if hasattr(self, 'raw_data') and self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
        
        n_res = raw['total_residues']
        n_clusters = raw['n_clusters']
        
        # Calculate better features that avoid the single-value problem
        
        # 1. Cluster density - ratio of clusters to possible connections
        # Max possible connections is n_res * (n_res - 1) / 2
        max_connections = n_res * (n_res - 1) / 2 if n_res > 1 else 1
        cluster_density = n_clusters / np.sqrt(max_connections) if max_connections > 0 else 0
        
        # 2. Average cluster size ratio - use ratio to max cluster size instead of total residues
        # This avoids the 1.0 value when there's only one cluster
        max_size = raw['max_cluster_size']
        avg_size = raw['avg_cluster_size']
        avg_size_ratio = avg_size / max_size if max_size > 0 else 0
        
        # If there's only one cluster, this will be 1.0, but at least it's meaningful
        
        # 3. Clusters per residue - more informative than normalized cluster count
        # Apply a scaling to spread values when counts are low
        clusters_per_residue = (n_clusters / n_res) * np.log1p(n_clusters) if n_res > 0 else 0
        
        # To help avoid single values, add small random noise for ML training
        # Note: only do this for ML training, not for final analysis
        # jitter = 1e-4  # Very small amount of noise
        # avg_size_ratio += np.random.uniform(-jitter, jitter)
        # clusters_per_residue += np.random.uniform(-jitter, jitter)
        
        ml_data = {
            'cluster_density': cluster_density,
            'avg_cluster_size_ratio': avg_size_ratio,
            'clusters_per_residue': clusters_per_residue
        }
        
        self.ml_data = ml_data
        return ml_data
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Create custom plots for ContactClusterFeature distributions.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Plot raw features
        for idx, col in enumerate(self.raw_columns[:4]):  # Skip cluster_sizes
            ax = fig.add_subplot(gs[0, idx])
            
            # Check if we have a single-value feature
            if col in raw_data.columns and raw_data[col].nunique() <= 1:
                # Special case for single value
                val = raw_data[col].iloc[0]
                ax.bar([val], [1], width=0.01, color='blue', alpha=0.6)
                ax.set_title(f"Raw {col}")
                ax.text(val, 0.5, f"Single value: {val}", 
                       ha='center', va='center', fontsize=10)
                
                # Add explanation for single-value n_clusters
                if col == 'n_clusters' and val == 1:
                    ax.text(val, 0.3, "Consider lowering contact threshold\nto find more clusters", 
                           ha='center', va='center', fontsize=8, color='red',
                           bbox=dict(facecolor='white', alpha=0.7))
            else:
                if col in raw_data.columns:
                    data = raw_data[col].dropna()
                    if not data.empty:
                        sns.histplot(data=data, ax=ax, bins=30, kde=True, color='blue', alpha=0.6)
                        ax.set_title(f"Raw {col}")
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, f'Column {col} not found', ha='center', va='center')
        
        # Plot ML-ready features
        for idx, col in enumerate(self.ml_columns):
            ax = fig.add_subplot(gs[1, idx])
            
            if col in ml_data.columns:
                data = ml_data[col].dropna()
                if data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                elif data.nunique() <= 1:
                    # Special case for single values
                    val = data.iloc[0]
                    ax.bar([val], [1], width=0.01, color='red', alpha=0.6)
                    ax.set_title(f"ML-ready {col}")
                    ax.text(val, 0.5, f"Single value: {val:.6f}", 
                           ha='center', va='center', fontsize=10)
                    
                    # Explanation for single-value ml features
                    if col == 'avg_cluster_size_ratio' and val == 1.0:
                        ax.text(val, 0.3, "Expected when n_clusters=1\nImproved with multiple clusters", 
                               ha='center', va='center', fontsize=8, color='red',
                               bbox=dict(facecolor='white', alpha=0.7))
                else:
                    sns.histplot(data=data, ax=ax, bins=30, kde=True, color='red', alpha=0.6)
                    ax.set_title(f"ML-ready {col}")
            else:
                ax.text(0.5, 0.5, f'Column {col} not found', ha='center', va='center')
        
        plt.suptitle('Contact Cluster Distributions')
        plt.tight_layout()
        return fig
        
class InteractionDensityFeature(BaseFeature, ContactFeatureMixin):
    """
    Patterns of local vs non-local contact density in the structure.
    
    Interaction density patterns reveal:
    - Balance between local and non-local structure
    - Potential for independent folding units
    - Overall compactness
    
    For templating: Balanced local/nonlocal density often indicates
    stable, well-organized structures that might template effectively.
    
    ML-ready output columns:
    - local_density: Float [0-1], density of local contacts
      Higher values indicate strong local structure
    - nonlocal_density: Float [0-1], density of non-local contacts
      Higher values suggest stable tertiary organization
    """
    def __init__(self):
        super().__init__()
        self.raw_columns = ['local_contacts', 'nonlocal_contacts', 'total_residues']
        self.ml_columns = ['local_density', 'nonlocal_density']

    def calculate_raw(self, pdb_path: str) -> Dict[str, float]:
        contact_map = self._get_contact_map(pdb_path)
        n_res = len(contact_map)
        
        local_contacts = 0
        nonlocal_contacts = 0
        
        for i in range(n_res):
            for j in range(i+1, min(i+4, n_res)):
                if contact_map[i,j]:
                    local_contacts += 1
            for j in range(i+4, n_res):
                if contact_map[i,j]:
                    nonlocal_contacts += 1

        raw = {
            'local_contacts': local_contacts,
            'nonlocal_contacts': nonlocal_contacts,
            'total_residues': n_res
        }
        self.raw_data = raw
        return raw        
        

    def calculate_ml_ready(self, pdb_path: str) -> Dict[str, float]:
        if self.raw_data:
            raw = self.raw_data
        else:
            raw = self.calculate_raw(pdb_path)
        n_res = raw['total_residues']
        max_local = 3 * n_res  # Maximum possible local contacts
        max_nonlocal = (n_res * (n_res-1))//2 - max_local  # Maximum possible non-local contacts
        ml_data = {
            'local_density': raw['local_contacts']/max_local if max_local > 0 else 0,
            'nonlocal_density': raw['nonlocal_contacts']/max_nonlocal if max_nonlocal > 0 else 0
        }
        self.ml_data = ml_data
        return ml_data
        