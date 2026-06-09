from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any
import numpy as np

class BaseFeature(ABC):
    """Abstract base class for peptide features"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.raw_data = None
        self.ml_data = None
        self.raw_columns = []  # List of column names this feature produces in raw form
        self.ml_columns = []
    
    @abstractmethod
    def calculate_raw(self, sequence: str) -> Dict[str, Any]:
        """Calculate raw feature value"""
        pass
    
    @abstractmethod
    def calculate_ml_ready(self, sequence: str) -> Dict[str, float]:
        """Calculate ML-ready version of feature"""
        pass
    
    def calculate_bulk(self, sequences: List[str]) -> pd.DataFrame:
        """Calculate feature for multiple sequences"""
        return pd.DataFrame([
            self.calculate_ml_ready(seq) for seq in sequences
        ])
    def load_raw_data(self, data):
        """Load raw feature data from external source"""
        if isinstance(data, pd.Series):
            self.raw_data = data.to_dict()
        else:
            self.raw_data = data
        return self.raw_data
    
    def plot_distributions(self, raw_data: pd.DataFrame, ml_data: pd.DataFrame) -> plt.Figure:
        """
        Default distribution plotting for features.
        
        Args:
            raw_data: DataFrame with raw feature values
            ml_data: DataFrame with ML-ready feature values
            
        Returns:
            matplotlib Figure object
        """
        # Check if we have data to plot
        if raw_data.empty or ml_data.empty:
            print(f"Warning: No data to plot for {self.__class__.__name__}")
            # Return empty figure
            fig = plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, 'No data available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            return fig

        fig = plt.figure(figsize=(15, 10))
        n_raw_cols = raw_data.shape[1]
        n_ml_cols = ml_data.shape[1]
        
        if n_raw_cols == 0 or n_ml_cols == 0:
            plt.text(0.5, 0.5, 'No features available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            return fig

        gs = plt.GridSpec(2, max(n_raw_cols, n_ml_cols))
        
        # Plot raw features
        for idx, col in enumerate(raw_data.columns):
            if raw_data[col].empty:
                continue
            ax = plt.subplot(gs[0, idx])
            data = raw_data[col].dropna()
            
            # Skip non-numeric data
            if not pd.api.types.is_numeric_dtype(data):
                ax.text(0.5, 0.5, f"Non-numeric: {col}", ha='center', va='center')
                continue
                
            # Handle binning based on data characteristics
            if len(data) > 0:
                if data.nunique() <= 1:
                    # Special case for single value - create range around it
                    val = data.iloc[0]
                    value_range = max(abs(val * 0.1), 0.1) if val != 0 else 0.2
                    bins = np.linspace(val - value_range, val + value_range, 10)
                    sns.histplot(data=data, ax=ax, bins=bins, color='blue', alpha=0.6, kde=True)
                elif data.std() < 1e-6 and data.nunique() > 1:  # Very low variance but multiple values
                    # Create bins with forced spread for better visualization
                    mean_val = data.mean()
                    spread = max(data.std() * 10, abs(mean_val * 0.2), 0.1)
                    bins = np.linspace(data.min() - spread/10, data.max() + spread/10, 15)
                    sns.histplot(data=data, ax=ax, bins=bins, color='blue', alpha=0.6, kde=True)
                    # Add rugplot to show actual data points
                    sns.rugplot(data=data, ax=ax, color='darkblue', height=0.05)
                else:
                    # Normal case with reasonable variance
                    bins = min(max(15, data.nunique()), 50)  # Between 15 and 50 bins
                    sns.histplot(data=data, ax=ax, bins=bins, color='blue', alpha=0.6, kde=True)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                
            ax.set_title(f'Raw {col}')
            ax.tick_params(labelrotation=45)
        
        # Plot ML-ready features with more aggressive handling for low-variance
        for idx, col in enumerate(ml_data.columns):
            if ml_data[col].empty:
                continue
            ax = plt.subplot(gs[1, idx])
            data = ml_data[col].dropna()
            
            # Skip non-numeric data
            if not pd.api.types.is_numeric_dtype(data):
                ax.text(0.5, 0.5, f"Non-numeric: {col}", ha='center', va='center')
                continue
                
            # Handle binning based on data characteristics
            if len(data) > 0:
                if data.nunique() <= 1:
                    # Even more aggressive for single ML values
                    val = data.iloc[0]
                    value_range = max(abs(val * 0.15), 0.15) if val != 0 else 0.3
                    bins = np.linspace(val - value_range, val + value_range, 15)
                    sns.histplot(data=data, ax=ax, bins=bins, color='red', alpha=0.6, kde=True)
                    # Add text to indicate artificial spread
                    ax.text(0.5, 0.9, f'Single value: {val:.6f}', 
                            transform=ax.transAxes, ha='center', fontsize=8)
                elif data.std() < 1e-5:  # Extremely low variance
                    # Use very aggressive spreading for ML features with tiny variance
                    mean_val = data.mean()
                    std_val = data.std() if data.std() > 0 else 0.001
                    min_val, max_val = data.min(), data.max()
                    # Ensure spread is at least 20x the actual range
                    actual_range = max_val - min_val
                    needed_spread = max(actual_range * 20, abs(mean_val * 0.3), 0.2)
                    bins = np.linspace(min_val - needed_spread/10, max_val + needed_spread/10, 20)
                    
                    # Plot with wider bins and add rugplot
                    sns.histplot(data=data, ax=ax, bins=bins, color='red', alpha=0.4, kde=True)
                    sns.rugplot(data=data, ax=ax, color='darkred', height=0.1)
                    
                    # Add annotation about low variance
                    ax.text(0.5, 0.9, f'Low variance: σ={std_val:.6f}', 
                            transform=ax.transAxes, ha='center', fontsize=8)
                elif data.min() >= 0 and data.max() <= 1 and (data.max() - data.min()) < 0.1:
                    # Special case for normalized features (0-1) with small range
                    min_val, max_val = data.min(), data.max()
                    margin = max((max_val - min_val) * 0.2, 0.01)
                    bins = np.linspace(max(0, min_val - margin), min(1, max_val + margin), 20)
                    sns.histplot(data=data, ax=ax, bins=bins, color='red', alpha=0.6, kde=True)
                else:
                    # Normal case with reasonable variance
                    bins = min(max(20, data.nunique()), 50)  # Between 20 and 50 bins
                    sns.histplot(data=data, ax=ax, bins=bins, color='red', alpha=0.6, kde=True)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                
            ax.set_title(f'ML-ready {col}')
            ax.tick_params(labelrotation=45)
        
        plt.tight_layout()
        return fig

    

    