import pandas as pd
from pathlib import Path

class Peptide:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.fasta_path = None
        self.pdb_path = None

    def __repr__(self):
        return f"Peptide(sequence={self.sequence[:10]}..., fasta={self.fasta_path}, pdb={self.pdb_path})"

    def is_valid(self) -> bool:
        """
        A peptide is considered valid if it has a non-empty sequence
        and both FASTA and PDB paths are assigned.
        """
        return len(self.sequence) > 0 and self.fasta_path is not None and self.pdb_path is not None

class ExperimentDataProcessor:
    """
    Processes peptide experiment data from CSV format.
    
    Features:
      - Loads data from a CSV file (default location: data/data.csv).
      - Creates a list of Peptide objects based on the first column (the peptide sequences).
      - Splits the dataset into separate experiments based on the naming convention.
        For instance, if the columns are named like "Sum of Intensity Chlor WS 1  T14",
        "Sum of Intensity Chlor WS 1  T21", etc., all columns sharing the prefix
        (everything before the first "T") are grouped as one experiment.
      - Provides a method write_to_experiments() that writes each experiment's data
        to its own CSV file in a specified output folder (e.g., data/experiments).
      - Also provides methods to write individual FASTA files and create PDB mappings.
    """
    
    def __init__(self, csv_path: str, low_memory: bool = False):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path, low_memory=low_memory)
        # Assume the first column contains the peptide sequences.
        sequences = self.data.iloc[:, 0]
        self.peptides = [Peptide(seq) for seq in sequences]
        # If a 'count' column exists, drop it.
        if 'count' in self.data.columns:
            self.data = self.data.iloc[:, :-1]

    def clean_peptides(self):
        """
        Remove any peptides that are not valid. 
        A valid peptide must have a non-empty sequence and non-null FASTA and PDB paths.
        """
        original_count = len(self.peptides)
        self.peptides = [p for p in self.peptides if p.is_valid()]
        removed_count = original_count - len(self.peptides)
        print(f"Removed {removed_count} invalid peptides")
        return self.peptides

    def split_into_experiments(self) -> dict:
        """
        Splits the main data into separate experiments.
        
        Each experiment is determined by the prefix of the column names. For example, 
        given columns like "Sum of Intensity Chlor WS 1  T14", "Sum of Intensity Chlor WS 1  T21",
        all columns sharing "Sum of Intensity Chlor WS 1" are grouped together.
        
        Returns:
            A dictionary mapping the experiment name to its DataFrame. Each DataFrame includes 
            the peptide sequence (first column) and the intensity columns for that experiment.
        """
        experiments = {}
        # Exclude the first column (peptide sequence)
        cols = self.data.columns[1:]
        for col in cols:
            # Split on the first occurrence of 'T'
            parts = col.split('T', 1)
            exp_name = parts[0].strip() if len(parts) > 1 else col.strip()
            experiments.setdefault(exp_name, []).append(col)
        
        # Build a dictionary of experiment name to DataFrame
        experiment_dfs = {}
        for exp_name, exp_cols in experiments.items():
            exp_df = self.data[[self.data.columns[0]] + exp_cols]
            experiment_dfs[exp_name] = exp_df
        return experiment_dfs

    def write_to_experiments(self, output_folder: str):
        """
        Writes each experiment's DataFrame to its own CSV file.
        
        The files are saved in the specified output folder (e.g., data/experiments) with
        filenames derived from the experiment names (spaces replaced with underscores).
        """
        experiments = self.split_into_experiments()
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        for exp_name, df in experiments.items():
            # Create a safe filename
            safe_exp_name = exp_name.replace(" ", "_").replace("/", "_")
            file_path = output_path / f"{safe_exp_name}.csv"
            df.to_csv(file_path, index=False)
            print(f"Wrote experiment '{exp_name}' to {file_path}")
        return experiments

    def write_peptides_to_fasta(self, output_dir: str):
        """
        Writes each peptide to an individual FASTA file.
        
        The FASTA files are stored in the specified directory (e.g., data/peptides).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, peptide in enumerate(self.peptides, start=1):
            fasta_name = f"peptide_{idx}.fasta"
            fasta_file = output_path / fasta_name
            if not fasta_file.exists():
                with open(fasta_file, 'w') as f:
                    f.write(f">peptide_{idx}\n{peptide.sequence}\n")
            peptide.fasta_path = str(fasta_file)
        return self.peptides

    def create_pdb_mapping(self, pdb_dir: str):
        """
        Creates mappings for PDB files for each peptide.
        
        Looks for a PDB file in a folder named for each peptide (e.g., peptide_1, peptide_2, …)
        matching a wildcard pattern. The first matching PDB file is assigned to the peptide.
        """
        pdb_dir = Path(pdb_dir)
        for idx, peptide in enumerate(self.peptides, start=1):
            peptide_folder = pdb_dir / f"peptide_{idx}"
            pdb_pattern = f"peptide_{idx}_relaxed_rank_001_alphafold2_ptm_model_*_seed_000.pdb"
            files = list(peptide_folder.glob(pdb_pattern))
            files.sort()
            peptide.pdb_path = str(files[0]) if files else None
            if peptide.pdb_path is None:
                print(f"Peptide {idx}: pdb_path = {peptide.pdb_path}")
        return self.peptides

if __name__ == "__main__":
    # Example usage:
    # Set the main data file (make sure to move your core CSV to data/data.csv)
    data_csv = "data/data.csv"
    processor = ExperimentDataProcessor(data_csv)
    
    # Write separate CSVs for each experiment into data/experiments folder.
    processor.write_to_experiments("data/experiments")
    
    # Write FASTA files for each peptide (e.g., stored in data/peptides)
    processor.write_peptides_to_fasta("data/peptides")
    
    # Create PDB mappings using relative path from current file
    current_dir = Path(__file__).parent
    pdb_dir = current_dir.parent / "features" / "folding" / "folding_outputs" / "fold_outputs"
    processor.create_pdb_mapping(str(pdb_dir))