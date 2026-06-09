from pathlib import Path
import subprocess
import shutil
from typing import List, Dict
import json
import os
import pandas as pd
from datetime import datetime
import logging
import subprocess
from pathlib import Path
import time

class PeptideFoldingPipeline:
    def __init__(self, fasta_dir: str, colabfold_path: str, database_path: str, output_base: str):
        self.fasta_dir = Path(fasta_dir)
        self.colabfold_path = Path(colabfold_path)
        self.database_path = Path(database_path)
        self.output_base = Path(output_base)
        
        # Setup directories
        self.fold_outputs = self.output_base / "fold_outputs"
        self.fold_outputs.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_base / "temp_fold"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = self.output_base / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Track progress
        self.successful_peptides = []
        self.failed_peptides = []
        
    def _setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"folding_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def fold_peptide(self, fasta_path: Path) -> bool:
        try:
            logging.info(f"Starting folding for {fasta_path.stem}")
            
            with open(os.devnull, 'w') as null:
                result = subprocess.run(
                    [
                        "colabfold_batch",
                        "--msa-mode", "mmseqs2_uniref_env",
                        "--data", str(self.database_path),
                        "--templates",
                        "--amber",
                        str(fasta_path),
                        str(self.temp_dir)
                    ],
                    stdout=null,
                    stderr=null,
                    check=True,
                    timeout=360  # Timeout after 6 minutes
                )
            return True
        except subprocess.TimeoutExpired as te:
            logging.error(f"Timeout while folding {fasta_path}: {str(te)}")
            return False
        except Exception as e:
            logging.error(f"Error folding {fasta_path}: {str(e)}")
            return False

    def organize_outputs(self, peptide_id: str) -> bool:
        try:
            peptide_dir = self.fold_outputs / peptide_id
            peptide_dir.mkdir(exist_ok=True)
            
            # Check if visualization exists
            image_path = peptide_dir / f"{peptide_id}_structure.png"
            needs_visualization = not image_path.exists()
            
            patterns = [
                f"{peptide_id}_relaxed_rank_001_*.pdb",
                f"{peptide_id}_scores_rank_001_*.json",
                f"{peptide_id}_predicted_aligned_error_v1.json",
                f"{peptide_id}_plddt.png"
            ]
            
            files_found = False
            for pattern in patterns:
                for file in self.temp_dir.glob(pattern):
                    shutil.copy2(file, peptide_dir)
                    files_found = True
            
            # Clean temp directory
            for path in self.temp_dir.iterdir():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                    
            # Visualize if needed
            if needs_visualization:
                self.visualize_peptide(peptide_id)
                
            return files_found
            
        except Exception as e:
            logging.error(f"Error organizing outputs for {peptide_id}: {str(e)}")
            return False

    def process_all_peptides(self):
        total_peptides = len(list(self.fasta_dir.glob("*.fasta")))
        processed = 0
        
        for fasta_file in self.fasta_dir.glob("*.fasta"):
            peptide_id = fasta_file.stem
            processed += 1
            peptide_dir = self.fold_outputs / peptide_id
            image_path = peptide_dir / f"{peptide_id}_structure.png"
            
            logging.info(f"Processing {processed}/{total_peptides}: {peptide_id}")
            
            # Check if folding needed
            if peptide_dir.exists() and not image_path.exists():
                logging.info(f"Already folded, generating visualization for {peptide_id}")
                self.visualize_peptide(peptide_id)
                self.successful_peptides.append(peptide_id)
                continue
            elif peptide_dir.exists():
                logging.info(f"Skipping {peptide_id} - already processed")
                self.successful_peptides.append(peptide_id)
                continue
            
            # New peptide processing
            try:
                if self.fold_peptide(fasta_file) and self.organize_outputs(peptide_id):
                    self.successful_peptides.append(peptide_id)
                    logging.info(f"Successfully processed {peptide_id}")
                else:
                    self.failed_peptides.append(peptide_id)
                    logging.error(f"Failed to process {peptide_id}")
            except Exception as e:
                self.failed_peptides.append(peptide_id)
                logging.error(f"Unexpected error processing {peptide_id}: {str(e)}")
        
        self._save_results()

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'successful': pd.DataFrame({'peptide_id': self.successful_peptides}),
            'failed': pd.DataFrame({'peptide_id': self.failed_peptides})
        }
        
        for name, df in results.items():
            output_file = self.output_base / f"{name}_peptides_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"Saved {name} peptides list to {output_file}")

    def visualize_peptide(self, peptide_id: str) -> bool:
        try:
            pdb_path = next(self.fold_outputs.glob(f"{peptide_id}/{peptide_id}_relaxed_rank_001_*.pdb"))
            image_path = self.fold_outputs / peptide_id / f"{peptide_id}_structure.png"
            
            cmd = [
                "pymol",
                "-cqQ",
                "-d",
                f"load {str(pdb_path.absolute())}; "
                f"hide everything; show cartoon; "
                f"util.rainbow; "  # Changed from color spectrum
                f"bg_color white; zoom; rotate y, 90; "
                f"ray 1200, 1200; "
                f"save {str(image_path.absolute())}; "
                f"quit"
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)  # Timeout after 5 minutes
            
            if not image_path.exists():
                logging.error(f"Image not created at {image_path}")
                logging.error(f"PyMOL output: {result.stdout}\n{result.stderr}")
                
            return True
            
        except subprocess.TimeoutExpired as te:
            logging.error(f"Timeout during visualization for {peptide_id}: {str(te)}")
            return False
        except Exception as e:
            logging.error(f"Error visualizing {peptide_id}: {str(e)}")
            return False

if __name__ == "__main__":
    pipeline = PeptideFoldingPipeline(
        fasta_dir="../peptide_fastas",
        colabfold_path="ColabFold",
        database_path="/storage-1/peptide_templating/databases",
        output_base="folding_outputs"
    )
    pipeline.process_all_peptides()