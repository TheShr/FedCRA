#!/usr/bin/env python
"""
Diagnostic script to check dataset column names and labels.
Helps identify correct n_features and label_name for config.
"""

import pandas as pd
from pathlib import Path
import sys

DATASET_DIR = Path('/workspace/fed_iomt/dataset/data/iomt_traffic')

def analyze_dataset(file_path):
    print(f"\n{'='*70}")
    print(f"Dataset: {file_path.name}")
    print(f"{'='*70}")
    
    try:
        df = pd.read_csv(file_path, nrows=100)  # Read first 100 rows for speed
        
        print(f"\nShape: {df.shape}")
        print(f"\nColumn Names ({len(df.columns)} total):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nFirst row:")
        print(df.iloc[0].to_dict())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.any() else "  None")
        
        # Try to identify label column
        print(f"\nPossible label columns (non-numeric or categorical):")
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 100:
                unique_count = df[col].nunique()
                print(f"  - {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {df[col].unique()[:10].tolist()}")
        
        # Recommended configuration
        print(f"\n{'Recommended Config:':^70}")
        print(f"  n_features: {len(df.columns) - 1}  # (one column for label)")
        print(f"  label_name: 'Label'  # (change if different)")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)
    
    datasets = list(DATASET_DIR.glob('*.bz2')) + list(DATASET_DIR.glob('*.csv'))
    
    if not datasets:
        print(f"No CSV or BZ2 files found in {DATASET_DIR}")
        sys.exit(1)
    
    print(f"Found {len(datasets)} dataset(s)")
    
    for dataset_file in sorted(datasets):
        analyze_dataset(dataset_file)
    
    print(f"\n{'='*70}")
    print("Analysis complete. Use findings to update config.yaml")
    print(f"{'='*70}\n")
