import os
import glob
import sys

# make package importable when running script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocess import process_record
import yaml

# Load your config
with open('configs/preprocess_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Find all .dat files in the raw directory
raw_files = glob.glob('data/raw/*.dat')

for file_path in raw_files:
    # Remove the .dat extension to get the record base path
    record_base = os.path.splitext(file_path)[0]
    try:
        process_record(record_base, 'data/interim', cfg)
    except Exception as e:
        print(f"Failed to process {record_base}: {e}")