"""Prepare strict training/test record manifests for PhysioNet Apnea-ECG.

This script generates two CSV files under `configs/`:
 - `configs/training_records.csv` containing the official training record IDs (a01-a20, b01-b05, c01-c10)
 - `configs/test_records.csv` containing the remaining records (x01-x35)

It strictly ignores metadata files and alternative signal variants (e.g., a01r, a01er).
"""
import os
import argparse
import csv
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to PhysioNet dataset folder')
    parser.add_argument('--out', default='configs')
    args = parser.parse_args()

    files = os.listdir(args.dataset)
    # Get all unique base names
    bases = sorted({os.path.splitext(f)[0] for f in files if os.path.splitext(f)[0]})

    # Strict regex matching
    # ^ : starts with
    # [a-c] : letter a, b, or c
    # \d{2} : exactly two digits
    # $ : ends immediately (prevents a01r, a01er from matching)
    train_pattern = re.compile(r"^[a-c]\d{2}$")
    test_pattern = re.compile(r"^x\d{2}$")

    train = [b for b in bases if train_pattern.match(b)]
    test = [b for b in bases if test_pattern.match(b)]

    os.makedirs(args.out, exist_ok=True)
    train_csv = os.path.join(args.out, 'training_records.csv')
    test_csv = os.path.join(args.out, 'test_records.csv')

    with open(train_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record'])
        for r in train:
            w.writerow([r])

    with open(test_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record'])
        for r in test:
            w.writerow([r])

    print(f"Wrote {len(train)} training records to {train_csv}")
    print(f"Wrote {len(test)} test records to {test_csv}")

if __name__ == '__main__':
    main()