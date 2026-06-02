"""Master script to process interim RR data into ML features and generate label CSVs.
Extracts RR windows, applies interpolation, computes HRV, reads .apn labels, and saves manifests.

Usage: python scripts/extract_all_features.py --interim data/interim --raw data/raw --out data/processed
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import wfdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.interpolate import clean_and_interpolate
from src.features.hrv_linear import extract_hrv_features

def split_rr_windows(rr_ms, window_s=60.0):
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0: return []
    t = np.cumsum(rr) / 1000.0
    windows, start, i = [], 0.0, 0
    while start < t[-1]:
        mask = (t >= start) & (t < start + window_s)
        idx = np.where(mask)[0]
        windows.append(rr[idx] if idx.size > 0 else np.array([]))
        start += window_s
    return windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim', default='data/interim')
    parser.add_argument('--raw', default='data/raw', help='Path to raw .apn files')
    parser.add_argument('--out', default='data/processed')
    args = parser.parse_args()

    hrv_dir = os.path.join(args.out, 'hrv')
    rr_dir = os.path.join(args.out, 'rr_windows')
    os.makedirs(hrv_dir, exist_ok=True)
    os.makedirs(rr_dir, exist_ok=True)

    interim_files = glob.glob(os.path.join(args.interim, '*_interim.npz'))
    all_rows = []

    # FIX: Changed loop variable to 'npzpath' to match the rest of the block
    for npzpath in sorted(interim_files):
        recname = os.path.basename(npzpath).replace('_interim.npz', '')
        data = np.load(npzpath, allow_pickle=True)
        rr = data.get('rr')
        if rr is None or len(rr) == 0:
            continue

        # Read Apnea Annotations from PhysioNet
        apn_path = os.path.join(args.raw, recname)
        try:
            ann = wfdb.rdann(apn_path, 'apn')
            labels = [1 if sym == 'A' else 0 for sym in ann.symbol]
        except Exception as e:
            print(f"Skipping {recname}: Missing/corrupt .apn annotations ({e})")
            continue

        windows = split_rr_windows(rr)
        
        # Match windows to labels
        for i, w in enumerate(windows):
            if i >= len(labels): break # Stop if windows exceed annotations
            
            w_clean = clean_and_interpolate(w)
            if w_clean is None or len(w_clean) < 30: 
                continue # Discard unrecoverable/empty segments
                
            basename = f"{recname}_win{i:04d}"
            
            # Save RR window for TDA
            np.save(os.path.join(rr_dir, f"{basename}.npy"), w_clean)
            
            # Save HRV features
            feats = extract_hrv_features(w_clean)
            np.save(os.path.join(hrv_dir, f"{basename}_hrv.npy"), feats)
            
            all_rows.append({
                'filename': basename, 
                'label': labels[i], 
                'group': recname # patient_id for GroupKFold
            })
        print(f"Processed {recname}: {len(windows)} windows")

    # Split and save manifests strictly by PhysioNet rules
    df = pd.DataFrame(all_rows)
    os.makedirs('configs', exist_ok=True)
    
    train_df = df[df['group'].str.match(r"^[a-c]\d{2}$")]
    test_df = df[df['group'].str.match(r"^x\d{2}$")]
    
    train_df.to_csv('configs/training_labels.csv', index=False)
    test_df.to_csv('configs/test_labels.csv', index=False)
    print(f"\nSaved {len(train_df)} training windows and {len(test_df)} test windows.")

if __name__ == '__main__':
    main()