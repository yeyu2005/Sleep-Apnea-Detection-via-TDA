"""Analyze robustness variants against original features and produce a summary CSV."""
import os
import glob
import numpy as np
import pandas as pd

def analyze(orig_hrv_dir, rob_dir, out_csv):
    rows = []
    
    print(f"Loading original HRV features from {orig_hrv_dir}...")
    orig_files = sorted(glob.glob(os.path.join(orig_hrv_dir, '*_hrv.npy')))
    orig_map = {os.path.basename(f).replace('_hrv.npy', ''): np.load(f) for f in orig_files}
    
    print(f"Found {len(orig_map)} original HRV arrays. Analyzing drift...")

    for variant in os.listdir(rob_dir):
        vpath = os.path.join(rob_dir, variant)
        if not os.path.isdir(vpath): continue
        
        v_files = sorted(glob.glob(os.path.join(vpath, '*_hrv.npy')))
        print(f" - Checking variant: {variant} ({len(v_files)} files)")
        
        for vf in v_files:
            base = os.path.basename(vf).replace('_hrv.npy', '')
            if base in orig_map:
                orig = orig_map[base]
                var = np.load(vf)
                diff = var - orig
                rec = base.split('_')[0] # extract 'a01' from 'a01_win0001'
                
                rows.append({
                    'record': rec, 
                    'variant': variant, 
                    'window': base, 
                    'orig_mean_rr': float(orig[0]), 
                    'var_mean_rr': float(var[0]), 
                    'delta_mean_rr': float(diff[0]) # Drift!
                })
                
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return out_csv
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_hrv', required=True, help="Path to original HRV features")
    parser.add_argument('--rob_dir', required=True, help="Path to robustness output folder")
    parser.add_argument('--out', default='robustness_summary.csv')
    args = parser.parse_args()
    
    res = analyze(args.orig_hrv, args.rob_dir, args.out)
    if res:
        print(f"\nFinished! Wrote analysis to {res}")
    else:
        print("No robustness data found.")