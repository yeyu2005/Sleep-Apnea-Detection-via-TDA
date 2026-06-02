"""Analyze robustness variants generated under `robustness/` folders and produce summary CSV.

This script compares HRV features of original windows vs variants and writes a CSV summary.
"""
import os
import glob
import numpy as np
import pandas as pd


def analyze(base_dir, out_csv):
    rows = []
    recs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for rec in recs:
        orig_hrv_dir = os.path.join(base_dir, rec, 'hrv_feats')
        rob_dir = os.path.join(base_dir, rec, 'robustness')
        if not os.path.isdir(orig_hrv_dir) or not os.path.isdir(rob_dir):
            continue
        orig_files = sorted(glob.glob(os.path.join(orig_hrv_dir, '*_hrv.npy')))
        # map basename to features
        orig_map = {os.path.basename(f).split('_hrv.npy')[0]: np.load(f) for f in orig_files}
        for variant in os.listdir(rob_dir):
            vpath = os.path.join(rob_dir, variant)
            if not os.path.isdir(vpath):
                continue
            v_files = sorted(glob.glob(os.path.join(vpath, '*_hrv.npy')))
            for vf in v_files:
                base = os.path.basename(vf).split('_hrv.npy')[0]
                if base in orig_map:
                    orig = orig_map[base]
                    var = np.load(vf)
                    diff = var - orig
                    rows.append({'record': rec, 'variant': variant, 'window': base, 'orig_mean_rr': float(orig[0]), 'var_mean_rr': float(var[0]), 'delta_mean_rr': float(diff[0])})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return out_csv
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', default='robustness_summary.csv')
    args = parser.parse_args()
    res = analyze(args.base, args.out)
    if res:
        print('Wrote', res)
    else:
        print('No robustness data found')
