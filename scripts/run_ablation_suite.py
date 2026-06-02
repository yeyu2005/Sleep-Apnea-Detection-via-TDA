"""Run ablation suite over RF / BiLSTM / TST with HRV / TDA / Fusion features.

Generates per-ablation results under `outdir/<ablation_name>/` and saves
fold-level metrics to JSON.

Usage:
  python scripts/run_ablation_suite.py --hrv_dir data/processed/hrv --tda_dir data/processed/tda \
      --labels configs/training_labels.csv --outdir experiments/ablations --seq_len 5
"""
import os
import argparse
import json
import tempfile
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.train import run_group_cv


ABLATIONS = [
    ('rf', None, 'hrv'),
    ('rf', None, 'tda'),
    ('rf', None, 'fusion'),
    ('nn', 'bilstm', 'hrv'),
    ('nn', 'bilstm', 'tda'),
    ('nn', 'bilstm', 'fusion'),
    ('nn', 'tst', 'hrv'),
    ('nn', 'tst', 'tda'),
    ('nn', 'tst', 'fusion'),
]


def make_zero_features(filenames, out_dir, dim, suffix):
    os.makedirs(out_dir, exist_ok=True)
    for fn in filenames:
        base = os.path.basename(fn)
        fname = os.path.join(out_dir, base)
        np.save(fname, np.zeros(dim, dtype=float))


def discover_shape(sample_dir, default_dim):
    import glob
    files = glob.glob(os.path.join(sample_dir, '*.npy'))
    if not files:
        return default_dim
    x = np.load(files[0])
    return x.shape[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hrv_dir', required=True)
    parser.add_argument('--tda_dir', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seq_len', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # get label filenames
    import pandas as pd
    labels = pd.read_csv(args.labels)
    fnames = labels['filename'].tolist()

    # discover dims
    hrv_dim = discover_shape(args.hrv_dir, default_dim=9)
    tda_dim = discover_shape(args.tda_dir, default_dim=256)

    results = {}
    for model_type, arch, mode in ABLATIONS:
        name = f"{model_type}_{arch or 'mlp'}_{mode}"
        print(name)
        out = os.path.join(args.outdir, name)
        os.makedirs(out, exist_ok=True)

        # prepare feature dirs
        tmpdir = tempfile.mkdtemp(prefix='ablation_')
        hrv_used = args.hrv_dir
        tda_used = args.tda_dir
        if mode == 'hrv':
            # create zeroed PI dir
            ztda = os.path.join(tmpdir, 'tda_zero')
            make_zero_features(fnames, ztda, tda_dim, '_pi.npy')
            tda_used = ztda
        elif mode == 'tda':
            zhrv = os.path.join(tmpdir, 'hrv_zero')
            make_zero_features(fnames, zhrv, hrv_dim, '_hrv.npy')
            hrv_used = zhrv

        config = {'arch': arch or 'mlp', 'seq_len': args.seq_len}
        try:
            res = run_group_cv(hrv_used, tda_used, args.labels, out, model_type=model_type, n_splits=5, config=config)
            # run_group_cv returns (fold_window, fold_patient, agg dict)
            results[name] = res[2]
        except Exception as e:
            results[name] = {'error': str(e)}
            print("no cv")

    # save summary
    with open(os.path.join(args.outdir, 'ablation_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Ablation suite complete. Summary written to', os.path.join(args.outdir, 'ablation_summary.json'))


if __name__ == '__main__':
    main()
