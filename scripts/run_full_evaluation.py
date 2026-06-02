"""Run full GroupKFold cross-validation or a held-out split evaluation.

Example usage:
  python scripts/run_full_evaluation.py --hrv_dir data/processed/hrv --tda_dir data/processed/tda \
    --labels configs/training_labels.csv --outdir experiments/full_eval --model rf --n_splits 5
"""
import argparse
import os
import yaml
import numpy as np

sys_config = {}
try:
    from src.training.train import run_group_cv
except Exception:
    # allow running from repository root when invoked as script
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.training.train import run_group_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hrv_dir', required=True)
    parser.add_argument('--tda_dir', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--model', choices=['rf','nn'], default='rf')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--arch', choices=['mlp','bilstm','tst'], default='mlp')
    parser.add_argument('--seq_len', type=int, default=None)
    args = parser.parse_args()

    config = {'epochs':50, 'batch_size':64, 'lr':1e-4, 'wd':1e-4, 'patience':5, 'arch': args.arch, 'seq_len': args.seq_len}
    os.makedirs(args.outdir, exist_ok=True)
    run_group_cv(args.hrv_dir, args.tda_dir, args.labels, args.outdir, model_type=args.model, n_splits=args.n_splits, config=config)


if __name__ == '__main__':
    main()
