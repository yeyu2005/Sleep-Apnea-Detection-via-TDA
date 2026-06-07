"""Compute median Takens parameters (tau and embedding dim) over RR windows in a folder.
Saves results into `configs/tda_config.yaml` (updates tau and embedding_dim).
"""
import os
import sys
import glob
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.takens import estimate_takens_params


def compute_median(rr_dir, sample_n=100):
    all_files = sorted(glob.glob(os.path.join(rr_dir, '*.npy')))
    train_files = [f for f in all_files if not os.path.basename(f).startswith('x')]
    if len(train_files) == 0:
        raise RuntimeError('No rr windows found in ' + rr_dir)
    files = train_files[:sample_n]
    taus = []
    dims = []
    for f in files:
        x = np.load(f)
        try:
            tau, dim = estimate_takens_params(x, max_lag=50, max_dim=10)
            taus.append(tau)
            dims.append(dim)
        except Exception:
            continue
    if len(taus) == 0:
        raise RuntimeError('No valid estimates')
    return int(np.median(taus)), int(np.median(dims))


def main():
    rr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'rr_windows'))
    tda_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'tda_config.yaml'))
    tau_med, dim_med = compute_median(rr_dir)
    print('Median tau, dim:', tau_med, dim_med)
    with open(tda_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['tau'] = int(tau_med)
    cfg['embedding_dim'] = int(dim_med)
    with open(tda_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)
    print('Updated', tda_cfg)


if __name__ == '__main__':
    main()
    
