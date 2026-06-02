"""Generate robustness variants (additive noise and downsampling) for RR windows and recompute HRV features.

Usage: python scripts/generate_robustness_variants.py --base data/processed/sample --noise 0 10 20 --downsample 2 3
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.hrv_linear import extract_hrv_features


def add_noise(rr, sigma_ms):
    rr = np.asarray(rr, dtype=float)
    noisy = rr + np.random.randn(*rr.shape) * sigma_ms
    noisy = np.clip(noisy, 300.0, 2000.0)
    return noisy


def downsample_rr(rr, k):
    rr = np.asarray(rr, dtype=float)
    if k <= 1:
        return rr.copy()
    n = len(rr)
    m = n // k
    if m == 0:
        return np.array([np.sum(rr)])
    ds = np.array([np.sum(rr[i*k:(i+1)*k]) for i in range(m)])
    # include remainder as last window
    rem = n % k
    if rem > 0:
        ds = np.concatenate([ds, np.array([np.sum(rr[-rem:])])])
    return ds


def process_folder(base_dir, noise_levels, downsample_factors):
    recs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for rec in recs:
        rr_dir = os.path.join(base_dir, rec, 'rr_windows')
        if not os.path.isdir(rr_dir):
            continue
        out_base = os.path.join(base_dir, rec, 'robustness')
        os.makedirs(out_base, exist_ok=True)
        rr_files = sorted(glob.glob(os.path.join(rr_dir, '*.npy')))
        for f in rr_files:
            name = os.path.splitext(os.path.basename(f))[0]
            rr = np.load(f)
            # noise variants
            for sigma in noise_levels:
                noisy = add_noise(rr, sigma)
                od = os.path.join(out_base, f'noise_{sigma}')
                os.makedirs(od, exist_ok=True)
                np.save(os.path.join(od, f'{name}_rr.npy'), noisy)
                feats = extract_hrv_features(noisy)
                np.save(os.path.join(od, f'{name}_hrv.npy'), feats)
            # downsample variants
            for k in downsample_factors:
                ds = downsample_rr(rr, k)
                od = os.path.join(out_base, f'downsample_{k}')
                os.makedirs(od, exist_ok=True)
                np.save(os.path.join(od, f'{name}_rr.npy'), ds)
                feats = extract_hrv_features(ds)
                np.save(os.path.join(od, f'{name}_hrv.npy'), feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='Base processed sample dir (e.g., data/processed/sample)')
    parser.add_argument('--noise', nargs='*', type=float, default=[0.0], help='Noise sigma values (ms)')
    parser.add_argument('--downsample', nargs='*', type=int, default=[1], help='Downsampling factors')
    args = parser.parse_args()
    process_folder(args.base, args.noise, args.downsample)


if __name__ == '__main__':
    main()
