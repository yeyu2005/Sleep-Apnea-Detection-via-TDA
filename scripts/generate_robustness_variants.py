"""Generate robustness variants (additive noise and downsampling) for RR windows and recompute HRV features."""
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
    if k <= 1: return rr.copy()
    n = len(rr)
    m = n // k
    if m == 0: return np.array([np.sum(rr)])
    ds = np.array([np.sum(rr[i*k:(i+1)*k]) for i in range(m)])
    rem = n % k
    if rem > 0: ds = np.concatenate([ds, np.array([np.sum(rr[-rem:])])])
    return ds

def process_folder(rr_dir, out_base, noise_levels, downsample_factors):
    os.makedirs(out_base, exist_ok=True)
    rr_files = sorted(glob.glob(os.path.join(rr_dir, '*.npy')))
    
    if not rr_files:
        print(f"No files found in {rr_dir}")
        return
        
    print(f"Found {len(rr_files)} RR windows. Generating variants... This may take a moment.")
    
    count = 0
    for f in rr_files:
        name = os.path.splitext(os.path.basename(f))[0]
        rr = np.load(f)
        
        # Add Noise Variants
        for sigma in noise_levels:
            noisy = add_noise(rr, sigma)
            od = os.path.join(out_base, f'noise_{sigma}')
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(od, f'{name}_rr.npy'), noisy)
            np.save(os.path.join(od, f'{name}_hrv.npy'), extract_hrv_features(noisy))
            
        # Add Downsample Variants
        for k in downsample_factors:
            ds = downsample_rr(rr, k)
            od = os.path.join(out_base, f'downsample_{k}')
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(od, f'{name}_rr.npy'), ds)
            np.save(os.path.join(od, f'{name}_hrv.npy'), extract_hrv_features(ds))
            
        count += 1
        if count % 2000 == 0:
            print(f"Processed {count} / {len(rr_files)} windows...")
            
    print(f"Success! Generated robustness variants for {count} windows.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr_dir', required=True, help='Path to original RR windows')
    parser.add_argument('--out_dir', required=True, help='Output path for robust variants')
    parser.add_argument('--noise', nargs='*', type=float, default=[0.0])
    parser.add_argument('--downsample', nargs='*', type=int, default=[1])
    args = parser.parse_args()
    process_folder(args.rr_dir, args.out_dir, args.noise, args.downsample)

if __name__ == '__main__':
    main()
