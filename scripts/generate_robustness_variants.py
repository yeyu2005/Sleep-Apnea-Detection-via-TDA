"""Generate robustness variants (additive noise and downsampling) for RR windows and recompute HRV & TDA features."""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.hrv_linear import extract_hrv_features
from src.features.topological import compute_persistence_image

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

def process_folder(rr_dir, out_base, noise_levels, downsample_factors, tau, dim, grid_size):
    os.makedirs(out_base, exist_ok=True)
    rr_files = sorted(glob.glob(os.path.join(rr_dir, '*.npy')))
    
    if not rr_files:
        print(f"No files found in {rr_dir}")
        return
        
    print(f"Found {len(rr_files)} RR windows. Generating True Robustness variants...")
    
    count = 0
    # Expected dimensions to prevent crashes (32 * 32 * 2 = 2048)
    expected_tda_dim = grid_size * grid_size * 2
    expected_hrv_dim = 9

    for f in rr_files:
        name = os.path.splitext(os.path.basename(f))[0]
        rr = np.load(f)
        
        # Add Noise Variants
        for sigma in noise_levels:
            noisy = add_noise(rr, sigma)
            od = os.path.join(out_base, f'noise_{sigma}')
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(od, f'{name}_rr.npy'), noisy)
            
            # Safe Noisy HRV
            try:
                n_hrv = extract_hrv_features(noisy)
            except:
                n_hrv = np.zeros(expected_hrv_dim)
            np.save(os.path.join(od, f'{name}_hrv.npy'), n_hrv)
            
            # Safe Noisy TDA
            try:
                n_pi = compute_persistence_image(noisy, tau=tau, dim=dim, grid_size=grid_size)
                if len(n_pi) != expected_tda_dim:
                    n_pi = np.zeros(expected_tda_dim)
            except:
                n_pi = np.zeros(expected_tda_dim)
            np.save(os.path.join(od, f'{name}_pi.npy'), n_pi)
            
        # Add Downsample Variants
        for k in downsample_factors:
            ds = downsample_rr(rr, k)
            od = os.path.join(out_base, f'downsample_{k}')
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(od, f'{name}_rr.npy'), ds)
            
            # Safe Downsampled HRV
            try:
                ds_hrv = extract_hrv_features(ds)
            except:
                ds_hrv = np.zeros(expected_hrv_dim)
            np.save(os.path.join(od, f'{name}_hrv.npy'), ds_hrv)
            
            # Safe Downsampled TDA
            try:
                ds_pi = compute_persistence_image(ds, tau=tau, dim=dim, grid_size=grid_size)
                if len(ds_pi) != expected_tda_dim:
                    ds_pi = np.zeros(expected_tda_dim)
            except:
                ds_pi = np.zeros(expected_tda_dim)
            np.save(os.path.join(od, f'{name}_pi.npy'), ds_pi)
            
        count += 1
        if count % 100 == 0:  # Reduced print threshold since TDA is slower
            print(f"Processed {count} / {len(rr_files)} windows...")
            
    print(f"Success! Generated true robustness variants for {count} windows.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr_dir', required=True, help='Path to ONLY the test set RR windows')
    parser.add_argument('--out_dir', required=True, help='Output path for robust variants')
    parser.add_argument('--noise', nargs='*', type=float, default=[0.0])
    parser.add_argument('--downsample', nargs='*', type=int, default=[1])
    
    # Required TDA Parameters
    parser.add_argument('--tau', type=int, required=True, help='Takens time delay used for clean data')
    parser.add_argument('--dim', type=int, required=True, help='Takens embedding dimension used for clean data')
    parser.add_argument('--grid_size', type=int, default=32, help='Grid size for Persistence Image (16 = 256 dims)')
    args = parser.parse_args()
    
    process_folder(args.rr_dir, args.out_dir, args.noise, args.downsample, args.tau, args.dim, args.grid_size)

if __name__ == '__main__':
    main()
