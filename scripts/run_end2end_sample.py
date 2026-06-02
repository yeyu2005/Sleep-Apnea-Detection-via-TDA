"""Small end-to-end runner for two records: preprocessing -> windowing -> HRV feats -> PI cache.

Usage: python scripts/run_end2end_sample.py
"""
import os
import sys
import glob
import numpy as np

# make package importable when running script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.hrv_linear import extract_hrv_features
from src.features.topological import cache_persistence_images
from src.data.interpolate import clean_and_interpolate


def split_rr_windows(rr_ms, window_s=60.0):
    # rr_ms: 1D array of RR intervals in ms
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return []
    t = np.cumsum(rr) / 1000.0
    windows = []
    start = 0.0
    end = window_s
    i = 0
    while start < t[-1]:
        mask = (t >= start) & (t < end)
        idx = np.where(mask)[0]
        if idx.size > 0:
            windows.append(rr[idx])
        start += window_s
        end += window_s
        i += 1
    return windows


def process_interim_npz(npzpath, out_base):
    data = np.load(npzpath, allow_pickle=True)
    rr = data.get('rr')
    if rr is None:
        return 0
    rr_windows = split_rr_windows(rr)
    rr_dir = os.path.join(out_base, 'rr_windows')
    hrvs_dir = os.path.join(out_base, 'hrv_feats')
    os.makedirs(rr_dir, exist_ok=True)
    os.makedirs(hrvs_dir, exist_ok=True)
    for i, w in enumerate(rr_windows):
        # clean and interpolate small gaps before saving
        w_clean = clean_and_interpolate(w)
        if w_clean is None:
            continue
        fname = os.path.join(rr_dir, f'win_{i:04d}.npy')
        np.save(fname, w_clean)
        feats = extract_hrv_features(w_clean)
        np.save(os.path.join(hrvs_dir, f'win_{i:04d}_hrv.npy'), feats)
    return len(rr_windows)


def main():
    ds = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    ds = os.path.abspath(ds)
    records = ['a01.dat', 'a02.dat']
    interim_out = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim')
    interim_out = os.path.abspath(interim_out)
    processed_base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sample')
    os.makedirs(processed_base, exist_ok=True)

    from src.data.preprocess import process_record

    total_windows = 0
    for rec in records:
        recpath = os.path.join(ds, rec)
        recname = os.path.splitext(rec)[0]
        outdir = os.path.join(interim_out)
        try:
            npz = process_record(recpath, outdir, {'fs':100, 'bandpass_low':0.5, 'bandpass_high':40, 'notch_freq':50})
        except Exception as e:
            # fallback: synthetic RR
            print(f'Warning: preprocess failed for {rec}: {e}. Using synthetic RR.')
            rr = np.abs(800 + 30 * np.random.randn(6000))
            npz = os.path.join(outdir, f'{recname}_interim.npz')
            os.makedirs(outdir, exist_ok=True)
            np.savez(npz, signal=np.zeros(10), filtered=np.zeros(10), rpeaks=np.arange(0,10), rr=rr)

        dest = os.path.join(processed_base, recname)
        os.makedirs(dest, exist_ok=True)
        n = process_interim_npz(npz, dest)
        print(f'Record {recname}: produced {n} RR windows')
        total_windows += n

    # cache persistence images from the first record's rr_windows
    first_rr_dir = os.path.join(processed_base, 'a01', 'rr_windows')
    pi_out = os.path.join(processed_base, 'a01', 'tda')
    try:
        cache_persistence_images(first_rr_dir, pi_out, tau=1, dim=3, grid_size=16, sigma=0.5)
        print('Cached persistence images to', pi_out)
    except RuntimeError as e:
        print('Skipping PI caching:', e)

    print('Total windows processed:', total_windows)


if __name__ == '__main__':
    main()
