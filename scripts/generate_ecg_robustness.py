"""Generate robustness variants at the raw ECG level: additive noise and downsampling.

For each interim .npz produced by preprocessing (containing 'filtered' signal and sampling rate),
this script will:
 - add Gaussian noise (in mV units) to the filtered ECG and recompute R-peaks
 - downsample the filtered ECG to lower sampling rates and recompute R-peaks

Outputs a parallel folder structure under the record directory: `robustness/ecg_noise_0.1/` and `robustness/ecg_down_50/` etc.

Usage:
  python scripts/generate_ecg_robustness.py --interim data/interim --out data/processed/sample --noise 0.1 0.2 --downsample 50 25
"""
import os
import glob
import argparse
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import neurokit2 as nk
except Exception:
    nk = None


def add_ecg_noise(sig, sigma_mv):
    sig = np.asarray(sig, dtype=float)
    noisy = sig + np.random.randn(*sig.shape) * sigma_mv
    return noisy


def downsample_signal(sig, orig_fs, target_fs):
    if target_fs >= orig_fs:
        return sig, orig_fs
    factor = orig_fs / target_fs
    if not float(factor).is_integer():
        # use simple resampling
        import scipy.signal as sps
        num = int(len(sig) * target_fs / orig_fs)
        res = sps.resample(sig, num)
        return res, target_fs
    k = int(factor)
    # simple decimation
    res = sig[::k]
    return res, int(orig_fs // k)


def process_interim_file(npzpath, out_base, noise_levels, downsample_rates):
    data = np.load(npzpath, allow_pickle=True)
    if 'filtered' not in data:
        return 0
    sig = data['filtered']
    fs = int(data.get('fs', 100))
    name = os.path.splitext(os.path.basename(npzpath))[0]
    n_out = 0

    for sigma in noise_levels:
        noisy = add_ecg_noise(sig, sigma)
        # recompute rpeaks
        if nk is None:
            print('neurokit2 missing; skipping ECG-level robustness')
            break
        rpeaks = nk.ecg_peaks(noisy, sampling_rate=fs)[1].get('ECG_R_Peaks', [])
        rtimes = np.array(rpeaks) / float(fs)
        rr = np.diff(rtimes) * 1000.0
        rec_out = os.path.join(out_base, 'robustness', f'ecg_noise_{sigma}')
        os.makedirs(rec_out, exist_ok=True)
        np.savez(os.path.join(rec_out, name + f'_noise_{sigma}.npz'), filtered=noisy, fs=fs, rpeaks=rpeaks, rr=rr)
        n_out += 1

    for target in downsample_rates:
        sig_ds, fs_ds = downsample_signal(sig, fs, int(target))
        if nk is None:
            print('neurokit2 missing; skipping ECG-level robustness downsampling')
            break
        rpeaks = nk.ecg_peaks(sig_ds, sampling_rate=fs_ds)[1].get('ECG_R_Peaks', [])
        rtimes = np.array(rpeaks) / float(fs_ds)
        rr = np.diff(rtimes) * 1000.0
        rec_out = os.path.join(out_base, 'robustness', f'ecg_down_{fs_ds}')
        os.makedirs(rec_out, exist_ok=True)
        np.savez(os.path.join(rec_out, name + f'_down_{fs_ds}.npz'), filtered=sig_ds, fs=fs_ds, rpeaks=rpeaks, rr=rr)
        n_out += 1

    return n_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim', required=True, help='Interim directory with preprocessing npz files')
    parser.add_argument('--out', required=True, help='Base processed output directory (records)')
    parser.add_argument('--noise', nargs='*', type=float, default=[0.1, 0.2], help='Noise sigma values (mV)')
    parser.add_argument('--downsample', nargs='*', type=int, default=[50,25], help='Target sampling rates (Hz)')
    args = parser.parse_args()

    npz_files = glob.glob(os.path.join(args.interim, '*_interim.npz'))
    total = 0
    for p in npz_files:
        recname = os.path.splitext(os.path.basename(p))[0]
        rec_out = os.path.join(args.out, recname)
        os.makedirs(rec_out, exist_ok=True)
        n = process_interim_file(p, rec_out, args.noise, args.downsample)
        print(f'Processed {p}: generated {n} robustness variants')
        total += n
    print('Total robustness variants:', total)


if __name__ == '__main__':
    main()
