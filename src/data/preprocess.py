"""Preprocessing utilities: read WFDB records, filter, detect R-peaks, save interim outputs.

Usage:
    python -m src.data.preprocess --record <record_base_path> --outdir <path>

Saves: <outdir>/<record>_interim.npz containing filtered signal, rpeaks (indices), rr_intervals (ms)
"""
import argparse
import os
import numpy as np
from scipy import signal

try:
    import wfdb
except Exception:
    wfdb = None

try:
    import neurokit2 as nk
except Exception:
    nk = None


def bandpass_filter(sig, fs, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    lowc = low / nyq
    highc = high / nyq
    b, a = signal.butter(order, [lowc, highc], btype='band')
    return signal.filtfilt(b, a, sig)


def notch_filter(sig, fs, freq=50.0, q=30.0):
    # notch at freq
    w0 = freq / (fs / 2)
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, sig)


def process_record(record_path, outdir, config):
    record_name = os.path.splitext(os.path.basename(record_path))[0]
    print(f"Processing {record_name}")
    if wfdb is None:
        raise RuntimeError("wfdb is required to read PhysioNet records")
    if nk is None:
        raise RuntimeError("neurokit2 is required for R-peak detection")

    record_dir = os.path.dirname(record_path)
    base = os.path.splitext(os.path.basename(record_path))[0]
    # use wfdb.rdrecord to read
    rec = wfdb.rdrecord(os.path.join(record_dir, base))
    sig = rec.p_signal[:,0]  # Extracts the 1D signal array
    fs = config.get('fs', 100)

    # filtering
    sig_f = bandpass_filter(sig, fs, config.get('bandpass_low', 0.5), config.get('bandpass_high', 40))
    sig_f = notch_filter(sig_f, fs, config.get('notch_freq', 50))

    # R-peak detection via neurokit2
    _, rpeaks = nk.ecg_peaks(sig_f, sampling_rate=fs)
    # neurokit2 returns dict with 'ECG_R_Peaks'
    rpeak_idx = rpeaks.get('ECG_R_Peaks', [])
    rpeak_times = np.array(rpeak_idx) / float(fs)
    rr_intervals = np.diff(rpeak_times) * 1000.0  # ms

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{base}_interim.npz")
    np.savez(outpath, signal=sig, filtered=sig_f, rpeaks=rpeak_idx, rr=rr_intervals)
    print(f"Saved interim data to {outpath}")
    return outpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True, help='Path to record base (e.g., a01) or wfdb record file')
    parser.add_argument('--outdir', default='data/interim')
    parser.add_argument('--config', default='configs/preprocess_config.yaml')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    process_record(args.record, args.outdir, cfg)
