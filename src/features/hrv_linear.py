"""Linear (time- and frequency-domain) HRV feature extraction utilities.

Functions expect RR intervals in milliseconds (array-like).
"""
import numpy as np
from scipy.signal import welch


def sample_entropy(x, m=2, r_factor=0.2):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m + 2:
        return np.nan
    r = r_factor * np.std(x)

    def _count(m):
        nm = N - m + 1
        templates = np.lib.stride_tricks.sliding_window_view(x, m)
        count = 0
        for i in range(nm):
            # Chebyshev distance (max abs diff)
            d = np.max(np.abs(templates - templates[i]), axis=1)
            # exclude self-match
            count += np.sum(d <= r) - 1
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return np.nan
    if A == 0:
        return np.nan
    return -np.log(float(A) / float(B))


def extract_hrv_features(rr_ms, fs_interp=4.0):
    """Compute a compact HRV feature vector from RR intervals (ms).

    Returns numpy array in order:
      [mean_rr_ms, sdnn_ms, rmssd_ms, pnn50, lf_power, hf_power, lf_hf_ratio, total_power, sampen]
    """
    rr = np.asarray(rr_ms, dtype=float)
    
    # FIX 1: Strip out any NaNs or Infs that survived interpolation
    rr = rr[np.isfinite(rr)]
    
    if rr.size < 2:
        return np.array([np.nan] * 9)

    # Time-domain
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if diff_rr.size > 0 else np.nan
    pnn50 = np.mean(np.abs(diff_rr) > 50.0) if diff_rr.size > 0 else np.nan

    # Frequency-domain: interpolate instantaneous HR and compute PSD
    rr_s = rr / 1000.0
    
    beat_timestamps = np.concatenate(([0.0], np.cumsum(rr_s)[:-1]))
    hr_inst = 60.0 / rr_s
    
    # FIX 2: Validate the time range before calling np.arange
    valid_time_range = (beat_timestamps[-1] > beat_timestamps[0]) and np.isfinite(beat_timestamps[-1])
    
    if beat_timestamps.size < 4 or not valid_time_range:
        lf_power = np.nan
        hf_power = np.nan
        lf_hf = np.nan
        total_power = np.nan
    else:
        t_reg = np.arange(beat_timestamps[0], beat_timestamps[-1], 1.0 / fs_interp)
        # interpolate instantaneous heart rate (or RR) onto uniform grid
        hr_reg = np.interp(t_reg, beat_timestamps, hr_inst)
        # remove mean trend
        hr_reg_d = hr_reg - np.mean(hr_reg)
        nperseg = min(256, len(hr_reg_d))
        
        # Ensure nperseg is valid for Welch's method
        if len(hr_reg_d) < 2:
            lf_power, hf_power, lf_hf, total_power = np.nan, np.nan, np.nan, np.nan
        else:
            f, pxx = welch(hr_reg_d, fs=fs_interp, nperseg=nperseg)
            
            # band powers (LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz)
            def band_power(f, pxx, a, b):
                mask = (f >= a) & (f <= b)
                if np.any(mask):
                    return np.trapz(pxx[mask], f[mask])
                return 0.0

            lf_power = band_power(f, pxx, 0.04, 0.15)
            hf_power = band_power(f, pxx, 0.15, 0.4)
            total_power = np.trapz(pxx, f) if pxx.size > 0 else np.nan
            lf_hf = lf_power / hf_power if hf_power > 0 else np.nan

    # Nonlinear (sample entropy) computed on HR series
    try:
        sampen = sample_entropy(hr_inst)
    except Exception:
        sampen = np.nan

    feats = np.array([mean_rr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf, total_power, sampen])
    return feats


__all__ = ['extract_hrv_features', 'sample_entropy']
