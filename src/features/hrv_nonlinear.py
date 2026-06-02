"""Nonlinear HRV feature extractors (DFA, entropy wrappers).

This module currently provides a simple wrapper for sample entropy and a placeholder
for DFA (returns NaN). It can be extended to use `nolds` or `neurokit2` if available.
"""
import numpy as np
from .hrv_linear import sample_entropy


def _dfa(signal, scales=None):
    """Detrended Fluctuation Analysis (returns alpha exponent).

    signal: 1D array
    scales: iterable of window sizes (ints). If None, choose logarithmically spaced scales.
    """
    x = np.asarray(signal, dtype=float)
    N = x.size
    if N < 10:
        return np.nan
    # integrate (profile)
    y = np.cumsum(x - np.mean(x))
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(4, N//4)), num=10)).astype(int))
    F = []
    for s in scales:
        if s < 4:
            continue
        n_segments = N // s
        if n_segments < 2:
            continue
        rs = np.array([y[i*s:(i+1)*s] for i in range(n_segments)])
        # detrend each segment
        segs = rs
        xidx = np.arange(s)
        rms = []
        for seg in segs:
            # linear fit
            A = np.vstack([xidx, np.ones_like(xidx)]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, seg, rcond=None)
            trend = A.dot(coeffs)
            diff = seg - trend
            rms.append(np.sqrt(np.mean(diff**2)))
        F.append(np.sqrt(np.mean(np.array(rms)**2)))
    if len(F) < 2:
        return np.nan
    scales_used = np.array([s for s in scales if s >= 4 and (N // s) >= 2])
    if len(scales_used) != len(F):
        # align lengths
        minlen = min(len(scales_used), len(F))
        scales_used = scales_used[:minlen]
        F = F[:minlen]
    # linear slope of log-log
    coeffs = np.polyfit(np.log(scales_used), np.log(F), 1)
    alpha = coeffs[0]
    return float(alpha)


def extract_nonlinear_features(rr_ms):
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return np.array([np.nan, np.nan])
    rr_s = rr / 1000.0
    hr_inst = 60.0 / rr_s
    sampen = sample_entropy(hr_inst)
    dfa = _dfa(hr_inst)
    return np.array([sampen, dfa])


__all__ = ['extract_nonlinear_features']
