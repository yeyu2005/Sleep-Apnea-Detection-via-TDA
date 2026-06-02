"""RR cleaning and cubic spline interpolation for small gaps.

Function accepts either ``clean_and_interpolate(rr_intervals, ...)`` or the
legacy signature ``clean_and_interpolate(rr_times, rr_intervals, ...)`` for
backwards compatibility. Returns a filled RR array (ms) or ``None`` when the
segment contains an unrecoverable gap (>= 3 missing beats).
"""
import numpy as np
import pandas as pd


def clean_and_interpolate(rr_times_or_rr_intervals, rr_intervals=None, min_ms=300, max_ms=2000):
    # Backwards-compatible argument handling
    if rr_intervals is None:
        rr = np.array(rr_times_or_rr_intervals, dtype=float)
    else:
        # legacy call where first arg is rr_times (unused) and second is rr_intervals
        rr = np.array(rr_intervals, dtype=float)
    mask_bad = (rr < min_ms) | (rr > max_ms) | np.isnan(rr)
    rr[mask_bad] = np.nan

    # find consecutive NaN runs
    is_nan = np.isnan(rr)
    nan_runs = []
    start = None
    for i, v in enumerate(is_nan):
        if v and start is None:
            start = i
        if not v and start is not None:
            nan_runs.append((start, i-1))
            start = None
    if start is not None:
        nan_runs.append((start, len(rr)-1))

    # if run length >=3, drop that segment (mark as invalid by returning None for that window)
    for s,e in nan_runs:
        if (e - s + 1) >= 3:
            # indicate unrecoverable
            return None

    # cubic spline interpolate small gaps
    series = pd.Series(rr)
    #A cubic spline mathematically requires at least 4 valid data points.
    # If the window is too corrupted to have 4 beats, discard it.
    if series.dropna().shape[0] < 4:
        return None
        
    try:
        rr_filled = series.interpolate(method='cubic', limit_direction='both')
    except ValueError:
        # Failsafe: if the mathematical boundaries still fail on a weird edge case
        return None
        
    return np.array(rr_filled)
