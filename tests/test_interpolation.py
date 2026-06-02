import os
import sys
import numpy as np

# ensure project src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.interpolate import clean_and_interpolate


def test_small_gap_interpolation():
    rr = np.array([800, 810, np.nan, 820, 830])
    out = clean_and_interpolate(None, rr)
    assert out is not None
    assert not np.any(np.isnan(out))


def test_large_gap_rejected():
    rr = np.array([800, np.nan, np.nan, np.nan, 820])
    out = clean_and_interpolate(None, rr)
    assert out is None
