import os
import sys
import numpy as np

# ensure project src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.hrv_linear import extract_hrv_features
from src.features.hrv_nonlinear import extract_nonlinear_features


def test_extract_hrv_features_basic():
    rng = np.random.RandomState(0)
    # simulate 5 minutes of RR at 75 bpm ~ 800 ms
    n_beats = 60 * 5 * 1.25
    rr_ms = 800 + 30 * rng.randn(int(n_beats))
    rr_ms = np.clip(rr_ms, 300, 2000)
    feats = extract_hrv_features(rr_ms)
    assert feats.shape[0] == 9
    assert np.isfinite(feats[0])


def test_extract_nonlinear_features_basic():
    rng = np.random.RandomState(1)
    rr_ms = 800 + 30 * rng.randn(600)
    rr_ms = np.clip(rr_ms, 300, 2000)
    feats = extract_nonlinear_features(rr_ms)
    assert feats.shape[0] == 2
