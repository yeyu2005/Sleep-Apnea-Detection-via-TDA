import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.generate_robustness_variants import add_noise, downsample_rr


def test_add_noise_and_downsample():
    rng = np.random.RandomState(0)
    rr = np.abs(800 + 30 * rng.randn(100))
    noisy = add_noise(rr, 50.0)
    assert noisy.shape == rr.shape
    ds = downsample_rr(rr, 3)
    assert ds.size <= rr.size
