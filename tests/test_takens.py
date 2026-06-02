import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.takens import average_mutual_information, first_minimum_ami, false_nearest_neighbors, estimate_takens_params


def test_ami_on_sine():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 1.0 * t)
    amis = average_mutual_information(x, max_lag=20, bins=32)
    assert amis.size == 20
    tau = first_minimum_ami(x, max_lag=20, bins=32)
    assert isinstance(tau, int)


def test_fnn_on_sine():
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 0.5 * t)
    tau, dim = estimate_takens_params(x, max_lag=20, max_dim=6)
    assert isinstance(tau, int)
    assert isinstance(dim, int)
