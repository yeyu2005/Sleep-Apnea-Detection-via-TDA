"""Utilities to estimate Takens embedding parameters: time delay (tau) via
average mutual information (AMI) and embedding dimension via False Nearest Neighbors (FNN).

Functions operate on 1D time series (e.g., RR intervals or HR series).
"""
import numpy as np
from scipy.stats import entropy


def _hist_prob(x, bins=64):
    p, _ = np.histogram(x, bins=bins, density=True)
    p = p.astype(float)
    p = p[p > 0]
    return p


def mutual_information(x, y, bins=64):
    # compute MI between x and y using histogram estimator
    jhist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    pxy = jhist / np.sum(jhist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    denom = np.outer(px, py)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz]))
    return float(mi)


def average_mutual_information(x, max_lag=50, bins=64):
    x = np.asarray(x, dtype=float)
    n = len(x)
    lags = min(max_lag, max(1, n - 2))
    amis = []
    for lag in range(1, lags + 1):
        mi = mutual_information(x[:-lag], x[lag:], bins=bins)
        amis.append(mi)
    return np.array(amis)


def first_minimum_ami(x, max_lag=50, bins=64):
    amis = average_mutual_information(x, max_lag=max_lag, bins=bins)
    if amis.size == 0:
        return 1
    # find first local minimum
    for i in range(1, len(amis) - 1):
        if amis[i] < amis[i - 1] and amis[i] <= amis[i + 1]:
            return i + 1
    # fallback: argmin
    return int(np.argmin(amis) + 1)


def false_nearest_neighbors(x, max_dim=10, tau=1, Rtol=15.0, Atol=2.0):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < (max_dim + 1) * tau:
        return 1
    def embed(dim):
        M = N - (dim - 1) * tau
        X = np.empty((M, dim))
        for i in range(dim):
            X[:, i] = x[i * tau : i * tau + M]
        return X

    for dim in range(1, max_dim + 1):
        X = embed(dim)
        M = X.shape[0]
        # nearest neighbor indices
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=2).fit(X)
        dists, inds = nn.kneighbors(X, n_neighbors=2)
        R = dists[:, 1]
        if dim == max_dim:
            return max_dim
        # extend embedding
        Xp = embed(dim + 1)
        Mp = Xp.shape[0]
        # only consider rows that have a corresponding row in the extended embedding
        valid = (np.arange(M) < Mp) & (inds[:, 1] < Mp)
        if np.sum(valid) == 0:
            fnn_frac = 1.0
        else:
            idx = np.where(valid)[0]
            dist_next = np.abs(Xp[idx, -1] - Xp[inds[idx, 1], -1])
            R_valid = R[idx]
            # avoid division by zero
            R_valid_safe = np.where(R_valid == 0, 1e-8, R_valid)
            frac = dist_next / R_valid_safe
            # criterion
            fnn_mask = (frac > Rtol) | (dist_next > Atol)
            fnn_frac = float(np.mean(fnn_mask))
        if fnn_frac < 0.01:
            return dim
    return max_dim


def estimate_takens_params(x, max_lag=50, max_dim=10):
    # returns (tau, dim)
    tau = first_minimum_ami(x, max_lag=max_lag)
    dim = false_nearest_neighbors(x, max_dim=max_dim, tau=tau)
    return int(tau), int(dim)


__all__ = ['average_mutual_information', 'first_minimum_ami', 'false_nearest_neighbors', 'estimate_takens_params']
