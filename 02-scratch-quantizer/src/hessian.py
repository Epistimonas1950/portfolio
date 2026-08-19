"""The layer Hessian, and why it is the activation second moment.

For one linear layer the objective is

    L(W_hat) = || (W - W_hat) X ||_F^2 ,        X : (n_in, n_samples)

Write E = W - W_hat and expand: L = tr(E X X^T E^T). L is quadratic in each row of E
independently, and the Hessian of the scalar loss with respect to one row is

    H = 2 X X^T                                  (n_in, n_in)

the same matrix for every row -- which is exactly why a single Cholesky factorization
serves the whole layer. It is also, up to the factor 2, the same second moment that
the whitening step of the low-rank project (`01`) factors. Two different compression
methods; one statistic.

`H` is routinely singular in practice: some input channels are barely excited, so
their rows and columns of X X^T are near zero and the factorization fails. The fix is
a ridge, H + lambda * mean(diag H) * I, and the size of lambda is a genuine tradeoff
rather than a magic constant -- see `analysis/damping_sweep.py`.
"""

from __future__ import annotations

import numpy as np


def hessian(x: np.ndarray) -> np.ndarray:
    """H = 2 X X^T for X of shape (n_in, n_samples)."""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x @ x.T)


def damp(h: np.ndarray, ratio: float = 1e-2) -> tuple[np.ndarray, float]:
    """Add a ridge proportional to the mean diagonal. Returns (H_damped, lambda).

    Scaling by mean(diag H) makes `ratio` dimensionless, so the same value transfers
    across layers whose activation energies differ by orders of magnitude. Absolute
    damping does not transfer and is the usual reason a sweep looks unstable.
    """
    h = np.asarray(h, dtype=np.float64)
    if ratio < 0:
        raise ValueError("damping ratio must be non-negative")
    lam = ratio * float(np.mean(np.diag(h)))
    return h + lam * np.eye(h.shape[0]), lam


def inverse_cholesky(h: np.ndarray) -> np.ndarray:
    """Upper-triangular R with R^T R = H^{-1}, without ever forming H^{-1} badly.

    H = L L^T  (lower Cholesky)
    H^{-1} = L^{-T} L^{-1}
    Cholesky of H^{-1} is L_i L_i^T with L_i lower; take R = L_i^T.

    The sequential solver only ever needs R's diagonal and its rows to the right, so
    this one factorization replaces an explicit inverse at every column -- an O(n^3)
    setup instead of O(n^4) total, which is the whole reason the method is practical.
    """
    h = np.asarray(h, dtype=np.float64)
    lower = np.linalg.cholesky(h)                      # raises if H is not PD
    l_inv = np.linalg.solve(lower, np.eye(h.shape[0]))  # L^{-1}
    h_inv = l_inv.T @ l_inv
    h_inv = 0.5 * (h_inv + h_inv.T)                     # kill asymmetry from rounding
    return np.linalg.cholesky(h_inv).T
