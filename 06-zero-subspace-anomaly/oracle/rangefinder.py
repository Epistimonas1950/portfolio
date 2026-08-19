#!/usr/bin/env python3
"""Randomized range finding, and the probabilistic guarantee that justifies it.

This is the numpy reference for `src/rangefinder.c`. Both implement the same
algorithm; this one exists so the C can be checked against something, and so the
error bound can be tested rather than merely quoted.

The problem
-----------
Given `A` (m x n, columns are samples) find an orthonormal `Q` with few columns such
that `|| A - Q Q^T A ||` is close to the best possible, which by Eckart-Young is

    min over rank-k B of || A - B ||_F  =  ( sum_{j>k} sigma_j^2 )^{1/2}

The method
----------
Draw a Gaussian test matrix `Omega` (n x (k+p)), form `Y = A Omega`, and take
`Q = orth(Y)`. Each column of `Y` is a random linear combination of the columns of
`A`, weighted by the spectrum: directions with large singular values dominate, so
`k + p` random probes capture the top-k action of `A` with high probability.

The bound
---------
For a Gaussian `Omega` with oversampling `p >= 2` (Halko, Martinsson & Tropp 2011,
arXiv:0909.4061), the expected Frobenius-norm error obeys

    E || (I - Q Q^T) A ||_F  <=  ( 1 + k/(p-1) )^{1/2}  ( sum_{j>k} sigma_j^2 )^{1/2}

and in the spectral norm

    E || (I - Q Q^T) A ||_2  <=  ( 1 + sqrt(k/(p-1)) ) sigma_{k+1}
                                 + ( e sqrt(k+p) / p ) ( sum_{j>k} sigma_j^2 )^{1/2}

**What `p` buys.** Only the constant. With `k = 4` and `p = 6`, the Frobenius factor is
`(1 + 4/5)^{1/2} = 1.34`: the sketch is guaranteed, in expectation, to come within 34%
of the optimal rank-4 error. At `p = 1` the factor is undefined and at `p = 0` there is
no guarantee at all -- `Y` has exactly `k` columns and one unlucky draw that is nearly
orthogonal to a singular direction loses it outright. The deviation bounds in the same
paper fail with probability decaying like `p^{-p}`, so the handful of extra columns
that buys the modest constant above also buys a failure probability small enough to
stop thinking about. The cost is `p` extra columns everywhere: `O(mp)` memory and
`O(mnp)` work.

**What `q` buys.** Everything else. `p` cannot fix a slowly decaying spectrum, because
the tail term `(sum_{j>k} sigma_j^2)^{1/2}` is then genuinely large. Power iteration
replaces `A` with `B = (A A^T)^q A`, whose singular values are `sigma_j^{2q+1}`, so the
*ratio* of tail to signal is raised to the power `2q+1` before the bound is applied.
Because `|| (I - QQ^T) A || <= || (I - QQ^T) B ||^{1/(2q+1)}`, the bound for `A` becomes

    E || (I - Q Q^T) A ||_2  <=  [ (1 + sqrt(k/(p-1))) sigma_{k+1}^{2q+1}
                                   + (e sqrt(k+p)/p) ( sum_{j>k} sigma_j^{2(2q+1)} )^{1/2}
                                 ]^{1/(2q+1)}

which tends to `sigma_{k+1}`, the optimum, as `q` grows. The cost is `2q` extra passes
over `A` -- decisive when `A` is on disk, cheap when it is a 24 x 300 warm-up block.

**Why the orthonormalization inside the loop.** Forming `(A A^T)^q A Omega` by repeated
multiplication is numerically hopeless: the ratio between the largest and smallest
singular value of the product is `kappa^{2q+1}`, so at `q = 2` and `kappa = 10^3` the
weak directions are already below double precision and are rounded away. Re-
orthonormalizing between every application costs `O(m(k+p)^2)` and restores them. This
is the difference between the power-iteration and subspace-iteration forms of the
algorithm, and the one place where a textbook transcription silently produces garbage.
"""

from __future__ import annotations

import numpy as np


def randomized_range_finder(a: np.ndarray, k: int, p: int = 6, q: int = 1,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Orthonormal Q (m x (k+p)) whose range approximates the top-k range of A.

    a: (m, n) with columns as samples.
    p: oversampling. p >= 2 for the bound above to say anything; p = 0 is available
       as the no-oversampling control and is deliberately outside the guarantee.
    q: power iterations, with re-orthonormalization between applications.
    """
    if rng is None:
        raise ValueError("pass an explicit np.random.default_rng(seed); the results "
                         "in results/ are reproducible and a global RNG breaks that")
    m, n = a.shape
    ell = min(k + p, m, n)
    y = a @ rng.normal(size=(n, ell))
    qq, _ = np.linalg.qr(y)
    for _ in range(q):
        z, _ = np.linalg.qr(a.T @ qq)     # re-orthonormalize between every product,
        qq, _ = np.linalg.qr(a @ z)       # see the module docstring
    return qq


def randomized_svd(a: np.ndarray, k: int, p: int = 6, q: int = 1,
                   rng: np.random.Generator | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Approximate top singular vectors/values of A via the sketch.

    Once Q is in hand the rest is exact and cheap: B = Q^T A is (k+p) x n, its SVD
    costs O((k+p)^2 n), and U = Q U_B. Returns (U, sigma) with k + p columns, sorted
    descending -- the caller truncates, because rank selection needs to see the tail.
    """
    qq = randomized_range_finder(a, k, p, q, rng)
    b = qq.T @ a
    ub, s, _ = np.linalg.svd(b, full_matrices=False)
    return qq @ ub, s


def projection_error(a: np.ndarray, q: np.ndarray) -> float:
    """|| A - Q Q^T A ||_F, computed without forming the residual matrix twice."""
    return float(np.linalg.norm(a - q @ (q.T @ a), "fro"))


def optimal_error(a: np.ndarray, k: int) -> float:
    """The Eckart-Young floor: no rank-k projection can do better than this."""
    s = np.linalg.svd(a, compute_uv=False)
    return float(np.sqrt(np.sum(s[k:] ** 2)))


def frobenius_bound(a: np.ndarray, k: int, p: int) -> float:
    """(1 + k/(p-1))^{1/2} (sum_{j>k} sigma_j^2)^{1/2}, the expected-error bound.

    Undefined for p < 2 -- returns inf rather than a number, because reporting a
    finite bound outside its hypothesis is how a plot becomes a lie.
    """
    if p < 2:
        return float("inf")
    return float(np.sqrt(1.0 + k / (p - 1.0))) * optimal_error(a, k)


# --- rank selection ---------------------------------------------------------------
# BRIEF.md is explicit that r must not be hardcoded. Both criteria below are computed
# from the spectrum and both are reported; they answer different questions and can
# disagree, which is itself informative.


def rank_by_energy(sigma: np.ndarray, threshold: float = 0.95) -> int:
    """Smallest r with sum_{i<=r} sigma_i^2 / sum sigma_i^2 >= threshold.

    Energy is the natural criterion when the downstream quantity is a residual
    *energy*, which it is here: keeping 95% of the energy means a normal sample's
    score is bounded by 5% by construction. It is insensitive to where the spectrum
    actually breaks, so on a slowly decaying spectrum it returns a large r without
    complaint.
    """
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("sigma must be a non-empty 1-D array of singular values")
    energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    return int(np.searchsorted(energy, threshold) + 1)


def rank_by_gap(sigma: np.ndarray, r_max: int | None = None) -> int:
    """r maximizing sigma_r / sigma_{r+1}: the largest multiplicative gap.

    This is the right criterion when the model is "signal plus isotropic noise",
    because then the spectrum genuinely has a cliff and its location is the rank. It
    is fragile when there is no cliff: on a smooth spectrum it returns whichever
    adjacent pair happened to be furthest apart, which is noise.
    """
    if sigma.size < 2:
        return int(sigma.size)
    top = sigma.size - 1 if r_max is None else min(r_max, sigma.size - 1)
    floor = np.finfo(float).tiny
    ratios = sigma[:top] / np.maximum(sigma[1:top + 1], floor)
    return int(np.argmax(ratios) + 1)
