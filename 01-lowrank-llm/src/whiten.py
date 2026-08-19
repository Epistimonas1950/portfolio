"""Whitening: the change of variable that puts Eckart-Young back in charge.

The objective that matters for a compressed linear layer is the activation-weighted
error, not the weight-space error:

    L(W_hat) = || (W - W_hat) X ||_F^2 ,        X : (n_in, n_samples)

Write E = W - W_hat and expand the trace:

    L = tr( E X X^T E^T ) = tr( E M E^T ),      M = X X^T   (n_in, n_in)

M is symmetric positive semidefinite, so it has a Cholesky factor M = S S^T with S
lower triangular, and

    tr( E M E^T ) = tr( E S S^T E^T ) = || E S ||_F^2

hence the identity the whole project rests on:

    || (W - W_hat) X ||_F  =  || (W - W_hat) S ||_F                            (1)

The right-hand side is an *unweighted* Frobenius norm of a matrix difference. Since S
is invertible, W_hat -> W_hat S is a bijection of the rank-r matrices onto themselves,
so minimizing (1) over rank-r W_hat is exactly the Eckart-Young problem for W S:

    W S = U Sigma V^T ,  truncate to rank r ,  W_hat = U_r Sigma_r V_r^T S^{-1}   (2)

and that W_hat is the *global* minimizer of the activation-weighted error over all
rank-r matrices -- not a heuristic. Note what this says about plain truncated SVD: it
is also a global minimizer, of a different objective, one that nobody downstream cares
about.

Damping. M is routinely near-singular: activation directions that the calibration set
barely excites give near-zero eigenvalues, S^{-1} amplifies them by the inverse of
their square roots, and with fewer calibration samples than input channels M is
singular by construction. The fix is a ridge, and it has an exact interpretation
rather than being a fudge. With M_lambda = M + lambda I = S S^T,

    || E S ||_F^2 = tr( E (M + lambda I) E^T ) = || E X ||_F^2 + lambda || E ||_F^2   (3)

so ridge-damped whitened SVD minimizes a convex interpolation between the two
objectives, and lambda -> infinity recovers plain truncated SVD exactly (S -> sqrt(lambda) I).
The ridge is a dial between the two methods, which is also why the isotropic control
case must show no advantage: there M is already a multiple of the identity.

`ridge_ratio` is dimensionless -- lambda = ratio * mean(diag M) -- so the same value
transfers between layers whose activation energies differ by orders of magnitude. An
absolute lambda does not transfer, and that is the usual reason a damping sweep looks
unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Whitening:
    """The whitening factor for one layer, plus the conditioning report."""

    s: np.ndarray             # (n, n) lower triangular, M + lambda I = S S^T
    lam: float                # the absolute ridge actually added
    ridge_ratio: float        # the dimensionless ratio it came from
    cond_raw: float           # cond(M), before damping -- the number to report
    cond_damped: float        # cond(M + lambda I), what the solve actually sees
    n_samples: int

    @property
    def n(self) -> int:
        return self.s.shape[0]


def second_moment(x: np.ndarray) -> np.ndarray:
    """M = X X^T for X of shape (n_in, n_samples).

    Up to a factor of 2 this is the layer Hessian used by the quantization project:
    two different compression methods, one statistic.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"X must be 2-D (n_in, n_samples), got shape {x.shape}")
    return x @ x.T


def apply_ridge(m: np.ndarray, ratio: float = 1e-4) -> tuple[np.ndarray, float]:
    """Return (M + lambda I, lambda) with lambda = ratio * mean(diag M).

    Scaling by the mean diagonal is what makes `ratio` dimensionless and therefore
    transferable across layers; see the module docstring.
    """
    m = np.asarray(m, dtype=np.float64)
    if ratio < 0:
        raise ValueError("ridge ratio must be non-negative")
    lam = float(ratio) * float(np.mean(np.diag(m)))
    return m + lam * np.eye(m.shape[0]), lam


def whitening_factor(m: np.ndarray) -> np.ndarray:
    """Lower-triangular S with M = S S^T. Raises if M is not positive definite.

    numpy's cholesky raises LinAlgError on an indefinite or singular matrix, and that
    is the behaviour we want: an undamped near-singular second moment must stop the
    run rather than return a silently amplified S^{-1}.
    """
    return np.linalg.cholesky(np.asarray(m, dtype=np.float64))


def whiten(x: np.ndarray, ridge_ratio: float = 1e-4) -> Whitening:
    """Full whitening pipeline for one layer: M, ridge, Cholesky, conditioning."""
    x = np.asarray(x, dtype=np.float64)
    m = second_moment(x)
    cond_raw = condition_number(m)
    m_damped, lam = apply_ridge(m, ridge_ratio)
    s = whitening_factor(m_damped)
    return Whitening(s=s, lam=lam, ridge_ratio=float(ridge_ratio), cond_raw=cond_raw,
                     cond_damped=condition_number(m_damped), n_samples=x.shape[1])


def condition_number(m: np.ndarray) -> float:
    """2-norm condition number via singular values, tolerating a singular matrix.

    np.linalg.cond raises nothing here but returns inf for a singular M, which is the
    honest answer and exactly what the conditioning table should print.
    """
    sv = np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    smallest = float(sv[-1])
    return float(sv[0] / smallest) if smallest > 0.0 else float("inf")


def solve_upper_triangular(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve U Z = B for upper-triangular U, by back substitution.

    scipy.linalg.solve_triangular is not available in this environment, and
    np.linalg.solve would throw the triangular structure away and run a full LU at
    O(n^3); back substitution is O(n^2 k) for k right-hand sides, and in the map-back
    below k = r << n. Tested against np.linalg.solve in tests/test_whiten.py, because
    an untested hand-rolled solver underneath the headline number would be a bad
    trade.
    """
    u = np.asarray(u, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = u.shape[0]
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"U must be square, got shape {u.shape}")
    if b.shape[0] != n:
        raise ValueError(f"U is {n}x{n} but B has {b.shape[0]} rows")

    z = np.array(b, dtype=np.float64, copy=True)
    for i in range(n - 1, -1, -1):
        if u[i, i] == 0.0:
            raise np.linalg.LinAlgError(
                f"upper-triangular solve: zero pivot at index {i}; the second moment "
                "is singular -- increase the ridge ratio")
        z[i] = (z[i] - u[i, i + 1:] @ z[i + 1:]) / u[i, i]
    return z


def unwhiten_rows(vt_r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Map the truncated right factor back: B = V_r^T S^{-1}, without forming S^{-1}.

    B = V_r^T S^{-1}  <=>  B S = V_r^T  <=>  S^T B^T = V_r, and S^T is upper
    triangular, so one back substitution does it.
    """
    vt_r = np.asarray(vt_r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return solve_upper_triangular(s.T, vt_r.T).T


def whitened_weights(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """W S -- the matrix whose truncated SVD solves the activation-weighted problem."""
    w = np.asarray(w, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if w.shape[1] != s.shape[0]:
        raise ValueError(f"W has {w.shape[1]} columns but S is {s.shape[0]}x{s.shape[1]}")
    return w @ s
