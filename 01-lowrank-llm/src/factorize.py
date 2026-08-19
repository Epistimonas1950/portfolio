"""The two factorizations, side by side: plain truncated SVD and whitened SVD.

Both return the same object -- factors A (m x r) and B (r x n) with W_hat = A B, so
`mn` stored parameters become `r(m + n)` -- and both are global minimizers of a
Frobenius-norm problem. They differ only in *which* matrix gets truncated:

    plain      truncate W    -> minimizes || W - W_hat ||_F          (Eckart-Young)
    whitened   truncate W S  -> minimizes || (W - W_hat) X ||_F      (Eckart-Young
                                 in the whitened variable; see whiten.py, eq. 1-2)

Nothing downstream of a layer consumes W. It consumes W X. So the second problem is
the one worth solving, and the first is a well-posed answer to a question nobody
asked. That is the entire content of this repo, and the gap between the two columns
is measured in analysis/pareto.py.

Diagnostics travel with the factorization rather than being recomputed later: the
singular values that were truncated (so the allocation problem in allocate.py gets
its loss curve for free), the tail energy sum_{i>r} sigma_i^2, and the conditioning of
the second moment, which is the number that says whether the whitened result should be
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .whiten import Whitening, unwhiten_rows, whiten, whitened_weights


@dataclass
class Factorization:
    """W_hat = A B, plus everything needed to report on it honestly."""

    a: np.ndarray                        # (m, r)
    b: np.ndarray                        # (r, n)
    rank: int
    method: str
    singular_values: np.ndarray          # of the matrix that was truncated
    tail_energy: float                   # sum_{i>r} sigma_i^2, the allocation loss
    tail_fraction: float                 # the same, relative to sum_i sigma_i^2
    ridge_lambda: float = float("nan")
    ridge_ratio: float = float("nan")
    cond_raw: float = float("nan")       # cond(M) before damping
    cond_damped: float = float("nan")    # cond(M + lambda I)
    w_hat: np.ndarray = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.a.shape[0], self.b.shape[1]


def _tail(sigma: np.ndarray, r: int) -> tuple[float, float]:
    total = float(np.sum(sigma ** 2))
    tail = float(np.sum(sigma[r:] ** 2))
    return tail, (tail / total if total > 0.0 else 0.0)


def _check_rank(r: int, w: np.ndarray) -> int:
    k = min(w.shape)
    if not isinstance(r, (int, np.integer)) or r < 1:
        raise ValueError(f"rank must be a positive integer, got {r!r}")
    if r > k:
        raise ValueError(f"rank {r} exceeds min(m, n) = {k}; there is nothing to truncate")
    return int(r)


def plain_truncated_svd(w: np.ndarray, r: int) -> Factorization:
    """Rank-r truncated SVD of W. The Eckart-Young optimum in the unweighted norm.

    Included as the baseline that must never be beaten in weight space -- and must
    lose badly in the norm that matters, whenever X is anisotropic.
    """
    w = np.asarray(w, dtype=np.float64)
    r = _check_rank(r, w)
    u, sigma, vt = np.linalg.svd(w, full_matrices=False)
    a = u[:, :r] * sigma[:r]
    b = vt[:r, :]
    tail, frac = _tail(sigma, r)
    return Factorization(a=a, b=b, rank=r, method="plain", singular_values=sigma,
                         tail_energy=tail, tail_fraction=frac, w_hat=a @ b)


def whitened_svd(w: np.ndarray, x: np.ndarray, r: int, ridge: float = 1e-4,
                 whitening: Whitening | None = None) -> Factorization:
    """Rank-r activation-aware factorization: truncate W S, then map back by S^{-1}.

    w:  (m, n) weight matrix
    x:  (n, n_samples) calibration activations for this layer
    r:  target rank
    ridge: dimensionless ratio, lambda = ridge * mean(diag X X^T). See whiten.py --
           the ridge interpolates between this method and plain truncated SVD, so
           ridge=0 is the pure activation-weighted optimum and large ridge degrades
           continuously back into the baseline.
    whitening: an already-computed Whitening for this layer. Sweeps over rank reuse
           one Cholesky instead of paying O(n^3) per rank, which is the difference
           between the pareto script taking seconds and taking minutes.
    """
    w = np.asarray(w, dtype=np.float64)
    r = _check_rank(r, w)
    if whitening is None:
        x = np.asarray(x, dtype=np.float64)
        if w.shape[1] != x.shape[0]:
            raise ValueError(
                f"W has {w.shape[1]} input columns but X has {x.shape[0]} rows")
        whitening = whiten(x, ridge)

    ws = whitened_weights(w, whitening.s)
    u, sigma, vt = np.linalg.svd(ws, full_matrices=False)

    # A = U_r Sigma_r ,  B = V_r^T S^{-1}. Folding Sigma_r into A is arbitrary but
    # keeps A B a two-matmul forward pass with no third factor to store.
    a = u[:, :r] * sigma[:r]
    b = unwhiten_rows(vt[:r, :], whitening.s)
    tail, frac = _tail(sigma, r)
    return Factorization(a=a, b=b, rank=r, method="whitened", singular_values=sigma,
                         tail_energy=tail, tail_fraction=frac,
                         ridge_lambda=whitening.lam, ridge_ratio=whitening.ridge_ratio,
                         cond_raw=whitening.cond_raw, cond_damped=whitening.cond_damped,
                         w_hat=a @ b)


def whitened_spectrum(w: np.ndarray, x: np.ndarray,
                      ridge: float = 1e-4) -> tuple[np.ndarray, Whitening]:
    """Singular values of W S and the whitening that produced them.

    The allocation problem needs only this curve per layer -- the loss of truncating
    layer l to rank r is the tail sum_{i>r} sigma_i^2 -- so it is worth getting
    without building a factorization for every candidate rank.
    """
    wh = whiten(x, ridge)
    sigma = np.linalg.svd(whitened_weights(np.asarray(w, dtype=np.float64), wh.s),
                          compute_uv=False)
    return sigma, wh


def activation_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    """|| (W - W_hat) X ||_F -- the only error that changes the layer's output."""
    return float(np.linalg.norm((np.asarray(w) - np.asarray(w_hat)) @ np.asarray(x)))


def relative_activation_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    """The same, normalized by ||W X||_F so it is comparable across layers.

    A value approaching 1 means the residual carries as much energy as the layer's
    own output, i.e. the compressed layer has stopped conveying anything useful; it
    is not a saturating scale and can exceed 1.
    """
    denom = float(np.linalg.norm(np.asarray(w) @ np.asarray(x)))
    return activation_error(w, w_hat, x) / (denom if denom else 1.0)


def weight_error(w: np.ndarray, w_hat: np.ndarray) -> float:
    """|| W - W_hat ||_F -- the norm plain truncated SVD is optimal in."""
    return float(np.linalg.norm(np.asarray(w) - np.asarray(w_hat)))
