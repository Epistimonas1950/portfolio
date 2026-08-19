"""A minimal low-bit quantizer, vendored so this repo stands alone.

Vendored 2026-08-18 from `02-scratch-quantizer/src/{grid,hessian,sequential}.py`.
This is the same mathematics as that project, reduced to
the two pieces the composition study needs: a uniform grid, and sequential quantization
with Cholesky error compensation. It is vendored rather than imported because each
project in this portfolio is a standalone repository -- a cross-repo import would make
`make test` here depend on a checkout of something else. The full treatment, with the
damping sweep, the column-ordering study and the cross-layer error bound, lives in `02`.

The objective, for a single matrix W against the activations X it multiplies:

    minimize   || W X - W_hat X ||_F^2      over W_hat on the b-bit grid

Round-to-nearest is exactly optimal for the *unweighted* problem ||W - W_hat||_F and
blind to X. Sequential quantization instead decides one column at a time and pushes each
rounding error into the columns not yet decided, using the layer Hessian H = 2 X X^T:

    W[:, q+1:]  -=  (delta / R_qq) (outer) R[q, q+1:]        R^T R = H^{-1}, R upper

which is why it wins at low bit-width and matters here: a factored layer W ~ A B has two
matrices to quantize, and they see completely different inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A symmetric uniform b-bit grid, one scale per row of the matrix."""

    bits: int
    per_channel: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError(f"bits must be in [2, 16], got {self.bits}")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def params(self, w: np.ndarray) -> np.ndarray:
        axis = 1 if self.per_channel else None
        amax = np.max(np.abs(np.atleast_2d(w)), axis=axis, keepdims=True)
        # A dead row would give scale 0; clamp so it round-trips to exactly zero.
        scale = np.maximum(amax / self.qmax, np.finfo(np.float64).tiny)
        return scale.reshape(-1, 1) if self.per_channel else scale.reshape(1, 1)

    def round_trip(self, w: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return scale * np.clip(np.round(w / scale), -self.qmax - 1, self.qmax)


def rtn(w: np.ndarray, grid: Grid) -> np.ndarray:
    """Round-to-nearest. Optimal for ||W - W_hat||_F, blind to the activations."""
    w = np.asarray(w, dtype=np.float64)
    return grid.round_trip(w, grid.params(w))


def _inverse_cholesky(h: np.ndarray) -> np.ndarray:
    """Upper-triangular R with R^T R = H^{-1}, via one Cholesky of H."""
    lower = np.linalg.cholesky(h)
    l_inv = np.linalg.solve(lower, np.eye(h.shape[0]))
    h_inv = l_inv.T @ l_inv
    return np.linalg.cholesky(0.5 * (h_inv + h_inv.T)).T


def sequential(w: np.ndarray, x: np.ndarray, grid: Grid,
               damping: float = 1e-2, act_order: bool = True) -> np.ndarray:
    """Quantize W column by column, compensating into the columns not yet decided.

    x: (n_in, n_samples) -- the activations this particular matrix multiplies. For the
       second factor of W ~ A B that is the layer input X; for the first factor it is
       B_hat X, which is the whole subtlety of composing the two methods.
    """
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"W has {w.shape[1]} columns but X has {x.shape[0]} rows")

    n_in = w.shape[1]
    h = 2.0 * (x @ x.T)
    h = h + damping * float(np.mean(np.diag(h))) * np.eye(n_in)

    # Most-excited input channels first: their error then has the largest remaining
    # set to spread into. Whichever column goes last has nothing left to absorb it.
    order = np.argsort(-np.diag(h)) if act_order else np.arange(n_in)
    inv = np.argsort(order)

    wp = w[:, order].copy()
    r = _inverse_cholesky(h[np.ix_(order, order)])
    scale = grid.params(wp)
    q = np.empty_like(wp)

    for j in range(n_in):
        col = wp[:, j]
        q_col = grid.round_trip(col.reshape(-1, 1), scale).ravel()
        q[:, j] = q_col
        if j + 1 < n_in:
            wp[:, j + 1:] -= np.outer((col - q_col) / r[j, j], r[j, j + 1:])
    return q[:, inv]
