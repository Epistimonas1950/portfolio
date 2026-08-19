"""Sequential quantization with error compensation -- the core of the project.

The objective is the constrained least-squares problem

    minimize   || W X - W_hat X ||_F^2      subject to W_hat on the b-bit grid,

which is combinatorial. The practical relaxation, due to optimal brain surgeon and
carried into quantization by GPTQ: decide the columns one at a time, and after each
rounding decision, *correct the columns that have not been decided yet* so the layer
output is restored as far as the remaining freedom allows.

Concretely, with H = 2 X X^T the layer Hessian: rounding column q to w_q_hat incurs
weight error delta = w_q - w_q_hat, and the update to the remaining weights that
minimizes the resulting output error is

    delta_remaining  =  -( delta / [H^{-1}]_qq ) * H^{-1}[:, q]

restricted to the not-yet-quantized coordinates. In terms of the upper Cholesky
factor R of H^{-1} (so R^T R = H^{-1}) this reads, for the rows to the right of q,

    W[:, q+1:]  -=  (delta / R[q, q]) (outer) R[q, q+1:]

which is what the loop below does. Every rounding decision is therefore paid for by
the weights that have not been decided -- and that is why sequential quantization
crushes independent rounding at 3 and 4 bits while being nearly indistinguishable
from it at 8.

The numerics, not the formula, are where this gets interesting: H is singular in
practice (damping), the column order changes the answer (ordering.py), and the error
each layer emits becomes the next layer's input perturbation (propagation.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .hessian import damp, hessian, inverse_cholesky
from .ordering import column_order
from .outliers import detect_outlier_channels


@dataclass
class QuantResult:
    """A quantized layer plus everything needed to report on it honestly."""

    w_hat: np.ndarray
    order: np.ndarray
    outlier_cols: np.ndarray
    damping_lambda: float
    condition_number: float
    output_error: float = field(default=float("nan"))
    weight_error: float = field(default=float("nan"))


def _quantize_column(col: np.ndarray, grid: Grid, scale: np.ndarray,
                     zero: np.ndarray) -> np.ndarray:
    return grid.round_trip(col.reshape(-1, 1), scale, zero).ravel()


def sequential_quantize(w: np.ndarray, x: np.ndarray, grid: Grid,
                        damping: float = 1e-2, ordering: str = "salience",
                        groupsize: int | None = None,
                        n_outliers: int = 0) -> QuantResult:
    """Quantize one linear layer against the activations it actually sees.

    w:  (n_out, n_in) weight matrix
    x:  (n_in, n_samples) calibration activations
    """
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if w.shape[1] != x.shape[0]:
        raise ValueError(f"W has {w.shape[1]} input columns but X has {x.shape[0]} rows")

    n_out, n_in = w.shape
    h_raw = hessian(x)
    outliers = detect_outlier_channels(h_raw, n_outliers=n_outliers)

    h, lam = damp(h_raw, damping)
    cond = float(np.linalg.cond(h))
    order = column_order(ordering, w, h)

    # Work in the permuted basis so the loop is a plain left-to-right sweep.
    inv_order = np.argsort(order)
    w_perm = w[:, order].copy()
    h_perm = h[np.ix_(order, order)]
    r = inverse_cholesky(h_perm)

    keep_fp16 = np.zeros(n_in, dtype=bool)
    keep_fp16[outliers] = True
    keep_fp16_perm = keep_fp16[order]

    q_perm = np.empty_like(w_perm)
    step = n_in if groupsize is None else groupsize
    scale = zero = None

    for j in range(n_in):
        if j % step == 0:
            # Recompute group statistics from the *current* (already compensated)
            # weights, not the originals: compensation has moved them, and scaling to
            # the stale range clips.
            block = w_perm[:, j:j + step]
            scale, zero = grid.params(block)

        col = w_perm[:, j]
        if keep_fp16_perm[j]:
            q_perm[:, j] = col           # outlier channel: left in full precision
            continue

        q_col = _quantize_column(col, grid, scale, zero)
        q_perm[:, j] = q_col

        if j + 1 < n_in:
            delta = (col - q_col) / r[j, j]
            w_perm[:, j + 1:] -= np.outer(delta, r[j, j + 1:])

    w_hat = q_perm[:, inv_order]
    res = QuantResult(w_hat=w_hat, order=order, outlier_cols=outliers,
                      damping_lambda=lam, condition_number=cond)
    res.output_error = output_error(w, w_hat, x)
    res.weight_error = float(np.linalg.norm(w - w_hat))
    return res


def output_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    """|| (W - W_hat) X ||_F -- the only error that changes the model's behaviour."""
    return float(np.linalg.norm((w - w_hat) @ x))


def relative_output_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    """The same thing normalized by ||W X||_F, so it is comparable across layers."""
    denom = np.linalg.norm(w @ x)
    return output_error(w, w_hat, x) / float(denom if denom else 1.0)
