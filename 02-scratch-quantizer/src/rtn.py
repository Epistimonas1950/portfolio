"""Round-to-nearest: the baseline that everything has to beat.

RTN solves

    minimize   ||W - W_hat||_F^2      over W_hat on the grid

which decouples completely -- every weight is rounded independently, and the answer
is the nearest grid point. That is exactly optimal for the wrong objective. What a
layer is for is producing `W X`, so the error that matters is ||(W - W_hat) X||_F,
and RTN is blind to X.

At 8 bits the grid is fine enough that the distinction barely registers, which is why
8-bit RTN is nearly free and nobody publishes it. Below 4 bits the gap opens up and
this file becomes the thing to beat.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid


def rtn_quantize(w: np.ndarray, grid: Grid,
                 groupsize: int | None = None) -> np.ndarray:
    """Round every weight to its nearest grid point.

    groupsize: if set, recompute (scale, zero) every `groupsize` columns. Finer
    groups track local dynamic range better at the cost of more stored metadata --
    see `grid.effective_bits`.
    """
    w = np.asarray(w, dtype=np.float64)
    out = np.empty_like(w)
    cols = w.shape[1]
    step = cols if groupsize is None else groupsize
    for start in range(0, cols, step):
        block = w[:, start:start + step]
        scale, zero = grid.params(block)
        out[:, start:start + step] = grid.round_trip(block, scale, zero)
    return out
