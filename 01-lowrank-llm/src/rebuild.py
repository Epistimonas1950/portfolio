"""Parameter accounting, done honestly, and reconstruction of a factored layer.

Replacing W (m x n) by A B with A (m x r) and B (r x n) stores r(m + n) numbers
instead of mn. That is a saving only while

    r (m + n) < m n        i.e.        r  <  m n / (m + n)                      (4)

so a square 256 x 256 layer breaks even at r = 127: factoring it at rank 128 costs
*more* storage than leaving it dense. Sweeps that quote "rank 200 of 256" as
compression are quoting an expansion, and the guard below exists so this repo cannot
do that by accident.

The compression ratio reported everywhere in this repo is the honest one,

    ratio = (parameters before) / (parameters after)

counted over exactly the matrices that were factored -- no biases, no embeddings, no
"effective" adjustments. A real model has parameters this method never touches, and
folding those in would inflate the ratio while changing nothing about the mathematics
being tested, so the numbers here are per-layer or per-stack over the factored layers
only, and the README says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dense_params(m: int, n: int) -> int:
    """Parameters in the uncompressed layer."""
    return int(m) * int(n)


def factored_params(m: int, n: int, r: int) -> int:
    """Parameters in the rank-r factorization A (m x r) and B (r x n)."""
    return int(r) * (int(m) + int(n))


def break_even_rank(m: int, n: int) -> int:
    """Largest rank at which the factorization is still strictly smaller than dense.

    Equation (4). Returns 0 when no rank compresses -- which happens for very thin
    layers, e.g. m = 1, where a factorization can never pay for itself.
    """
    m, n = int(m), int(n)
    r = (m * n - 1) // (m + n)
    return max(0, min(r, min(m, n)))


def compression_ratio(m: int, n: int, r: int) -> float:
    """dense / factored. Greater than 1 means the layer actually got smaller."""
    return dense_params(m, n) / factored_params(m, n, r)


def rebuild_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """W_hat = A B, with the shape check that catches a transposed factor."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must both be 2-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions disagree: A is {a.shape}, B is {b.shape}")
    return a @ b


@dataclass
class LayerReport:
    """What one compressed layer actually cost and actually lost."""

    name: str
    m: int
    n: int
    rank: int
    dense: int
    factored: int
    ratio: float
    compresses: bool
    relative_error: float = float("nan")

    @property
    def params_saved(self) -> int:
        return self.dense - self.factored


def report_layer(name: str, m: int, n: int, r: int,
                 relative_error: float = float("nan")) -> LayerReport:
    """Assemble the honest accounting for one layer at rank r."""
    dense = dense_params(m, n)
    factored = factored_params(m, n, r)
    return LayerReport(name=name, m=int(m), n=int(n), rank=int(r), dense=dense,
                       factored=factored, ratio=dense / factored,
                       compresses=factored < dense, relative_error=relative_error)


def stack_compression(reports: list[LayerReport]) -> float:
    """Achieved compression over a whole stack: total dense / total factored.

    Deliberately the ratio of sums, not the mean of ratios. Averaging per-layer
    ratios overweights small layers and is the standard way this number gets
    flattered.
    """
    dense = sum(r.dense for r in reports)
    factored = sum(r.factored for r in reports)
    if factored == 0:
        raise ValueError("stack has no parameters after factorization")
    return dense / factored
