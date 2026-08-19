r"""Composing low-rank factorization with quantization -- and in which order.

The brief for this project volunteers a limitation up front: *at equal compression,
4-bit quantization usually beats low-rank factorization on modern LLMs.* That is a claim
about a crossover, and a crossover is something you measure rather than assert. This
module is the instrument.

Two methods, one budget
-----------------------
Both methods shrink the same layer, and both are parameterised. Low-rank trades away
rank `r`; quantization trades away bit-width `b`. Storing a factored, quantized layer
W ~ A B costs

    r (m + n) b  +  (m + r) * 16          bits          [payload + per-row fp16 scales]

against `m n * 16` for the dense fp16 original. So a target compression does not pick a
single configuration -- it picks a *curve* in the (r, b) plane, and the question "how
should I spend the budget" becomes "where on that curve is the error smallest". That is
the experiment; everything else here supports it.

Order matters, and one order does not work
------------------------------------------
**Low-rank first, then quantize** is the composition that actually composes. Factor
W ~ A B, then quantize the two factors -- and note that they do *not* see the same
input. B multiplies the layer activations X. A multiplies `B_hat X`, the output of the
already-quantized first factor. Quantizing A against X, or against B X, is measuring the
wrong Hessian; the correct one is over the r propagated directions.

**Quantize first, then factor** does not save what it appears to. Quantizing W puts it
on the grid, but the SVD of a grid-valued matrix has factors that are *not* grid-valued,
so they must be stored in fp16 -- at which point the storage is `r(m+n)*16`, exactly the
low-rank-only cost, and the first quantization bought nothing at all. Re-quantizing the
factors afterwards recovers the storage but pays the rounding error twice. Both variants
are implemented below so the claim is a measurement and not an argument.

Refitting the first factor
--------------------------
Once B is quantized, A is no longer optimal: it was chosen for the exact B. The best A
given `B_hat` is the least-squares solution

    A*  =  (W X)(B_hat X)^T [ (B_hat X)(B_hat X)^T ]^{-1}

which costs one r x r solve and recovers part of what quantizing B destroyed. This is
the same idea as the error compensation inside the quantizer, applied one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .factorize import relative_activation_error, whitened_svd
from .quantize import Grid, rtn, sequential

FP16 = 16


@dataclass
class Composite:
    """One compressed layer, with its storage accounted for honestly."""

    method: str
    rank: int | None
    bits: int | None
    w_hat: np.ndarray = field(repr=False)
    storage_bits: int = 0
    rel_error: float = float("nan")
    detail: str = ""

    def compression(self, m: int, n: int) -> float:
        return (m * n * FP16) / self.storage_bits

    def effective_bits(self, m: int, n: int) -> float:
        """Bits per weight of the *original* matrix -- the only comparable axis."""
        return self.storage_bits / (m * n)


def dense_bits(m: int, n: int) -> int:
    return m * n * FP16


def quantized_bits(m: int, n: int, bits: int) -> int:
    """Payload plus one fp16 scale per row."""
    return m * n * bits + m * FP16


def factored_bits(m: int, n: int, r: int, bits: int | None) -> int:
    """Factored storage. bits=None means the factors stay in fp16."""
    if bits is None:
        return r * (m + n) * FP16
    return r * (m + n) * bits + (m + r) * FP16


def ranks_for_budget(m: int, n: int, bits: int, target_compression: float) -> int:
    """The largest rank whose factored+quantized storage still hits the target.

    Inverting factored_bits for r; returns 0 when even rank 1 overshoots the budget,
    which is itself informative -- at aggressive compression the low-rank arm of the
    trade-off simply is not available. Capped at min(m, n), above which there is
    nothing left to truncate and the configuration is just quantization with extra
    steps -- a capped row therefore *beats* its target compression, and the tables
    report achieved compression rather than the target for exactly that reason.
    """
    budget = dense_bits(m, n) / target_compression
    r = int((budget - m * FP16) // ((m + n) * bits + FP16))
    return max(min(r, min(m, n)), 0)


# --------------------------------------------------------------------------- methods

def quantize_only(w: np.ndarray, x: np.ndarray, bits: int,
                  aware: bool = True) -> Composite:
    grid = Grid(bits)
    w_hat = sequential(w, x, grid) if aware else rtn(w, grid)
    m, n = w.shape
    return Composite(method=f"quantize only ({'aware' if aware else 'RTN'})",
                     rank=None, bits=bits, w_hat=w_hat,
                     storage_bits=quantized_bits(m, n, bits),
                     rel_error=relative_activation_error(w, w_hat, x))


def lowrank_only(w: np.ndarray, x: np.ndarray, r: int,
                 ridge: float = 1e-4) -> Composite:
    f = whitened_svd(w, x, r, ridge=ridge)
    m, n = w.shape
    return Composite(method="low-rank only (fp16 factors)", rank=r, bits=None,
                     w_hat=f.w_hat, storage_bits=factored_bits(m, n, r, None),
                     rel_error=relative_activation_error(w, f.w_hat, x))


def _refit_first_factor(w: np.ndarray, x: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
    """A* = argmin ||W X - A B_hat X||_F, an r x r normal-equation solve."""
    z = b_hat @ x                                  # (r, n_samples)
    gram = z @ z.T
    # Ridge for the same reason as everywhere else: quantizing B can leave two of its
    # rows nearly parallel, and then the Gram matrix is singular.
    gram += 1e-10 * float(np.trace(gram)) / gram.shape[0] * np.eye(gram.shape[0])
    return np.linalg.solve(gram, z @ (w @ x).T).T


def lowrank_then_quantize(w: np.ndarray, x: np.ndarray, r: int, bits: int,
                          aware: bool = True, refit: bool = True,
                          ridge: float = 1e-4) -> Composite:
    """Factor first, then quantize both factors against the inputs they actually see."""
    grid = Grid(bits)
    f = whitened_svd(w, x, r, ridge=ridge)

    # B multiplies the layer activations X.
    b_hat = sequential(f.b, x, grid) if aware else rtn(f.b, grid)

    # A multiplies B_hat X -- the output of the factor that has already been decided.
    a = _refit_first_factor(w, x, b_hat) if refit else f.a
    z = b_hat @ x
    a_hat = sequential(a, z, grid) if aware else rtn(a, grid)

    w_hat = a_hat @ b_hat
    m, n = w.shape
    tag = f"{'aware' if aware else 'RTN'}, {'refit' if refit else 'no refit'}"
    return Composite(method=f"low-rank -> quantize ({tag})", rank=r, bits=bits,
                     w_hat=w_hat, storage_bits=factored_bits(m, n, r, bits),
                     rel_error=relative_activation_error(w, w_hat, x), detail=tag)


def quantize_then_lowrank(w: np.ndarray, x: np.ndarray, r: int, bits: int,
                          requantize: bool = False,
                          ridge: float = 1e-4) -> Composite:
    """Quantize first, then factor the grid-valued matrix.

    requantize=False is the honest accounting: the SVD factors of a grid-valued matrix
    are not themselves grid-valued, so they cost fp16 and the first quantization saved
    nothing. requantize=True recovers the storage by rounding the factors too -- and
    pays the rounding error a second time.
    """
    grid = Grid(bits)
    w_q = sequential(w, x, grid)
    f = whitened_svd(w_q, x, r, ridge=ridge)
    m, n = w.shape

    if not requantize:
        return Composite(method="quantize -> low-rank (fp16 factors)", rank=r,
                         bits=bits, w_hat=f.w_hat,
                         storage_bits=factored_bits(m, n, r, None),
                         rel_error=relative_activation_error(w, f.w_hat, x),
                         detail="quantization bought no storage")

    b_hat = sequential(f.b, x, grid)
    a_hat = sequential(f.a, b_hat @ x, grid)
    w_hat = a_hat @ b_hat
    return Composite(method="quantize -> low-rank -> requantize", rank=r, bits=bits,
                     w_hat=w_hat, storage_bits=factored_bits(m, n, r, bits),
                     rel_error=relative_activation_error(w, w_hat, x),
                     detail="rounded twice")
