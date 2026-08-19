"""Quantization grids: scales, zero-points, symmetric and asymmetric.

A b-bit uniform grid maps a real value w to an integer code and back:

    q      = clip(round(w / s) + z,  qmin, qmax)          (quantize)
    w_hat  = s * (q - z)                                  (dequantize)

`s` is the step (scale) and `z` the zero-point. Two families:

  symmetric   z = 0, s = max|w| / qmax               grid is centred on zero
  asymmetric  z chosen so that min(w) maps to qmin   grid spans the actual range

and two granularities: one (s, z) for the whole tensor, or one per output channel
(one per row of W). Per-channel costs `rows` extra floats and is almost always worth
it, because rows of a trained weight matrix differ in dynamic range by an order of
magnitude or more -- a single tensor-wide step is then set by the widest row and
wastes most of the grid on every other row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A uniform quantization grid.

    bits:       number of bits per weight (2..8 are meaningful here)
    symmetric:  True for a zero-centred grid, False to fit the observed range
    per_channel: True for one (scale, zero) per row of W, False for one per tensor
    """

    bits: int
    symmetric: bool = True
    per_channel: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError(f"bits must be in [2, 16], got {self.bits}")

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.symmetric else 2**self.bits - 1

    @property
    def levels(self) -> int:
        return 2**self.bits

    def params(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute (scale, zero_point) for a weight block `w` of shape (rows, cols).

        Returns arrays broadcastable against `w`: shape (rows, 1) when per_channel,
        else shape (1, 1).
        """
        w = np.atleast_2d(w)
        axis = 1 if self.per_channel else None

        if self.symmetric:
            amax = np.max(np.abs(w), axis=axis, keepdims=True)
            # A dead row (all zeros) would give scale 0 and divide by zero. Clamp to
            # the smallest positive normal so the row round-trips to exactly zero.
            scale = np.maximum(amax / self.qmax, np.finfo(w.dtype).tiny)
            zero = np.zeros_like(scale)
        else:
            wmin = np.min(w, axis=axis, keepdims=True)
            wmax = np.max(w, axis=axis, keepdims=True)
            scale = np.maximum((wmax - wmin) / (self.qmax - self.qmin),
                               np.finfo(w.dtype).tiny)
            zero = np.round(self.qmin - wmin / scale)

        if not self.per_channel:
            scale = scale.reshape(1, 1)
            zero = zero.reshape(1, 1)
        return scale, zero

    def quantize(self, w: np.ndarray, scale: np.ndarray,
                 zero: np.ndarray) -> np.ndarray:
        """Real weights -> integer codes."""
        return np.clip(np.round(w / scale) + zero, self.qmin, self.qmax)

    def dequantize(self, q: np.ndarray, scale: np.ndarray,
                   zero: np.ndarray) -> np.ndarray:
        """Integer codes -> the reconstructed real weights."""
        return scale * (q - zero)

    def round_trip(self, w: np.ndarray, scale: np.ndarray,
                   zero: np.ndarray) -> np.ndarray:
        """Nearest point of the grid to `w`. This is round-to-nearest, per element."""
        return self.dequantize(self.quantize(w, scale, zero), scale, zero)


def effective_bits(rows: int, cols: int, bits: int, groupsize: int | None,
                   fp16_cols: int = 0) -> float:
    """Bits per weight once scales, zero-points and any fp16 columns are counted.

    An honest bits/weight number includes the metadata. A 4-bit quantizer with a
    per-16-column group scale stored in fp16 is really carrying 4 + 16/16 = 5 bits
    per weight, and reporting it as "4-bit" against a baseline that stores one scale
    per tensor is not a fair comparison.
    """
    total = rows * cols
    payload = (total - rows * fp16_cols) * bits + rows * fp16_cols * 16
    n_groups = 1 if groupsize is None else int(np.ceil(cols / groupsize))
    metadata = rows * n_groups * 16 * 2  # scale + zero, fp16 each
    return (payload + metadata) / total
