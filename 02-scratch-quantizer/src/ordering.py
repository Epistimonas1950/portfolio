"""Column orderings for sequential quantization.

Sequential quantization decides one column at a time and pushes the rounding error it
just made into the columns that have not been decided yet. So the order is not a
detail: whichever column goes last has no remaining columns to absorb its error, and
whichever goes first is compensated by everything else.

Three orderings, all defensible, and they do not agree:

  natural     0, 1, 2, ...          no information used
  salience    descending diag(H)    the most-excited input channels first, so their
                                    error is spread over the largest possible
                                    remaining set. This is GPTQ's "act-order".
  magnitude   descending ||W[:, j]||  a weight-space proxy that ignores activations
                                    -- included precisely so the comparison shows
                                    that the activation statistic is what matters.
"""

from __future__ import annotations

import numpy as np

ORDERINGS = ("natural", "salience", "magnitude")


def column_order(name: str, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the permutation of column indices to quantize in."""
    if name == "natural":
        return np.arange(w.shape[1])
    if name == "salience":
        return np.argsort(-np.diag(h))
    if name == "magnitude":
        return np.argsort(-np.linalg.norm(w, axis=0))
    raise ValueError(f"unknown ordering {name!r}; expected one of {ORDERINGS}")
