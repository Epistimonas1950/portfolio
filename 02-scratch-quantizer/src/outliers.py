"""Activation outlier channels.

A handful of input channels in a trained transformer carry activations one to two
orders of magnitude larger than the rest. Their contribution to ||(W - W_hat) X||_F
is scaled by that magnitude, so a uniform grid spends its resolution on channels that
do not matter and under-resolves the few that do.

The cheap, standard fix: find them from diag(H) = 2 * sum_t x_t^2 (the per-channel
activation energy the layer actually saw), and keep those columns of W in fp16. Cost
is `n_outliers / n_cols` extra bits per weight, accounted for in
`grid.effective_bits`.
"""

from __future__ import annotations

import numpy as np


def detect_outlier_channels(h: np.ndarray, n_outliers: int = 0,
                            z_threshold: float | None = None) -> np.ndarray:
    """Indices of the outlier input channels, from the Hessian diagonal.

    Give either an explicit count, or a threshold in units of standard deviations of
    log-energy (log, because the energies span decades and are roughly log-normal for
    the non-outlier bulk).
    """
    energy = np.diag(h).astype(np.float64)
    if n_outliers:
        return np.sort(np.argsort(-energy)[:n_outliers])
    if z_threshold is None:
        return np.array([], dtype=int)
    log_e = np.log(np.maximum(energy, np.finfo(np.float64).tiny))
    z = (log_e - log_e.mean()) / (log_e.std() + 1e-12)
    return np.sort(np.flatnonzero(z > z_threshold))
