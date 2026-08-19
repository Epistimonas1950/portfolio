"""Synthetic layers with the two properties that make real LLM layers hard.

A quantizer tested on isotropic Gaussian activations will report that RTN is fine,
because with X X^T proportional to the identity the activation-weighted objective and
the weight-space objective coincide -- and RTN is exactly optimal for the latter. The
gap this project is about only exists when X is anisotropic.

So the generator below controls the two things that actually matter:

  spectral decay   the activation covariance has eigenvalues spanning `cond` orders
                   of magnitude, so some input directions are strongly excited and
                   others are nearly dead. This is what makes H ill-conditioned and
                   forces the damping question.
  outlier channels a few input channels with magnitudes `outlier_scale` times the
                   rest, reproducing the emergent-outlier behaviour reported in real
                   transformer activations.

Everything here is seeded and reproducible. No downloads, no model weights: the point
is to isolate the numerics, and a synthetic layer where the ground truth is known is
a better instrument for that than a real one where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticLayer:
    w: np.ndarray            # (n_out, n_in)
    x: np.ndarray            # (n_in, n_samples)
    outlier_channels: np.ndarray
    cond_target: float


def make_layer(n_out: int = 128, n_in: int = 256, n_samples: int = 512,
               cond: float = 1e4, n_outliers: int = 4,
               outlier_scale: float = 20.0, seed: int = 0) -> SyntheticLayer:
    """Generate one linear layer and the activations it sees.

    cond: target condition number of the activation covariance. 1e4 is a mild,
          realistic value; set it to 1.0 for the isotropic control case where the
          activation-aware method should show no advantage at all.
    """
    rng = np.random.default_rng(seed)

    w = rng.normal(scale=1.0 / np.sqrt(n_in), size=(n_out, n_in))

    # Activation covariance with a geometric spectrum spanning `cond`.
    spectrum = np.logspace(0.0, -np.log10(cond), n_in)
    basis, _ = np.linalg.qr(rng.normal(size=(n_in, n_in)))
    root = basis @ np.diag(np.sqrt(spectrum)) @ basis.T
    x = root @ rng.normal(size=(n_in, n_samples))

    outliers = rng.choice(n_in, size=n_outliers, replace=False) if n_outliers else \
        np.array([], dtype=int)
    if n_outliers:
        x[outliers, :] *= outlier_scale

    return SyntheticLayer(w=w, x=x, outlier_channels=np.sort(outliers),
                          cond_target=cond)


def make_stack(n_layers: int = 4, width: int = 128, n_samples: int = 512,
               cond: float = 1e3, seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """A chain of square layers plus the input activations, for propagation studies.

    Layer l+1's input is layer l's output, which is the whole point: quantization
    error does not stay where it was made.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.logspace(0.0, -np.log10(cond), width)
    basis, _ = np.linalg.qr(rng.normal(size=(width, width)))
    x0 = basis @ np.diag(np.sqrt(spectrum)) @ basis.T @ rng.normal(size=(width, n_samples))

    weights = []
    for _ in range(n_layers):
        m = rng.normal(scale=1.0 / np.sqrt(width), size=(width, width))
        # Normalize the spectral norm so activations neither explode nor vanish along
        # the chain; otherwise the propagation experiment measures the scaling of the
        # random matrices rather than the quantization error.
        m /= np.linalg.norm(m, 2)
        weights.append(m)
    return weights, x0
