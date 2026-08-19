"""Synthetic layers whose activation covariance is a knob, not an accident.

The effect this repo measures exists only because activations are anisotropic. If
X X^T were a multiple of the identity, whitening would be a scalar, the
activation-weighted objective and the unweighted one would coincide, and plain
truncated SVD would already be optimal. So the generator below makes the anisotropy
the independent variable:

  cond          the activation covariance has eigenvalues spanning `cond` orders of
                magnitude, geometrically spaced. cond=1.0 gives the isotropic control
                case, where the advertised advantage must collapse -- that mirror test
                is the reason this parameter exists.
  n_spiked      a few input channels scaled up by `spike_scale`, reproducing the
                emergent-outlier behaviour reported in real transformer activations.
                They show up as large diagonal entries of M = X X^T.
  w_cond        the weight matrix gets its own controlled spectrum. Real weight
                matrices are not iid Gaussian; an iid W has a nearly flat
                Marchenko-Pastur spectrum and is a poor low-rank target for *any*
                method, which would flatter the comparison rather than sharpen it.

`shifted_activations` draws a *different* covariance with the same spectrum, so the
calibration-set distribution-shift risk can be measured instead of hand-waved.

Everything is seeded through an explicit np.random.default_rng. No downloads, no model
weights: the point is to isolate the numerics, and a synthetic layer whose ground truth
is known is a better instrument for that than a real one whose ground truth is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticLayer:
    """One linear layer, the activations it sees, and the recipe that made them."""

    name: str
    w: np.ndarray             # (n_out, n_in)
    x: np.ndarray             # (n_in, n_samples)
    root: np.ndarray          # (n_in, n_in) symmetric square root of the covariance
    spiked_channels: np.ndarray
    spike_scale: float
    cond_target: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.w.shape


def _covariance_root(n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """Symmetric square root of a covariance with a geometric spectrum spanning `cond`.

    A random orthogonal basis is used so no coordinate direction is special; the
    spiked channels below are the only axis-aligned structure in the generator.
    """
    if cond < 1.0:
        raise ValueError("cond must be >= 1 (1.0 is the isotropic control case)")
    spectrum = np.logspace(0.0, -np.log10(cond), n)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return basis @ np.diag(np.sqrt(spectrum)) @ basis.T


def _weight_matrix(n_out: int, n_in: int, w_cond: float,
                   rng: np.random.Generator) -> np.ndarray:
    """A weight matrix with a controlled singular-value spectrum, unit Frobenius scale.

    w_cond=1.0 gives an exactly flat spectrum (the hardest possible low-rank target).
    The default, one decade over the full spectrum, leaves a 256-column layer with a
    stable rank around 55 -- flat enough that plain truncated SVD has nothing cheap to
    discard, which is the regime trained transformer projection matrices are reported
    to sit in and the regime where this comparison is worth making.
    """
    k = min(n_out, n_in)
    left, _ = np.linalg.qr(rng.normal(size=(n_out, n_out)))
    right, _ = np.linalg.qr(rng.normal(size=(n_in, n_in)))
    sigma = np.logspace(0.0, -np.log10(w_cond), k)
    w = (left[:, :k] * sigma) @ right[:k, :]
    return w / np.sqrt(n_in)


def make_layer(n_out: int = 128, n_in: int = 256, n_samples: int = 512,
               cond: float = 1e5, n_spiked: int = 0, spike_scale: float = 20.0,
               w_cond: float = 10.0, seed: int = 0,
               name: str = "layer") -> SyntheticLayer:
    """Generate one linear layer and the calibration activations it sees.

    cond: target condition number of the activation covariance. 1e4-1e6 is the range
          where whitening is supposed to matter; set it to 1.0 for the isotropic
          control case, where the advantage must largely disappear.
    """
    if n_samples < 1 or n_in < 1 or n_out < 1:
        raise ValueError("layer dimensions must be positive")
    rng = np.random.default_rng(seed)

    w = _weight_matrix(n_out, n_in, w_cond, rng)
    root = _covariance_root(n_in, cond, rng)
    x = root @ rng.normal(size=(n_in, n_samples))

    if n_spiked:
        if n_spiked > n_in:
            raise ValueError("cannot spike more channels than the layer has")
        spiked = np.sort(rng.choice(n_in, size=n_spiked, replace=False))
        x[spiked, :] *= spike_scale
    else:
        spiked = np.array([], dtype=int)

    return SyntheticLayer(name=name, w=w, x=x, root=root, spiked_channels=spiked,
                          spike_scale=spike_scale if n_spiked else 1.0,
                          cond_target=cond)


def shifted_activations(layer: SyntheticLayer, seed: int, mix: float = 1.0,
                        n_samples: int | None = None) -> np.ndarray:
    """Activations from a *shifted* distribution, for the calibration-transfer test.

    The shifted covariance root is (1 - mix) * root_calib + mix * root_other, where
    root_other has the same eigenvalue spectrum but an independent eigenbasis. So
    mix=0 is a fresh draw from the calibration distribution (sampling noise only) and
    mix=1 is a full change of eigenbasis at matched spectrum -- the worst case that
    still keeps the layer's activation energy comparable, which is what makes the two
    columns fair to put side by side.
    """
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    n_in = layer.root.shape[0]
    n_samples = n_samples if n_samples is not None else layer.x.shape[1]
    # Note the degenerate case this makes honest: for cond=1 the covariance root is
    # exactly the identity whatever basis is drawn, so an isotropic layer has no
    # eigenbasis to rotate and every mix returns the same distribution. The shift
    # columns of results/pareto.csv coincide there for that reason, not because the
    # shift machinery is inert.
    other = _covariance_root(n_in, layer.cond_target, rng)
    root = (1.0 - mix) * layer.root + mix * other
    x = root @ rng.normal(size=(n_in, n_samples))
    if layer.spiked_channels.size:
        x[layer.spiked_channels, :] *= layer.spike_scale
    return x


#: Deliberately heterogeneous: shapes differ, so the parameter cost of one unit of
#: rank differs per layer, and the activation conditioning spans four decades, so the
#: spectra decay at very different rates. A rank allocator that ignores either of
#: those is going to lose here, which is the point of the comparison.
STACK_SHAPES: tuple[tuple[int, int, float], ...] = (
    (128, 256, 1e6),
    (192, 192, 1e5),
    (256, 128, 1e4),
    (128, 128, 1e3),
    (256, 256, 1e5),
    (64, 320, 1e2),
)


def make_stack(n_samples: int = 512, seed: int = 0, n_spiked: int = 2,
               shapes: tuple[tuple[int, int, float], ...] = STACK_SHAPES
               ) -> list[SyntheticLayer]:
    """A stack of independent layers with heterogeneous shapes and spectra.

    The layers are independent rather than chained: rank allocation is a per-layer
    budget problem, and chaining would confound it with error propagation, which is
    the neighbouring project's subject rather than this one's.
    """
    return [
        make_layer(n_out=m, n_in=n, n_samples=n_samples, cond=cond,
                   n_spiked=n_spiked, seed=seed * 100 + i, name=f"L{i}")
        for i, (m, n, cond) in enumerate(shapes)
    ]
