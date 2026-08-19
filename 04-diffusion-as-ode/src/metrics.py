r"""Distributional error metrics that are exact against a known Gaussian mixture.

FID needs an Inception network and a real image model; neither is available here (see
STATUS.md). What *is* available is something stronger for this purpose: the target
distribution is known in closed form, so the metrics below compare samples against the
true law rather than against another finite sample. That removes half the Monte-Carlo
noise and makes the remaining floor computable.

1-Wasserstein
-------------
On the line, W1(P, Q) = int_0^1 |F_P^{-1}(u) - F_Q^{-1}(u)| du. With an empirical P
from n samples the inverse CDF is the order statistic x_(i) on ((i-1)/n, i/n], so the
midpoint rule gives

    W1  ~=  (1/n) sum_i | x_(i) - F_Q^{-1}((i - 1/2)/n) |

Sorting is the whole algorithm. The estimator has an O(n^{-1/2}) statistical floor
that no sampler can beat, so every table here reports that floor explicitly, measured
by pushing the same inputs through the *exact* flow map.

Energy distance
---------------
    D^2(P,Q) = 2 E|X - Y| - E|X - X'| - E|Y - Y'|,   X,X' ~ P,  Y,Y' ~ Q

which is non-negative and zero only if P = Q. Every term is closed form here, so no
n^2 pairwise matrix is ever built:

  * E|X - X'| for the empirical P, from sorted samples:
        E|X - X'| = (2/n^2) sum_i x_(i) (2i - n - 1)          (i = 1..n)
    because sum_{i<j} (x_(j) - x_(i)) telescopes into that single pass.
  * E|x - Y| for Y ~ N(m, s^2) is the folded-normal mean,
        E|x - Y| = (x - m)(2 Phi(z) - 1) + 2 s phi(z),  z = (x - m)/s
    and for a mixture it is the w_k-weighted sum of those.
  * E|Y - Y'| for a mixture: Y - Y' is itself a mixture, over pairs (k, l), of
    N(mu_k - mu_l, v_k + v_l), so it is the same folded-normal formula summed over
    K^2 pairs with weights w_k w_l.

Moments and mode weights
------------------------
Mean, variance and skewness against their exact values (src/sde.py), and the total
variation between the empirical and true mixing weights after assigning each sample to
the component with the highest responsibility. The last one is what actually detects
a sampler that has dropped a mode -- a failure the moments can miss entirely.
"""

from __future__ import annotations

import math

import numpy as np

from .sde import GaussianMixture, _normal_cdf

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _folded_normal_mean(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E|Z| for Z ~ N(m, s^2), elementwise."""
    z = m / s
    return m * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * s * _normal_pdf(z)


def wasserstein1(samples: np.ndarray, target_quantiles: np.ndarray) -> float:
    """W1 between the empirical law of `samples` and the law whose midpoint quantiles
    F^{-1}((i-1/2)/n) are `target_quantiles` (precompute them once; they are fixed)."""
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    q = np.asarray(target_quantiles, dtype=np.float64).ravel()
    if x.size != q.size:
        raise ValueError(f"{x.size} samples against {q.size} target quantiles")
    return float(np.mean(np.abs(x - q)))


def target_midpoint_quantiles(mixture: GaussianMixture, n: int) -> np.ndarray:
    """F^{-1}((i - 1/2)/n), i = 1..n -- the fixed reference for `wasserstein1`."""
    return mixture.quantile((np.arange(n) + 0.5) / n)


def _mean_abs_self(samples: np.ndarray) -> float:
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.sum(x * (2.0 * i - n - 1.0)) / (n * n))


def _mean_abs_cross(samples: np.ndarray, mixture: GaussianMixture) -> float:
    x = np.asarray(samples, dtype=np.float64).ravel()[:, None]
    m = mixture.means[None, :, 0]
    s = np.sqrt(mixture.variances)[None, :]
    return float(np.mean((mixture.weights[None, :] * _folded_normal_mean(x - m, s)).sum(1)))


def _mean_abs_target(mixture: GaussianMixture) -> float:
    mu, v, w = mixture.means[:, 0], mixture.variances, mixture.weights
    dm = mu[:, None] - mu[None, :]
    ds = np.sqrt(v[:, None] + v[None, :])
    return float((w[:, None] * w[None, :] * _folded_normal_mean(dm, ds)).sum())


def energy_distance(samples: np.ndarray, mixture: GaussianMixture,
                    target_self: float | None = None) -> float:
    """D^2 = 2 E|X-Y| - E|X-X'| - E|Y-Y'|, exact against the mixture.

    `target_self` (E|Y-Y'|) depends only on the mixture; pass it in to avoid
    recomputing it for every sampler.
    """
    if target_self is None:
        target_self = _mean_abs_target(mixture)
    return float(2.0 * _mean_abs_cross(samples, mixture) - _mean_abs_self(samples)
                 - target_self)


def moment_errors(samples: np.ndarray, mixture: GaussianMixture) -> dict[str, float]:
    """Absolute errors in mean, variance and skewness against the exact values."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    mean, var, skew, _ = mixture.moments()
    xm, xv = float(x.mean()), float(x.var())
    xs = float(np.mean((x - xm) ** 3) / xv**1.5)
    return {"mean_error": abs(xm - mean), "var_error": abs(xv - var),
            "skew_error": abs(xs - skew)}


def mode_weight_error(samples: np.ndarray, mixture: GaussianMixture) -> float:
    """Total variation between empirical and true mixing weights.

    Each sample is assigned to argmax_k P(component k | x). Detects mode collapse and
    mode reweighting, which the low-order moments can average away.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    k = mixture.responsibilities(x).argmax(axis=1)
    counts = np.bincount(k, minlength=mixture.n_components) / x.shape[0]
    return float(0.5 * np.abs(counts - mixture.weights).sum())
