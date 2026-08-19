r"""Forward SDEs, their marginals, and the closed-form score of a Gaussian mixture.

This is the module that lets the whole project run with no neural network. In a real
diffusion model the score grad_x log p_t(x) is what the network estimates; here the
prior is a Gaussian mixture, for which the perturbed marginal is *again* a Gaussian
mixture with known parameters, so the score is available in closed form. The network
is replaced by its exact analytic counterpart on purpose: every number this repo
reports is then a property of the integrator, not of a training run.

The forward SDE
---------------
Both families below are the linear, additive-noise SDEs of Song et al. (2011.13456):

    dx = a(t) x dt + g(t) dw

Because the drift is linear and the diffusion is state-independent, the transition
kernel is Gaussian:

    p(x_t | x_0) = N( alpha(t) x_0 , sigma(t)^2 I )

Variance preserving (VP), with beta(t) = beta_min + t (beta_max - beta_min):

    a(t) = -beta(t)/2,   g(t) = sqrt(beta(t))
    B(t) = int_0^t beta(s) ds = beta_min t + (beta_max - beta_min) t^2 / 2
    alpha(t) = exp(-B(t)/2),      sigma(t)^2 = 1 - exp(-B(t))

so alpha^2 + sigma^2 = 1 -- the variance is preserved for a unit-variance prior.

Variance exploding (VE), with sigma(t) = sigma_min (sigma_max/sigma_min)^t:

    a(t) = 0,   g(t)^2 = d sigma^2 / dt = 2 sigma(t)^2 log(sigma_max/sigma_min)
    alpha(t) = 1

The marginal of a Gaussian mixture
----------------------------------
If p_0 = sum_k w_k N(mu_k, v_k I) then, convolving each component with the kernel,

    p_t = sum_k w_k N( alpha(t) mu_k ,  V_k(t) I ),    V_k(t) = alpha(t)^2 v_k + sigma(t)^2

which is the single fact everything else rests on. Differentiating log p_t gives

    grad log p_t(x) = - sum_k r_k(x,t) (x - alpha mu_k) / V_k(t)

with responsibilities r_k = softmax_k( log w_k + log N(x; alpha mu_k, V_k) ). The
softmax is evaluated by subtracting the row maximum; with well-separated components
and small V_k the raw densities underflow long before the responsibilities do.

Log-SNR
-------
The exponential integrator needs the log signal-to-noise ratio and its inverse:

    lambda(t) = log( alpha(t) / sigma(t) )

For VP, alpha^2 = e^{-B} and sigma^2 = 1 - e^{-B}, so lambda = -log(e^B - 1)/2 and

    B = log(1 + e^{-2 lambda})

which inverts through the quadratic B(t) in closed form. No root-finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)
_ERFC = np.frompyfunc(math.erfc, 1, 1)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Phi(z), via erfc so the left tail does not cancel against 1."""
    return 0.5 * np.asarray(_ERFC(-np.asarray(z, dtype=np.float64) / SQRT2),
                            dtype=np.float64)


# --------------------------------------------------------------------------------------
# The prior, and every marginal of it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianMixture:
    """sum_k weights[k] N(means[k], variances[k] * I), isotropic components.

    Used for two things: the data prior p_0, and -- with rescaled means and inflated
    variances -- every perturbed marginal p_t. `variances` may contain exact zeros,
    which makes a component a point mass; that case is not degenerate for t > 0
    because the forward noise sigma(t)^2 is added to it.
    """

    weights: np.ndarray            # (K,)
    means: np.ndarray              # (K, d)
    variances: np.ndarray          # (K,)  isotropic

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64).ravel()
        m = np.asarray(self.means, dtype=np.float64)
        v = np.asarray(self.variances, dtype=np.float64).ravel()
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if not (w.shape[0] == m.shape[0] == v.shape[0]):
            raise ValueError(f"weights {w.shape}, means {m.shape}, variances {v.shape} "
                             "must agree on the number of components")
        if np.any(w < 0) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-12):
            raise ValueError("mixture weights must be non-negative and sum to 1")
        if np.any(v < 0):
            raise ValueError("component variances must be non-negative")
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "variances", v)

    @property
    def n_components(self) -> int:
        return self.weights.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    # -- densities -----------------------------------------------------------------

    def _log_component_densities(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (log w_k + log N(x; mu_k, V_k), x - mu_k) for x of shape (n, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        diff = x[:, None, :] - self.means[None, :, :]              # (n, K, d)
        sq = np.einsum("nkd,nkd->nk", diff, diff)                  # (n, K)
        v = self.variances[None, :]
        log_w = np.log(np.where(self.weights > 0, self.weights, np.finfo(float).tiny))
        logp = log_w[None, :] - 0.5 * self.dim * np.log(2.0 * np.pi * v) - sq / (2.0 * v)
        return logp, diff

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        """softmax over components -- the posterior P(component k | x)."""
        logp, _ = self._log_component_densities(x)
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        return r / r.sum(axis=1, keepdims=True)

    def score(self, x: np.ndarray) -> np.ndarray:
        """grad_x log p(x), exact. Shape in (n, d), shape out (n, d)."""
        logp, diff = self._log_component_densities(x)
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        return -np.einsum("nk,nkd->nd", r / self.variances[None, :], diff)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density on a 1-D grid; shape (n,) in, (n,) out."""
        self._require_1d()
        x = np.asarray(x, dtype=np.float64).ravel()
        sd = np.sqrt(self.variances)
        z = (x[:, None] - self.means[None, :, 0]) / sd[None, :]
        return (self.weights[None, :] * np.exp(-0.5 * z * z)
                / (sd[None, :] * math.sqrt(2.0 * np.pi))).sum(axis=1)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """F(x) = sum_k w_k Phi((x - mu_k)/sqrt(V_k)); 1-D only."""
        self._require_1d()
        x = np.asarray(x, dtype=np.float64).ravel()
        sd = np.sqrt(self.variances)
        z = (x[:, None] - self.means[None, :, 0]) / sd[None, :]
        return (self.weights[None, :] * _normal_cdf(z)).sum(axis=1)

    def quantile(self, p: np.ndarray, newton_steps: int = 4) -> np.ndarray:
        """F^{-1}(p), to machine precision. 1-D only.

        Bisection to bracket-width ~1e-11 (guaranteed to converge because F is
        strictly increasing), then Newton with the exact density, which squares the
        error each step and lands on the rounding level. Newton alone is not safe
        here: with well-separated components the density is ~0 between the modes and
        an unguarded Newton step flies off.
        """
        self._require_1d()
        p = np.asarray(p, dtype=np.float64)
        if np.any(p <= 0.0) or np.any(p >= 1.0):
            raise ValueError("quantile levels must lie strictly inside (0, 1)")
        sd_max = float(np.sqrt(self.variances).max())
        lo = float(self.means.min()) - 10.0 * sd_max - 1.0
        hi = float(self.means.max()) + 10.0 * sd_max + 1.0
        for _ in range(200):                       # widen until the bracket is valid
            if self.cdf(np.array([lo]))[0] <= p.min():
                break
            lo -= (hi - lo)
        for _ in range(200):
            if self.cdf(np.array([hi]))[0] >= p.max():
                break
            hi += (hi - lo)

        a = np.full(p.shape, lo)
        b = np.full(p.shape, hi)
        for _ in range(60):
            mid = 0.5 * (a + b)
            go_right = self.cdf(mid) < p          # root is above mid
            a = np.where(go_right, mid, a)
            b = np.where(go_right, b, mid)
        x = 0.5 * (a + b)
        for _ in range(newton_steps):
            dens = self.pdf(x)
            step = np.where(dens > 1e-300, (self.cdf(x) - p) / np.maximum(dens, 1e-300), 0.0)
            x = np.clip(x - step, a, b)            # never leave the bracket
        return x

    # -- moments and sampling -------------------------------------------------------

    def moments(self) -> tuple[float, float, float, float]:
        """(mean, variance, skewness, excess kurtosis) in closed form; 1-D only."""
        self._require_1d()
        w, mu, v = self.weights, self.means[:, 0], self.variances
        mean = float(w @ mu)
        c = mu - mean
        var = float(w @ (c * c + v))
        m3 = float(w @ (c**3 + 3.0 * c * v))
        m4 = float(w @ (c**4 + 6.0 * c * c * v + 3.0 * v * v))
        sd = math.sqrt(var)
        return mean, var, m3 / sd**3, m4 / var**2 - 3.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n i.i.d. draws, shape (n, d)."""
        k = rng.choice(self.n_components, size=n, p=self.weights)
        z = rng.normal(size=(n, self.dim))
        return self.means[k] + np.sqrt(self.variances[k])[:, None] * z

    def stratified(self, n: int) -> np.ndarray:
        """The n midpoint quantiles F^{-1}((i-1/2)/n): a deterministic stand-in for a
        sample whose empirical moments converge like O(n^-2) instead of O(n^-1/2)."""
        return self.quantile((np.arange(n) + 0.5) / n).reshape(-1, 1)

    def _require_1d(self) -> None:
        if self.dim != 1:
            raise ValueError("CDF / quantile / moment helpers are 1-D only; the exact "
                             "quantile transport used as ground truth exists because "
                             "monotone transport is unique on the line")


# --------------------------------------------------------------------------------------
# The forward SDEs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VPSDE:
    """Variance-preserving SDE, dx = -beta(t)/2 x dt + sqrt(beta(t)) dw."""

    beta_min: float = 0.1
    beta_max: float = 20.0
    t_min: float = 1e-3            # never integrate to 0: sigma(0) = 0 and lambda -> inf
    t_max: float = 1.0

    name: str = "vp"

    def beta(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def beta_integral(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def alpha(self, t: float | np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * self.beta_integral(t))

    def sigma(self, t: float | np.ndarray) -> np.ndarray:
        # -expm1(-B) rather than 1 - exp(-B): at t = t_min, B ~ 1e-4 and the naive form
        # loses four digits to cancellation.
        return np.sqrt(-np.expm1(-self.beta_integral(t)))

    def drift_coeff(self, t: float | np.ndarray) -> np.ndarray:
        return -0.5 * self.beta(t)

    def diffusion(self, t: float | np.ndarray) -> np.ndarray:
        return np.sqrt(self.beta(t))

    def log_snr(self, t: float | np.ndarray) -> np.ndarray:
        return -0.5 * np.log(np.expm1(self.beta_integral(t)))

    def t_of_log_snr(self, lam: float | np.ndarray) -> np.ndarray:
        b = np.logaddexp(0.0, -2.0 * np.asarray(lam, dtype=np.float64))
        d = self.beta_max - self.beta_min
        return (-self.beta_min + np.sqrt(self.beta_min**2 + 2.0 * d * b)) / d

    def marginal(self, prior: GaussianMixture, t: float) -> GaussianMixture:
        a, s = float(self.alpha(t)), float(self.sigma(t))
        return GaussianMixture(prior.weights, a * prior.means,
                               a * a * prior.variances + s * s)


@dataclass(frozen=True)
class VESDE:
    """Variance-exploding SDE, dx = sqrt(d sigma^2/dt) dw, sigma geometric in t.

    Kept because the exponential integrator's structure is clearer here: alpha == 1,
    so the linear part of the drift is *empty* and the whole benefit comes from
    stepping in log-SNR rather than in t.
    """

    sigma_min: float = 0.01
    sigma_max: float = 50.0
    t_min: float = 1e-3
    t_max: float = 1.0

    name: str = "ve"

    @property
    def _log_ratio(self) -> float:
        return math.log(self.sigma_max / self.sigma_min)

    def alpha(self, t: float | np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(t, dtype=np.float64))

    def sigma(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift_coeff(self, t: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(t, dtype=np.float64))

    def diffusion(self, t: float | np.ndarray) -> np.ndarray:
        return self.sigma(t) * math.sqrt(2.0 * self._log_ratio)

    def log_snr(self, t: float | np.ndarray) -> np.ndarray:
        return -np.log(self.sigma(t))

    def t_of_log_snr(self, lam: float | np.ndarray) -> np.ndarray:
        s = np.exp(-np.asarray(lam, dtype=np.float64))
        return np.log(s / self.sigma_min) / self._log_ratio

    def marginal(self, prior: GaussianMixture, t: float) -> GaussianMixture:
        s = float(self.sigma(t))
        return GaussianMixture(prior.weights, prior.means, prior.variances + s * s)


SDE = VPSDE | VESDE


# --------------------------------------------------------------------------------------
# The "network"
# --------------------------------------------------------------------------------------


def make_score(sde: SDE, prior: GaussianMixture):
    """The exact score of p_t, packaged with the signature a network would have.

    Returns score_fn(x, t) -> grad_x log p_t(x). This is the only object a sampler is
    allowed to call, and every call counts as one NFE (see src/nfe.py).
    """

    def score_fn(x: np.ndarray, t: float) -> np.ndarray:
        return sde.marginal(prior, float(t)).score(x)

    return score_fn


def noise_prediction(sde: SDE, score: np.ndarray, t: float) -> np.ndarray:
    """eps = -sigma(t) * score. The parameterization the exponential integrator uses.

    x_t = alpha x_0 + sigma eps means E[eps | x_t] = -sigma grad log p_t(x_t), so the
    two are the same object rescaled -- but eps stays O(1) as t -> 0 while the score
    blows up like 1/sigma, which is exactly why the exponential integrator approximates
    eps and not the score.
    """
    return -float(sde.sigma(t)) * score


def probability_flow_field(sde: SDE, score: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """dx/dt = f(x,t) - 0.5 g(t)^2 grad log p_t(x), given an already-evaluated score."""
    return float(sde.drift_coeff(t)) * x - 0.5 * float(sde.diffusion(t)) ** 2 * score


def reverse_sde_field(sde: SDE, score: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """The reverse-time SDE drift, f(x,t) - g(t)^2 grad log p_t(x)."""
    return float(sde.drift_coeff(t)) * x - float(sde.diffusion(t)) ** 2 * score
