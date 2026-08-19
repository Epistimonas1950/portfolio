"""Linear Thompson sampling: the same posterior, sampled instead of maximised.

With a Gaussian likelihood r = theta_a^T x + eta, eta ~ N(0, v^2), and a Gaussian prior
theta_a ~ N(0, (v^2 / lambda) I), the posterior after the pulls of arm a is exactly

    theta_a | data  ~  N( theta_hat_a ,  v^2 A_a^{-1} ),
    A_a = lambda I + X_a^T X_a,   theta_hat_a = A_a^{-1} X_a^T y_a.                  (1)

The policy draws one sample theta_tilde_a from each arm's posterior and plays

    a_t = argmax_a  theta_tilde_a^T x_t.                                             (2)

The relationship to LinUCB is worth stating precisely, because it explains why both
appear in this repo. LinUCB plays the *upper end* of the interval
theta_hat^T x + alpha sqrt(x^T A^{-1} x); Thompson plays a *draw* from a distribution
whose standard deviation in the direction x is exactly v sqrt(x^T A^{-1} x) -- the same
width, from the same matrix. Optimism replaces the sample by a deterministic quantile.
Both achieve Otilde(d sqrt(T)) (Lattimore & Szepesvari); they differ in that Thompson's
exploration is randomised, so it does not need the constant alpha to be tuned, and in
that it is trivially parallelisable across a batch.

The sampling cost is the one implementation detail that matters. Drawing from
N(theta_hat, v^2 A^{-1}) needs a factor L with L L^T = A^{-1}, i.e. a Cholesky, which is
O(d^3). Recomputing it for every arm every round is O(K d^3 T) and dominates everything
else. Only the pulled arm's A^{-1} changes, so the factor is cached per arm and
refreshed on update -- O(d^3) per round rather than O(K d^3).
"""

from __future__ import annotations

import numpy as np


class LinearThompson:
    """Linear Thompson sampling with a Gaussian posterior, per arm."""

    def __init__(self, n_arms: int, n_features: int, v: float = 0.25,
                 ridge: float = 1.0, seed: int = 0, name: str = "Thompson"):
        self.name = name
        self.n_arms = n_arms
        self.d = n_features
        self.v = float(v)
        self.ridge = float(ridge)
        self.rng = np.random.default_rng(seed)
        self.a_inv = np.stack([np.eye(n_features) / ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))
        self.theta_hat = np.zeros((n_arms, n_features))
        # Cholesky factor of A^{-1} per arm, refreshed only when that arm is pulled.
        self.chol = np.stack([np.eye(n_features) / np.sqrt(ridge)
                              for _ in range(n_arms)])
        self.pulls = np.zeros(n_arms, dtype=int)

    def select(self, x: np.ndarray, expected_costs: np.ndarray | None = None) -> int:
        z = self.rng.standard_normal((self.n_arms, self.d))
        theta_tilde = self.theta_hat + self.v * np.einsum("kij,kj->ki", self.chol, z)
        return int(np.argmax(theta_tilde @ x))

    def update(self, x: np.ndarray, arm: int, reward: float,
               cost: float | None = None) -> None:
        ax = self.a_inv[arm] @ x
        self.a_inv[arm] -= np.outer(ax, ax) / (1.0 + float(x @ ax))
        self.b[arm] += reward * x
        self.theta_hat[arm] = self.a_inv[arm] @ self.b[arm]
        # Symmetrise before factoring: Sherman-Morrison is exact in theory but the
        # outer-product subtraction leaves asymmetry at the 1e-17 level, and
        # numpy.linalg.cholesky reads only the lower triangle -- so an unsymmetrised
        # matrix would silently factor a slightly different one.
        sym = 0.5 * (self.a_inv[arm] + self.a_inv[arm].T)
        self.chol[arm] = np.linalg.cholesky(sym)
        self.pulls[arm] += 1
