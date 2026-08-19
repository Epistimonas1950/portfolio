"""LinUCB: optimism in the face of uncertainty, with the confidence width derived.

Model. Arm a has its own parameter, E[r | x, a] = theta_a^T x (the "disjoint" LinUCB of
Li et al.). After n_a pulls of arm a with contexts X_a and rewards y_a, the ridge
estimate is

    A_a  =  lambda I + X_a^T X_a ,   b_a = X_a^T y_a ,   theta_hat_a = A_a^{-1} b_a.   (1)

Where the width comes from
--------------------------
The self-normalised tail bound for vector-valued martingales (Lattimore & Szepesvari,
Bandit Algorithms, Ch. 20) says that if the noise is conditionally sigma-sub-Gaussian
then, with probability at least 1 - delta, simultaneously for all t,

    || theta_hat_a - theta_a ||_{A_a}  <=  sigma sqrt( 2 log(1/delta)
                                            + log( det A_a / det(lambda I) ) )
                                          + sqrt(lambda) || theta_a ||_2  =:  beta_t.  (2)

The A_a-weighted Cauchy-Schwarz inequality then transfers that from the parameter to
any direction we care about:

    | x^T theta_hat_a - x^T theta_a |
        =  | <x, theta_hat_a - theta_a> |
        <= || x ||_{A_a^{-1}} * || theta_hat_a - theta_a ||_{A_a}
        <= beta_t * sqrt( x^T A_a^{-1} x ).                                          (3)

So the *shape* of the confidence interval is forced: it is sqrt(x^T A_a^{-1} x), the
Mahalanobis length of the query direction under the accumulated design. A direction the
arm has been probed in many times has a small width; an unexplored direction has a
large one. That is the entire content of "optimism". The scalar in front, beta_t, is
what the theory pins down: with ||x|| <= 1 the determinant grows at most like
(lambda + t/d)^d, so beta_t = O( sigma sqrt(d log t) + sqrt(lambda) S ), and plugging
(3) into the standard elliptical-potential argument gives

    Regret(T)  =  O( d sqrt(T) log T )  =  Otilde( d sqrt(T) ).                      (4)

In practice beta_t is replaced by a constant `alpha`, because the theoretical constant
is loose by an order of magnitude and inflating it only slows learning. That is a
deviation from the theory and it is stated here rather than hidden: the exponent in (4)
is what this repo verifies (eval/regret.py), not the constant.

Why Sherman-Morrison
--------------------
Each round updates A_a by a rank one term, A_a <- A_a + x x^T. Re-inverting costs
O(d^3) per round, O(T d^3) overall. The Sherman-Morrison identity

    (A + x x^T)^{-1}  =  A^{-1}  -  (A^{-1} x)(A^{-1} x)^T / (1 + x^T A^{-1} x)      (5)

updates the inverse in O(d^2), i.e. O(T d^2) overall -- a factor d, which at T = 64,000
is the difference between a script and a coffee break. It is also numerically safe
*here* specifically: A is symmetric positive definite and only ever grows, so the
denominator 1 + x^T A^{-1} x is bounded below by 1 and there is no cancellation. That
is not true of the downdate (subtracting a rank one term), which is why this class
never removes an observation. `tests/test_bandits.py` checks (5) against an explicit
re-inversion and reports the measured agreement.
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    """Disjoint LinUCB. One ridge model per arm, optimistic index, O(d^2) updates."""

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0,
                 ridge: float = 1.0, name: str = "LinUCB"):
        if ridge <= 0:
            raise ValueError("ridge must be positive: A = ridge*I + X^T X must be "
                             "invertible before any data has arrived")
        self.name = name
        self.n_arms = n_arms
        self.d = n_features
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.a = np.stack([np.eye(n_features) * ridge for _ in range(n_arms)])
        self.a_inv = np.stack([np.eye(n_features) / ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))
        # theta_hat is cached and refreshed only for the arm that was pulled: the other
        # K-1 models did not change, and recomputing all of them every round is the
        # dominant cost at K arms.
        self.theta_hat = np.zeros((n_arms, n_features))
        self.pulls = np.zeros(n_arms, dtype=int)

    def index(self, x: np.ndarray) -> np.ndarray:
        """(K,) upper confidence bound per arm: theta_hat^T x + alpha * width."""
        ax = np.einsum("kij,j->ki", self.a_inv, x)
        width = np.sqrt(np.maximum(np.einsum("ki,i->k", ax, x), 0.0))
        return self.theta_hat @ x + self.alpha * width

    def select(self, x: np.ndarray, expected_costs: np.ndarray | None = None) -> int:
        """Argmax of the optimistic index. Costs are already inside the reward."""
        return int(np.argmax(self.index(x)))

    def update(self, x: np.ndarray, arm: int, reward: float,
               cost: float | None = None) -> None:
        ax = self.a_inv[arm] @ x
        denom = 1.0 + float(x @ ax)
        self.a_inv[arm] -= np.outer(ax, ax) / denom          # equation (5)
        self.a[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.theta_hat[arm] = self.a_inv[arm] @ self.b[arm]
        self.pulls[arm] += 1

    def reinverted(self, arm: int) -> np.ndarray:
        """A_a^{-1} computed the expensive way, for the agreement test."""
        return np.linalg.inv(self.a[arm])


class LinGreedy(LinUCB):
    """alpha = 0: the same ridge models, no optimism. The control for exploration.

    Included because "the bandit helped" is only a claim if the same estimator without
    the exploration bonus is worse. On a fleet with diverse contexts it often is not,
    which is the honest version of the brief's warning about first-class baselines.
    """

    def __init__(self, n_arms: int, n_features: int, ridge: float = 1.0):
        super().__init__(n_arms, n_features, alpha=0.0, ridge=ridge, name="LinGreedy")
