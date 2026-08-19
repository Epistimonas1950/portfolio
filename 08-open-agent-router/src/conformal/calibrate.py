"""Split conformal prediction, with the finite-sample guarantee derived, not cited.

The escalation rule this replaces is "escalate if the model's confidence is below 0.7".
That number is arbitrary, it is not comparable between a 3B model and a 32B model
because they are calibrated differently, and it comes with no guarantee of any kind.
Split conformal replaces it with a threshold that is *computed* from held-out data and
carries a distribution-free, finite-sample coverage bound.

Setup. Hold out a calibration set (X_1,Y_1),...,(X_n,Y_n) that the model did not train
on, and let (X_{n+1}, Y_{n+1}) be a fresh test point. Assume only that these n+1 pairs
are **exchangeable** -- their joint law is invariant to permutation. No distributional
form, no independence beyond exchangeability, no assumption that the model is any good.

Nonconformity score. Any measurable s(x, y) will do; the guarantee does not depend on
the choice, only the *usefulness* does. Here

    s(x, y)  =  1 - p_hat(y | x)                                                     (1)

the one minus the model's emitted probability of the candidate answer y. A large score
means "this answer looks wrong to the model".

The derivation. Let s_i = s(X_i, Y_i) for i = 1..n+1 and assume no ties almost surely
(true here: the emitted probabilities are continuous). Exchangeability of the pairs
implies exchangeability of the scores, so the rank of s_{n+1} among all n+1 of them is
uniform on {1,...,n+1}. Writing s_(k) for the k-th smallest of the n *calibration*
scores,

    P( s_{n+1} <= s_(k) )  =  P( rank(s_{n+1}) <= k )  =  k / (n+1).                 (2)

Choosing the smallest k that makes this at least 1 - alpha,

    k  =  ceil( (n+1)(1 - alpha) ),        q_hat  =  s_(k),                          (3)

which is the ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores. Then
define

    C(x)  =  { y : s(x, y) <= q_hat }.                                               (4)

By construction Y_{n+1} in C(X_{n+1}) if and only if s_{n+1} <= q_hat, so (2) and (3)
give the finite-sample bound

    P( Y_{n+1} in C(X_{n+1}) )  >=  1 - alpha,                                       (5)

and, when the scores are continuous, the matching upper bound 1 - alpha + 1/(n+1). The
probability in (5) is over the draw of the calibration set *and* the test point: it is
marginal coverage, not conditional. That distinction is the honest limitation of the
method and it is why eval/coverage.py reports coverage conditioned on difficulty as
well -- marginal coverage of 90% is perfectly compatible with 99% coverage on easy
queries and 60% on hard ones, which for a router is exactly the wrong way round.

Two failure modes that are handled explicitly rather than silently:

  * k > n, which happens when n < 1/alpha - 1 (at alpha = 0.01 you need at least 99
    calibration points). There is then no finite threshold that can be justified, and
    the honest output is the full label set, flagged.
  * a calibration set that is not exchangeable with the test set. Nothing in the code
    can detect this; (5) simply stops being true. eval/coverage.py breaks it on purpose
    and measures how far coverage falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """The output of split conformal: one number, and the provenance to defend it."""

    q_hat: float
    alpha: float
    n_cal: int
    rank: int                # k in equation (3)
    degenerate: bool         # True when k > n and C(x) is the full label set

    @property
    def guaranteed_coverage(self) -> float:
        """The finite-sample lower bound k/(n+1) that this q_hat actually delivers."""
        if self.degenerate:
            return 1.0
        return self.rank / (self.n_cal + 1)

    @property
    def upper_bound(self) -> float:
        """1 - alpha + 1/(n+1): the matching upper bound for continuous scores."""
        if self.degenerate:
            return 1.0
        return min(1.0, 1.0 - self.alpha + 1.0 / (self.n_cal + 1))


def nonconformity(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Equation (1): s = 1 - p_hat(y|x), evaluated at the true label. (N, L) -> (N,)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, L), got shape {probs.shape}")
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_quantile(scores: np.ndarray, alpha: float) -> Calibration:
    """Equation (3): the ceil((n+1)(1-alpha))/n empirical quantile of the scores.

    Deliberately not `np.quantile`. numpy's default interpolates between order
    statistics, and the guarantee in (2) is a statement about an *order statistic* --
    interpolating gives a threshold slightly below s_(k) and quietly voids the bound.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        raise ValueError("empty calibration set: split conformal needs held-out data")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        # No order statistic is high enough. Returning +inf means C(x) is the full
        # label set, which has trivial coverage 1 and set size L -- useless but honest,
        # and the `degenerate` flag makes it visible in the CSV rather than in a
        # surprising coverage number.
        return Calibration(q_hat=float("inf"), alpha=alpha, n_cal=n, rank=k,
                           degenerate=True)
    q = float(np.sort(scores)[k - 1])
    return Calibration(q_hat=q, alpha=alpha, n_cal=n, rank=k, degenerate=False)


def split_conformal(cal_probs: np.ndarray, cal_labels: np.ndarray,
                    alpha: float) -> Calibration:
    """Calibrate on held-out (probs, labels). One pass, one number out."""
    return conformal_quantile(nonconformity(cal_probs, cal_labels), alpha)


def prediction_sets(probs: np.ndarray, cal: Calibration) -> np.ndarray:
    """Equation (4) as a boolean mask. (N, L) -> (N, L)."""
    probs = np.asarray(probs, dtype=float)
    if cal.degenerate:
        return np.ones_like(probs, dtype=bool)
    return (1.0 - probs) <= cal.q_hat


def set_sizes(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=1)


def covered(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Whether the true label made it into each set."""
    return mask[np.arange(len(labels)), np.asarray(labels, dtype=int)]


def empirical_coverage(probs: np.ndarray, labels: np.ndarray,
                       cal: Calibration) -> float:
    return float(covered(prediction_sets(probs, cal), labels).mean())
