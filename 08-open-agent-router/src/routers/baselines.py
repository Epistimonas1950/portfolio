"""The baselines, including the one that is supposed to be hard to beat.

Four policies:

  FixedArm(k)          always the same model. `FixedArm(0)` is "always smallest",
                       `FixedArm(K-1)` is "always largest" -- between them they bracket
                       the cost-quality plane and every interesting policy lives inside.
  RandomRouter         uniform over arms. The control that proves a measurement is
                       sensitive to the policy: in eval/regret.py its cumulative regret
                       must grow with exponent 1, because it pays a constant expected
                       regret every round forever.
  ThresholdRouter      two cut-points on the difficulty-classifier score, fitted
                       offline on a labelled tuning split.

The threshold router is a first-class competitor, not a straw man, and BRIEF.md is
explicit about why: with a handful of arms and a decent difficulty signal, a monotone
rule "easy -> small, medium -> mid, hard -> large" can capture most of what there is to
capture. The bandit's extra machinery has to earn its place against it. So this
implementation is given every advantage that is honestly available:

  * it is fitted by exhaustive grid search over both cut-points,
  * on a *full-information* tuning split -- it sees the reward of every arm on every
    tuning query, which the bandits never do,
  * maximising exactly the objective the bandits are scored on, mean r = q - lambda c.

If it still loses, the loss is real. If it wins, that is the result and it gets
published, per BRIEF.md. What it cannot do is adapt when the workload drifts, because
its cut-points are frozen at fit time; that is the axis on which the bandit can pay for
itself, and it is the honest way to state the trade.
"""

from __future__ import annotations

import numpy as np


class FixedArm:
    """Always the same arm. No learning, no state."""

    def __init__(self, arm: int, name: str | None = None):
        self.arm = int(arm)
        self.name = name or f"always-arm-{arm}"

    def select(self, x: np.ndarray, expected_costs: np.ndarray | None = None) -> int:
        return self.arm

    def update(self, x: np.ndarray, arm: int, reward: float,
               cost: float | None = None) -> None:
        return None


class RandomRouter:
    """Uniform over arms. The exponent-1 control for the regret experiment."""

    def __init__(self, n_arms: int, seed: int = 0, name: str = "random"):
        self.n_arms = int(n_arms)
        self.name = name
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray, expected_costs: np.ndarray | None = None) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, x: np.ndarray, arm: int, reward: float,
               cost: float | None = None) -> None:
        return None


class ThresholdRouter:
    """Monotone cut-points on a difficulty score. Fitted offline, frozen at serve time.

    select: the smallest arm k such that score < cuts[k], else the largest arm.
    """

    def __init__(self, cuts: np.ndarray, score_index: int, n_arms: int,
                 name: str = "difficulty-threshold"):
        self.cuts = np.asarray(cuts, dtype=float)
        if self.cuts.size != n_arms - 1:
            raise ValueError(f"{n_arms} arms need {n_arms - 1} cut-points, "
                             f"got {self.cuts.size}")
        if np.any(np.diff(self.cuts) < 0):
            raise ValueError(f"cut-points must be non-decreasing, got {self.cuts}")
        self.score_index = int(score_index)
        self.n_arms = int(n_arms)
        self.name = name

    def select(self, x: np.ndarray, expected_costs: np.ndarray | None = None) -> int:
        s = float(x[self.score_index])
        return int(np.searchsorted(self.cuts, s, side="right"))

    def update(self, x: np.ndarray, arm: int, reward: float,
               cost: float | None = None) -> None:
        return None

    @classmethod
    def fit(cls, scores: np.ndarray, rewards: np.ndarray, score_index: int,
            n_grid: int = 41, name: str = "difficulty-threshold") -> "ThresholdRouter":
        """Exhaustive grid search over the cut-points on a full-information split.

        scores:  (N,)    the difficulty-classifier score for each tuning query
        rewards: (N, K)  the reward *every* arm would have earned -- full information,
                         which is a real advantage over the bandits and is stated as
                         such in the README.

        Only monotone (non-decreasing) cut vectors are considered. That is a modelling
        restriction, not an oversight: a non-monotone rule would be routing hard queries
        to the small model, and if that were profitable the capability ladder would be
        wrong and the whole project would be measuring nothing.
        """
        scores = np.asarray(scores, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        n_arms = rewards.shape[1]
        if n_arms != 3:
            raise NotImplementedError(
                f"ThresholdRouter.fit does the exhaustive search for 3 arms; got "
                f"{n_arms}. For more arms use a coordinate sweep over the K-1 cuts.")
        grid = np.linspace(0.0, 1.0, n_grid)
        best_value, best_cuts = -np.inf, np.array([0.0, 0.0])
        for c0 in grid:
            below0 = scores < c0
            for c1 in grid[grid >= c0]:
                arm = np.where(below0, 0, np.where(scores < c1, 1, 2))
                value = rewards[np.arange(len(scores)), arm].mean()
                if value > best_value:
                    best_value, best_cuts = value, np.array([c0, c1])
        return cls(best_cuts, score_index=score_index, n_arms=n_arms, name=name)
