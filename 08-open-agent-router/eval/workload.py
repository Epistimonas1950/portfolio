"""The two experimental harnesses, and the log-log fit both are read through.

Two workloads, deliberately separated, because they answer different questions:

  1. `LinearBandit` -- a synthetic bandit whose expected reward really is theta_a^T x.
     This is the only place the Otilde(d sqrt(T)) bound's premise actually holds, so it
     is the only place the exponent can be honestly checked. Nothing about a fleet of
     language models appears in it.
  2. `run_fleet` -- the simulated fleet (src/fleet/simulator.py), where the expected
     reward is a logistic in a latent difficulty the router only sees through a noisy
     classifier. The linear model is *misspecified* here, and the regret it produces is
     the honest number: a floor, not a sqrt.

Every number out of either is a simulated number. The first is not even simulating a
fleet; it is simulating the mathematical object the theorem is about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.cost import ComputeCostModel
from src.features import featurize
from src.fleet.simulator import LAMBDA, Workload, expected_reward_matrix


@dataclass(frozen=True)
class LinearBandit:
    """E[r | x, a] = theta_a^T x, contexts on the unit sphere, sub-Gaussian noise.

    `gap_scale` multiplies every theta_a, so it scales every instantaneous gap linearly
    while leaving the noise alone. That single knob indexes the *instance family* over
    which the minimax bound is a supremum -- see eval/regret.py, which is the whole
    reason this parameter exists.
    """

    d: int
    n_arms: int
    sigma: float
    gap_scale: float
    seed: int

    def theta(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return self.gap_scale * rng.normal(size=(self.n_arms, self.d)) / np.sqrt(self.d)


def run_linear_bandit(router_factory, inst: LinearBandit, horizon: int,
                      seed: int) -> np.ndarray:
    """Play `horizon` rounds and return the cumulative *expected* regret curve.

    Regret is accumulated on expected rewards, max_a theta_a^T x - theta_{a_t}^T x, not
    on the realized noisy draws. The noise is zero-mean and identical for every policy,
    so including it would add the same random walk of standard deviation sigma sqrt(T)
    to every curve -- at T = 64,000 and sigma = 1 that is +-250, which is the same order
    as the signal being measured. Expected regret is the quantity the theorem bounds and
    it is the quantity measured here.
    """
    rng = np.random.default_rng(seed)
    theta = inst.theta()
    router = router_factory(inst.n_arms, inst.d)
    inst_regret = np.empty(horizon)
    for t in range(horizon):
        x = rng.normal(size=inst.d)
        x /= np.linalg.norm(x)
        mu = theta @ x
        a = router.select(x, None)
        r = float(mu[a] + inst.sigma * rng.standard_normal())
        router.update(x, a, r, None)
        inst_regret[t] = float(mu.max() - mu[a])
    return np.cumsum(inst_regret)


@dataclass
class FleetRun:
    """What one policy did on one workload of the simulated fleet."""

    name: str
    arms: np.ndarray            # (T,) chosen arm
    success: np.ndarray         # (T,) realized
    cost: np.ndarray            # (T,) realized price
    reward: np.ndarray          # (T,) realized q - lambda c
    expected_reward: np.ndarray # (T,) E[r | x_t, a_t]
    regret: np.ndarray          # (T,) cumulative expected regret vs oracle_expected
    seconds: np.ndarray         # (T,) realized wall-clock of the chosen arm

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    @property
    def mean_cost(self) -> float:
        return float(self.cost.mean())

    @property
    def total_cost(self) -> float:
        return float(self.cost.sum())

    def arm_shares(self, n_arms: int) -> np.ndarray:
        return np.bincount(self.arms, minlength=n_arms) / len(self.arms)


def run_fleet(router, w: Workload, lam: float = LAMBDA, learn: bool = True,
              cost_model=None, reward_override: np.ndarray | None = None) -> FleetRun:
    """Route every query in `w`, with bandit feedback only.

    reward_override: (T, K) rewards to *learn from* instead of the true ones. Used by
    eval/surrogate_bias.py to feed the policy a biased signal while still scoring it
    against the truth. Everything reported here -- success, cost, regret -- is always
    computed from the true fleet, never from the surrogate.
    """
    cost_model = cost_model or ComputeCostModel()
    x_all = featurize(w)
    costs = w.cost_matrix(cost_model)
    exp_costs = w.expected_cost_matrix(cost_model)
    mu = expected_reward_matrix(w, lam, cost_model)
    best_mu = mu.max(axis=1)

    n = len(w)
    arms = np.empty(n, dtype=int)
    for t in range(n):
        a = router.select(x_all[t], exp_costs[t])
        arms[t] = a
        s = float(w.success[t, a])
        c = float(costs[t, a])
        signal = float(reward_override[t, a]) if reward_override is not None \
            else s - lam * c
        if learn:
            router.update(x_all[t], a, signal, c)

    rows = np.arange(n)
    success = w.success[rows, arms].astype(float)
    cost = costs[rows, arms]
    reward = success - lam * cost
    exp_reward = mu[rows, arms]
    regret = np.cumsum(best_mu - exp_reward)
    return FleetRun(name=getattr(router, "name", type(router).__name__), arms=arms,
                    success=success, cost=cost, reward=reward,
                    expected_reward=exp_reward, regret=regret,
                    seconds=w.seconds[rows, arms])


def replay_fixed(arms: np.ndarray, w: Workload, lam: float = LAMBDA,
                 name: str = "oracle", cost_model=None) -> FleetRun:
    """Score a precomputed arm sequence (an oracle) through the same accounting."""
    cost_model = cost_model or ComputeCostModel()
    costs = w.cost_matrix(cost_model)
    mu = expected_reward_matrix(w, lam, cost_model)
    rows = np.arange(len(w))
    success = w.success[rows, arms].astype(float)
    cost = costs[rows, arms]
    return FleetRun(name=name, arms=np.asarray(arms), success=success, cost=cost,
                    reward=success - lam * cost, expected_reward=mu[rows, arms],
                    regret=np.cumsum(mu.max(axis=1) - mu[rows, arms]),
                    seconds=w.seconds[rows, arms])


def loglog_slope(curve: np.ndarray, lo: int, hi: int,
                 n_points: int = 60) -> tuple[float, float, float]:
    """Least-squares fit of log(cum regret) against log(t) over [lo, hi].

    Returns (slope, r_squared, prefactor) where prefactor = exp(intercept), so the
    fitted curve is prefactor * t^slope and plot_results.py can draw it.

    Sampling t on a log grid rather than using every step is not cosmetic: with all
    64,000 points the fit is dominated by the dense tail and effectively measures the
    last half-decade only. A log-uniform grid weights each decade equally, which is what
    "the slope on log-log axes" means when you read it off by eye.
    """
    curve = np.asarray(curve, dtype=float)
    if hi > curve.size:
        raise ValueError(f"fit range hi={hi} exceeds curve length {curve.size}")
    if lo < 1 or lo >= hi:
        raise ValueError(f"need 1 <= lo < hi, got lo={lo}, hi={hi}")
    ts = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n_points)).astype(int))
    y = curve[ts - 1]
    keep = y > 0
    if keep.sum() < 3:
        raise ValueError("fewer than 3 positive regret values in the fit window")
    logt, logy = np.log(ts[keep]), np.log(y[keep])
    design = np.vstack([logt, np.ones(logt.size)]).T
    coef, residuals, *_ = np.linalg.lstsq(design, logy, rcond=None)
    ss_tot = float(((logy - logy.mean()) ** 2).sum())
    ss_res = float(residuals[0]) if residuals.size else float(
        ((logy - design @ coef) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(r2), float(np.exp(coef[1]))
