"""The budgeted case: why one number, a price, is enough -- and how to learn it online.

The problem. Over T queries, maximise total quality subject to a hard compute budget:

    maximise   sum_t q_{t, a_t}      subject to   sum_t c_{t, a_t}  <=  B.           (1)

This is a multiple-choice knapsack: one item must be chosen from each of T groups (the
K arms available on query t), under a single shared capacity. Solving it offline needs
the whole workload in advance and a dynamic program; a router has neither.

The structural result. Dualise the single coupling constraint with a multiplier p >= 0:

    L(p)  =  max_{a_1..a_T}  sum_t ( q_{t,a_t} - p * c_{t,a_t} )  +  p B.            (2)

The inner maximisation *decouples completely*: with p fixed, each query is answered by
whichever arm maximises q - p c, using nothing but that query's own numbers. So the
optimal policy for (1) has the form "act greedily on quality minus a price times cost",
and the entire budget constraint collapses into the scalar p. That is the single-price
(threshold) rule. Strong duality for the LP relaxation of (2) gives a p* at which the
greedy policy's spend meets the budget, and the integrality gap of a multiple-choice
knapsack is at most one item, so the single-price policy is within one query's quality
of the offline optimum. `offline_knapsack_dp` and `best_single_price` below compute
both, and tests/test_budget.py checks that bound on a small instance.

p* has a reading: it is the exchange rate between quality and compute at the margin, in
units of "probability of success per second". Below p* you should buy the second; above
it you should not. It is the same object as the lambda in r = q - lambda c, except that
lambda is a preference you assert and p* is a price the constraint imposes on you.

Learning p online. The dual function g(p) = L(p) is convex in p with subgradient
B/T - c_{t,a_t} at round t, so projected online gradient ascent gives

    p_{t+1}  =  max( 0,  p_t + eta * ( c_t - B / T ) ).                              (3)

Overspend relative to the per-query allowance B/T raises the price, which pushes the
greedy rule toward cheaper arms; underspend lowers it. Step size eta = eta_0 / sqrt(T)
is the standard online-convex-optimisation choice and preserves the O(sqrt(T)) rate.
The realized-spend curve tracking the straight line (B/T) t is what this looks like when
it works, and eval/pareto.py writes it out.

A hard cap, and an honest reserve. The dual update controls spend *in expectation*; it
does not forbid a single expensive call at t = T. So the router additionally refuses any
arm whose expected cost exceeds the budget it has left, which is a projection onto the
feasible set rather than a second heuristic.

That is still not enough for a strict guarantee, and the reason is worth stating because
it is the kind of gap that gets papered over. The router plans against *expected* cost
and is billed *realized* cost. Driven at the full allowance B/T, the dual drives realized
spend onto the budget line and then sits slightly above it, because a proportional
controller with no feedforward term tracks a ramp with a small steady-state lag. The
result is not a coin flip: across ten workload seeds the realized spend comes out at
1.00037 B to 1.00090 B -- it overshoots every time, by under a tenth of a percent
(tests/test_budget.py measures exactly this). So the dual is driven to a slightly
conservative target,

    allowance  =  (1 - reserve) * B / T,                                             (4)

with `reserve = 0.01`. One percent is an order of magnitude more than the overshoot it
has to absorb, and costs one percent of the budget; with it, spend lands at 0.9907 B to
0.9915 B over the same ten seeds and the constraint is never violated. Both figures are
in the test suite, because "the budget is respected" is only a claim if the version that
does not respect it -- by 0.06% -- is also on the record.
"""

from __future__ import annotations

import numpy as np


class BudgetedRouter:
    """Single-price routing with an online dual update on the budget constraint."""

    def __init__(self, n_arms: int, n_features: int, budget: float, horizon: int,
                 eta0: float = 2.0, ridge: float = 1.0, alpha: float = 1.0,
                 p_init: float = 0.0, hard_cap: bool = True, reserve: float = 0.01,
                 name: str = "budgeted (single-price)"):
        if horizon <= 0:
            raise ValueError("horizon must be positive; the dual step needs B/T")
        if not 0.0 <= reserve < 1.0:
            raise ValueError(f"reserve must be in [0,1), got {reserve}")
        self.name = name
        self.n_arms = n_arms
        self.d = n_features
        self.budget = float(budget)
        self.horizon = int(horizon)
        self.reserve = float(reserve)
        self.allowance = (1.0 - self.reserve) * self.budget / self.horizon
        self.eta = eta0 / np.sqrt(self.horizon)
        self.price = float(p_init)
        self.hard_cap = hard_cap
        self.alpha = float(alpha)
        self.spent = 0.0
        self.t = 0
        # Quality model: one optimistic ridge regression per arm on quality alone.
        # Cost is NOT folded into this target -- the whole point of the dual is that
        # the price multiplying cost is a free variable, so quality has to be estimated
        # separately from it.
        self.a_inv = np.stack([np.eye(n_features) / ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))
        self.theta_hat = np.zeros((n_arms, n_features))
        self.price_history: list[float] = []
        self.spend_history: list[float] = []

    def remaining(self) -> float:
        return self.budget - self.spent

    def quality_index(self, x: np.ndarray) -> np.ndarray:
        ax = np.einsum("kij,j->ki", self.a_inv, x)
        width = np.sqrt(np.maximum(np.einsum("ki,i->k", ax, x), 0.0))
        return self.theta_hat @ x + self.alpha * width

    def select(self, x: np.ndarray, expected_costs: np.ndarray) -> int:
        expected_costs = np.asarray(expected_costs, dtype=float)
        score = self.quality_index(x) - self.price * expected_costs
        if self.hard_cap:
            affordable = expected_costs <= self.remaining()
            if affordable.any():
                score = np.where(affordable, score, -np.inf)
            else:
                # Budget is gone. The query still has to be answered, so serve it from
                # the cheapest arm and record the overrun rather than pretending the
                # call did not happen.
                return int(np.argmin(expected_costs))
        return int(np.argmax(score))

    def update(self, x: np.ndarray, arm: int, reward: float, cost: float) -> None:
        """`reward` here is quality only; cost enters through the price, not the target."""
        ax = self.a_inv[arm] @ x
        self.a_inv[arm] -= np.outer(ax, ax) / (1.0 + float(x @ ax))
        self.b[arm] += reward * x
        self.theta_hat[arm] = self.a_inv[arm] @ self.b[arm]
        self.spent += float(cost)
        self.t += 1
        # Equation (3): projected dual ascent on the budget constraint.
        self.price = max(0.0, self.price + self.eta * (float(cost) - self.allowance))
        self.price_history.append(self.price)
        self.spend_history.append(self.spent)


# ---------------------------------------------------------------------------------
# Offline references, used to check the structural claim rather than to route.
# ---------------------------------------------------------------------------------

def offline_knapsack_dp(quality: np.ndarray, cost: np.ndarray, budget: float,
                        n_units: int = 2000) -> tuple[float, np.ndarray]:
    """Exact multiple-choice knapsack by DP on a discretised budget axis.

    quality, cost: (T, K). Returns (total quality, chosen arm per query).

    Costs are rounded *up* to a grid of `n_units` cells so the DP never spends more
    than the budget it thinks it is spending; the resulting solution is feasible for
    the true costs and its value is a lower bound on the true optimum, off by at most
    the discretisation. Only used on small instances in the test suite -- it is O(T K U).
    """
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    t, k = quality.shape
    unit = budget / n_units
    cost_u = np.ceil(cost / unit).astype(int)

    neg = -np.inf
    value = np.full(n_units + 1, neg)
    value[0] = 0.0
    choice = np.zeros((t, n_units + 1), dtype=np.int8)
    for i in range(t):
        new_value = np.full(n_units + 1, neg)
        new_choice = np.zeros(n_units + 1, dtype=np.int8)
        for a in range(k):
            cu = cost_u[i, a]
            if cu > n_units:
                continue
            shifted = np.full(n_units + 1, neg)
            shifted[cu:] = value[:n_units + 1 - cu] + quality[i, a]
            better = shifted > new_value
            new_value = np.where(better, shifted, new_value)
            new_choice = np.where(better, a, new_choice)
        value, choice[i] = new_value, new_choice

    end = int(np.argmax(value))
    total = float(value[end])
    arms = np.zeros(t, dtype=int)
    u = end
    for i in range(t - 1, -1, -1):
        a = int(choice[i, u])
        arms[i] = a
        u -= int(cost_u[i, a])
    return total, arms


def best_single_price(quality: np.ndarray, cost: np.ndarray, budget: float,
                      n_prices: int = 4000) -> tuple[float, float, np.ndarray]:
    """The best feasible policy of the form argmax_a (q - p c), swept over p.

    Returns (total quality, p*, chosen arms). This is the offline version of what
    `BudgetedRouter` learns online, and the pair (this, offline_knapsack_dp) is what
    makes the single-price claim a testable statement rather than a citation.
    """
    quality = np.asarray(quality, dtype=float)
    cost = np.asarray(cost, dtype=float)
    span = float(quality.max() - quality.min()) + 1e-12
    lo_c = float(cost.min())
    hi = 4.0 * span / max(lo_c, 1e-12)
    prices = np.concatenate([[0.0], np.geomspace(1e-6, hi, n_prices - 1)])
    best = (-np.inf, 0.0, np.zeros(len(quality), dtype=int))
    idx = np.arange(len(quality))
    for p in prices:
        arms = np.argmax(quality - p * cost, axis=1)
        spend = cost[idx, arms].sum()
        if spend <= budget:
            value = quality[idx, arms].sum()
            if value > best[0]:
                best = (float(value), float(p), arms)
    return best
