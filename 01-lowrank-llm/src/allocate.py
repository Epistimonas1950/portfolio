"""Rank allocation under a global parameter budget: four solvers, one problem.

Compressing every layer to the same rank, or to the same ratio, is leaving accuracy on
the table -- layers differ in shape (so a unit of rank costs a different number of
parameters) and in spectral decay (so a unit of rank buys a different amount of error
reduction). The allocation problem is

    minimize   sum_l  L_l(r_l)      subject to   sum_l  r_l (m_l + n_l)  <=  B     (5)

with the per-layer loss taken from the tail of the *whitened* spectrum,

    L_l(r) = sum_{i > r} sigma_{l,i}^2 ,     sigma_l = singular values of W_l S_l

This is not a heuristic proxy. By the whitening identity (whiten.py eq. 1),
L_l(r) is *exactly* ||(W_l - W_hat_l) X_l||_F^2 for the rank-r whitened optimum at
zero ridge, so (5) is the true sum of squared per-layer activation-weighted errors.
It is a proxy only for what a *stack* does end to end, where layer errors compose
through the layers above them.

Four solvers, and the interesting part is which of them agree:

  uniform      one compression fraction for every layer, bisected to hit the budget.
               The baseline everybody actually ships.
  greedy       repeatedly spend the next unit of rank wherever it buys the most loss
               reduction per parameter.
  lagrangian   relax the budget constraint into  L_l(r_l) + mu * r_l (m_l + n_l)  and
               bisect on mu until the budget binds. At the solution every layer has
               been taken to the point where its marginal loss per parameter equals
               mu -- the marginal rates are equalized, which is what optimality of a
               separable allocation looks like.
  knapsack_dp  an exact multiple-choice knapsack dynamic program over a discretised
               rank grid, so the three heuristics can be scored against a true
               optimum rather than against each other.

What the mathematics predicts, and the measurement confirms: because sigma is sorted
descending, the marginal gain of the (r+1)-th component of a layer, sigma_{l,r+1}^2,
is non-increasing in r. So (5) is a separable problem with convex per-layer losses,
and for that class incremental greedy is *optimal* up to the last partial item. Greedy
and the DP should therefore agree almost exactly, and the honest report is that they
do -- not a manufactured gap. The gaps that are real:

  * uniform against everything else: large, and it is the number that matters.
  * lagrangian against greedy: small, and caused by integrality. Bisecting mu lands on
    a discrete lattice of achievable costs, so the multiplier that first fits inside B
    usually leaves budget unspent. `leftover_params` reports exactly how much.
  * greedy against the DP: near zero, for the convexity reason above.

The DP is exact *on its grid*: over the rank grid it is given, and with budget
measured in units of the greatest common divisor of the per-rank costs so no cost is
rounded. Outside that grid it is a lower bound on nothing at all, and the docstring
says so rather than the README implying otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from .rebuild import break_even_rank, dense_params, factored_params


@dataclass
class LayerSpec:
    """One layer's shape and whitened spectrum -- everything allocation needs."""

    name: str
    m: int
    n: int
    sigma: np.ndarray            # whitened singular values, descending

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        if np.any(np.diff(self.sigma) > 1e-9 * max(1.0, float(self.sigma[0]))):
            raise ValueError(f"{self.name}: singular values must be descending")

    @property
    def cost_per_rank(self) -> int:
        return self.m + self.n

    @property
    def max_rank(self) -> int:
        return int(min(self.m, self.n, self.sigma.size))

    @property
    def dense(self) -> int:
        return dense_params(self.m, self.n)

    @property
    def break_even(self) -> int:
        return break_even_rank(self.m, self.n)

    def loss(self, r: int) -> float:
        """L(r) = sum_{i > r} sigma_i^2."""
        r = int(np.clip(r, 0, self.sigma.size))
        return float(np.sum(self.sigma[r:] ** 2))

    def gain(self, r: int) -> float:
        """Loss reduction from taking rank r-1 to rank r: sigma_r^2 (1-indexed)."""
        return float(self.sigma[r - 1] ** 2) if 1 <= r <= self.sigma.size else 0.0


@dataclass
class Allocation:
    """A rank per layer, plus what it cost and what it lost."""

    strategy: str
    ranks: list[int]
    params: int
    budget: int
    loss: float

    @property
    def leftover_params(self) -> int:
        return self.budget - self.params


def _finish(strategy: str, layers: list[LayerSpec], ranks: list[int],
            budget: int) -> Allocation:
    params = sum(factored_params(l.m, l.n, r) for l, r in zip(layers, ranks))
    if params > budget:
        raise AssertionError(f"{strategy} produced {params} params over budget {budget}")
    loss = sum(l.loss(r) for l, r in zip(layers, ranks))
    return Allocation(strategy=strategy, ranks=[int(r) for r in ranks], params=params,
                      budget=int(budget), loss=float(loss))


def _max_rank(layer: LayerSpec, respect_break_even: bool) -> int:
    """Cap the rank so a "compressed" layer is never larger than the dense one."""
    return min(layer.max_rank, layer.break_even) if respect_break_even else layer.max_rank


def _check_budget(layers: list[LayerSpec], budget: int, min_rank: int) -> None:
    floor_cost = sum(factored_params(l.m, l.n, min_rank) for l in layers)
    if budget < floor_cost:
        raise ValueError(
            f"budget {budget} cannot even afford rank {min_rank} everywhere "
            f"({floor_cost} parameters); raise the budget or lower min_rank")


def total_dense(layers: list[LayerSpec]) -> int:
    """Parameters in the uncompressed stack -- the denominator of every ratio."""
    return sum(l.dense for l in layers)


def allocate_uniform(layers: list[LayerSpec], budget: int, min_rank: int = 1,
                     respect_break_even: bool = True) -> Allocation:
    """One rank *fraction* for every layer, bisected until the budget binds.

    "Uniform" could mean equal rank or equal fraction; equal fraction is the stronger
    baseline, because equal rank would penalise wide layers for being wide, and
    beating a straw man is not a result.
    """
    _check_budget(layers, budget, min_rank)
    caps = [_max_rank(l, respect_break_even) for l in layers]

    def ranks_at(frac: float) -> list[int]:
        return [int(np.clip(int(np.floor(frac * cap)), min_rank, cap))
                for cap in caps]

    def cost(rs: list[int]) -> int:
        return sum(factored_params(l.m, l.n, r) for l, r in zip(layers, rs))

    lo, hi = 0.0, 1.0
    # 60 bisection steps takes the bracket well below one part in 1e18, i.e. far below
    # the spacing between achievable integer ranks; the loop is O(60 L) and free.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cost(ranks_at(mid)) <= budget:
            lo = mid
        else:
            hi = mid
    return _finish("uniform", layers, ranks_at(lo), budget)


def allocate_greedy(layers: list[LayerSpec], budget: int, min_rank: int = 1,
                    respect_break_even: bool = True) -> Allocation:
    """Spend each parameter where it buys the most loss reduction.

    Every candidate increment (layer l, rank r) has gain sigma_{l,r}^2 and cost
    (m_l + n_l). Sorting all increments by gain-per-parameter and applying them in
    order automatically respects the prefix constraint -- rank r cannot be bought
    before rank r-1 -- because within a layer the gains are non-increasing and the
    cost is constant, so their ratios are non-increasing too. An increment that does
    not fit is skipped; every later increment in that layer has identical cost, so
    skipping cannot strand a cheaper one behind it.
    """
    _check_budget(layers, budget, min_rank)
    caps = [_max_rank(l, respect_break_even) for l in layers]
    ranks = [int(min(min_rank, cap)) for cap in caps]
    spent = sum(factored_params(l.m, l.n, r) for l, r in zip(layers, ranks))

    idx, ratios = [], []
    for li, layer in enumerate(layers):
        for r in range(ranks[li] + 1, caps[li] + 1):
            idx.append((li, r))
            ratios.append(layer.gain(r) / layer.cost_per_rank)
    if idx:
        # Stable sort so that, at exactly equal gain-per-parameter, the increment
        # generated earlier (lower r within a layer) is still applied first.
        order = np.argsort(-np.asarray(ratios), kind="stable")
        for k in order:
            li, r = idx[k]
            if ranks[li] != r - 1:
                continue                                  # this layer was skipped
            cost = layers[li].cost_per_rank
            if spent + cost > budget:
                continue
            ranks[li] = r
            spent += cost
    return _finish("greedy", layers, ranks, budget)


def allocate_lagrangian(layers: list[LayerSpec], budget: int, min_rank: int = 1,
                        respect_break_even: bool = True,
                        iterations: int = 100) -> Allocation:
    """Relax the budget with a multiplier and bisect until it binds.

    For a fixed mu each layer solves min_r [ L_l(r) + mu r (m_l + n_l) ] independently,
    and because the marginal gains sigma_{l,r}^2 are non-increasing the solution is in
    closed form: keep every component whose squared singular value pays its own way,

        r_l(mu) = #{ i : sigma_{l,i}^2 > mu (m_l + n_l) }

    Total cost is non-increasing in mu, so bisection finds the smallest mu that fits
    inside B. At that mu every layer sits where its marginal loss per parameter equals
    mu -- the marginal rates are equalized across layers, which is the whole content
    of the relaxation.

    The catch, and it is the honest finding: the achievable costs form a discrete
    lattice, so the smallest feasible mu generally leaves budget unspent. See
    `Allocation.leftover_params`.
    """
    _check_budget(layers, budget, min_rank)
    caps = [_max_rank(l, respect_break_even) for l in layers]

    def ranks_at(mu: float) -> list[int]:
        out = []
        for layer, cap in zip(layers, caps):
            thresh = mu * layer.cost_per_rank
            r = int(np.count_nonzero(layer.sigma[:cap] ** 2 > thresh))
            out.append(int(np.clip(r, min(min_rank, cap), cap)))
        return out

    def cost(rs: list[int]) -> int:
        return sum(factored_params(l.m, l.n, r) for l, r in zip(layers, rs))

    hi = max(float(np.max(l.sigma ** 2)) / l.cost_per_rank for l in layers) * 2.0
    lo = 0.0
    if cost(ranks_at(lo)) <= budget:
        return _finish("lagrangian", layers, ranks_at(lo), budget)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if cost(ranks_at(mid)) <= budget:
            hi = mid
        else:
            lo = mid
    return _finish("lagrangian", layers, ranks_at(hi), budget)


def allocate_knapsack_dp(layers: list[LayerSpec], budget: int, min_rank: int = 1,
                         rank_step: int = 1, respect_break_even: bool = True,
                         max_bins: int = 400_000) -> Allocation:
    """Exact multiple-choice knapsack DP over a discretised rank grid.

    Budget is measured in units of g = gcd of the per-rank costs (m_l + n_l), so every
    option's cost is an exact whole number of units and nothing is rounded -- the
    usual trap in a discretised knapsack is rounding costs down, which produces an
    "optimum" the budget cannot actually buy.

    dp[k][u] = smallest achievable loss for the first k layers using at most u units.
    Each layer contributes one option per grid rank, so the recurrence is

        dp[k][u] = min over options o of  dp[k-1][u - cost(o)] + L_k(rank(o))

    which is the exact optimum over the grid. It is *not* the exact optimum over all
    integer ranks unless rank_step is 1.
    """
    _check_budget(layers, budget, min_rank)
    caps = [_max_rank(l, respect_break_even) for l in layers]

    unit = 0
    for layer in layers:
        unit = gcd(unit, layer.cost_per_rank)
    n_units = int(budget) // unit
    if n_units > max_bins:
        raise ValueError(
            f"DP would need {n_units} budget bins (unit={unit}); this exceeds "
            f"max_bins={max_bins}. Coarsen the layer widths or raise max_bins "
            "deliberately -- silently rebinning would make the 'exact' optimum a "
            "different problem.")

    options = []
    for layer, cap in zip(layers, caps):
        grid = list(range(min(min_rank, cap), cap + 1, max(1, rank_step)))
        if grid[-1] != cap:
            grid.append(cap)
        options.append([(r, r * layer.cost_per_rank // unit, layer.loss(r))
                        for r in grid])

    inf = float("inf")
    prev = np.zeros(n_units + 1, dtype=np.float64)
    choices: list[np.ndarray] = []
    for layer_options in options:
        best = np.full(n_units + 1, inf)
        arg = np.full(n_units + 1, -1, dtype=np.int32)
        for oi, (_, cost_u, loss) in enumerate(layer_options):
            if cost_u > n_units:
                continue
            cand = np.full(n_units + 1, inf)
            cand[cost_u:] = prev[:n_units + 1 - cost_u] + loss
            better = cand < best
            best[better] = cand[better]
            arg[better] = oi
        if not np.isfinite(best[n_units]):
            raise ValueError("DP found no feasible allocation; budget is too small")
        prev = best
        choices.append(arg)

    ranks = [0] * len(layers)
    u = n_units
    for k in range(len(layers) - 1, -1, -1):
        oi = int(choices[k][u])
        r, cost_u, _ = options[k][oi]
        ranks[k] = r
        u -= cost_u
    return _finish("knapsack_dp", layers, ranks, budget)


STRATEGIES = {
    "uniform": allocate_uniform,
    "greedy": allocate_greedy,
    "lagrangian": allocate_lagrangian,
    "knapsack_dp": allocate_knapsack_dp,
}
