"""Counters shared by every search in this package, and the branching-factor fit.

EFFECTIVE BRANCHING FACTOR.  b* is defined, as in the standard treatment, as the
branching factor of the *uniform* tree that a search of the same solution depth would
have to explore to generate the same number of nodes:

    N  =  b* + b*^2 + ... + b*^d                                              (EBF)

with N the nodes generated excluding the root and d the depth of the solution
returned.  (EBF) is strictly increasing in b* for b* > 0, so bisection finds the unique
root.  One definition, used by every benchmark in this repo, because b* is only
meaningful as a comparison and comparing two different definitions is meaningless.

The number to watch is not b* itself but how it moves: a stronger heuristic, or better
move ordering in the adversarial search, pulls it down toward 1 (perfect guidance) or
toward sqrt(b) (perfect alpha-beta ordering).  Absolute values depend on the domain's
branching, which here *shrinks by one at every ply* as parcels get delivered, so the
tree is not uniform and b* is a summary statistic, not a law.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchStats:
    """Everything a claim in the README might need to cite."""

    algorithm: str = ""
    heuristic: str = ""
    expansions: int = 0
    generated: int = 0
    re_expansions: int = 0
    expansions_below_cstar: int = 0
    max_frontier: int = 0
    iterations: int = 0            # IDA* threshold rounds
    seconds: float = 0.0
    depth: int = 0
    solved: bool = False
    budget_exceeded: bool = False
    f_at_expansion: dict = field(default_factory=dict, repr=False)

    @property
    def ebf(self) -> float:
        return effective_branching_factor(self.generated, self.depth)

    def as_row(self) -> dict:
        return {
            "algorithm": self.algorithm, "heuristic": self.heuristic,
            "expansions": self.expansions, "generated": self.generated,
            "re_expansions": self.re_expansions,
            "expansions_below_cstar": self.expansions_below_cstar,
            "max_frontier": self.max_frontier, "iterations": self.iterations,
            "depth": self.depth, "ebf": round(self.ebf, 4),
            "seconds": round(self.seconds, 5), "solved": int(self.solved),
            "budget_exceeded": int(self.budget_exceeded),
        }


def effective_branching_factor(generated: int, depth: int, tol: float = 1e-9) -> float:
    """Solve N = b + b^2 + ... + b^d for b by bisection.  NaN if it is undefined."""
    if depth <= 0 or generated <= 0:
        return float("nan")
    if generated < depth:          # fewer nodes than plies: no b* >= 1 fits
        return float("nan")

    def total(b: float) -> float:
        if abs(b - 1.0) < 1e-12:
            return float(depth)
        return b * (b ** depth - 1.0) / (b - 1.0)

    lo, hi = 1.0, 2.0
    while total(hi) < generated and hi < 1e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < generated:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


class BudgetExceeded(RuntimeError):
    """Raised when a search passes its node budget.

    Searches here are exponential by nature and this repo would rather report "A* hit a
    10^6-node ceiling at horizon N" -- a real measurement -- than hang.
    """

    def __init__(self, algorithm: str, budget: int, stats: SearchStats) -> None:
        super().__init__(f"{algorithm} exceeded its {budget}-node budget "
                         f"({stats.generated} generated, {stats.expansions} expanded)")
        self.stats = stats
