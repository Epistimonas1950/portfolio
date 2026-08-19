"""Set-size deferral, and what survives when you stack the guarantee across tiers.

The deferral rule. Calibrate tier i at level alpha_i (src/conformal/calibrate.py) and
run the query through the tiers cheapest-first. At tier i:

    accept its answer  iff  |C_i(x)| = 1,   otherwise escalate to tier i+1.          (1)

A singleton set is the calibrated statement "at level alpha_i, exactly one answer is
consistent with what this model believes". A set with three answers in it is the
calibrated statement "this model is not sure", and it means that *whatever the tier's
raw softmax happens to look like* -- which is the point, since a 3B model and a 32B
model are confident about entirely different things and their raw margins are not on a
common scale. The last tier has no one to escalate to and returns its set as-is.

Composition. Let i*(x) be the tier that answers. The returned set is C_{i*}(x), and

    { Y not in C_{i*}(X) }  subset  union_i { Y not in C_i(X) }                      (2)

because i* takes values in {1..m}, so a miss by the cascade is a miss by *some* tier.
The union bound then gives, for any split of the error budget with sum_i alpha_i <= alpha,

    P( Y not in C_{i*}(X) )  <=  sum_i P( Y not in C_i(X) )  <=  sum_i alpha_i <= alpha.
                                                                                     (3)

End-to-end distribution-free coverage for a multi-tier system, in finite samples.

The premise that makes (3) legal, and the way to get it wrong
--------------------------------------------------------------
Each P(Y not in C_i(X)) <= alpha_i is a marginal statement over the distribution the
tier was *calibrated on*. So every tier must be calibrated on an i.i.d. draw from the
full query distribution -- not on the queries that reached it. Calibrating tier 2 on the
escalated stream is the natural-looking mistake: the escalated stream is a selected
subpopulation (the ones tier 1 was unsure about), it is not exchangeable with the
marginal, and (3) silently stops holding. `build_cascade` therefore takes one
calibration workload and calibrates every tier on all of it.

Where the union bound is loose, and why that is the interesting part
--------------------------------------------------------------------
Two independent reasons, and they are worth separating because they have different
fixes:

  1. **The miss events are strongly positively correlated.** A query that is genuinely
     hard -- ambiguous, adversarial, out of domain -- is missed by every tier at once.
     The union bound is tight only when the events are disjoint, and here they are close
     to nested. Formally the slack is sum_i alpha_i - P(union), and the union of highly
     overlapping events is far smaller than the sum of their probabilities.
  2. **Tiers 2..m are only consulted on the escalated subset.** Their miscoverage on the
     queries that never reach them is charged to the budget in (3) and then never
     incurred. This part of the slack grows as the acceptance rate at tier 1 grows: a
     cascade that accepts 80% of queries at tier 1 pays for tier 2's and tier 3's full
     alpha while exposing itself to only 20% of it.

Both are measured by `cascade_report`, which returns sum_i alpha_i alongside the
realized miscoverage. A tighter bound would need either a joint calibration over the
tiers' scores (which costs the distribution-free property, since the escalation event
depends on the same scores), or per-tier conditional calibration on the escalated
subpopulation (which is not exchangeable with the marginal, so the conditional
guarantee it gives is not the guarantee in (3)). Saying which of the two you would
spend is a design decision, not a detail: the first buys tightness with assumptions,
the second buys a different -- and for a cascade, arguably more useful -- guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibrate import Calibration, prediction_sets, split_conformal


@dataclass(frozen=True)
class Tier:
    """One rung of the cascade: an arm index and the level it was calibrated at."""

    arm: int
    alpha: float
    calibration: Calibration
    name: str = ""


@dataclass
class CascadeReport:
    """Everything eval/coverage.py needs, and nothing it has to recompute."""

    alphas: tuple[float, ...]
    union_bound: float
    empirical_miscoverage: float
    slack: float
    answering_tier: np.ndarray        # (N,) which tier answered each query
    accept_rate: tuple[float, ...]    # per tier, share of ALL queries it answered
    escalation_rate: tuple[float, ...]  # per tier, share of arrivals it escalated
    reach_rate: tuple[float, ...]     # per tier, share of ALL queries reaching it
    mean_set_size: float
    accepted_accuracy: float
    mean_cost: float
    per_tier_miscoverage: tuple[float, ...] = field(default_factory=tuple)

    @property
    def empirical_coverage(self) -> float:
        return 1.0 - self.empirical_miscoverage


def build_cascade(cal_probs: np.ndarray, cal_labels: np.ndarray,
                  arms: tuple[int, ...], alphas: tuple[float, ...],
                  names: tuple[str, ...] | None = None) -> tuple[Tier, ...]:
    """Calibrate every tier on the SAME full-distribution calibration set.

    cal_probs: (N, K, L) emitted distributions for every arm on every calibration query.
    Passing the whole array, rather than each tier's escalated slice, is the thing that
    keeps equation (3) valid -- see the module docstring.
    """
    cal_probs = np.asarray(cal_probs, dtype=float)
    if len(arms) != len(alphas):
        raise ValueError(f"{len(arms)} tiers but {len(alphas)} alphas")
    names = names or tuple(f"tier{i}" for i in range(len(arms)))
    return tuple(
        Tier(arm=a, alpha=al, calibration=split_conformal(cal_probs[:, a, :],
                                                          cal_labels, al), name=nm)
        for a, al, nm in zip(arms, alphas, names))


def run_cascade(test_probs: np.ndarray, test_labels: np.ndarray,
                tiers: tuple[Tier, ...],
                costs: np.ndarray | None = None) -> CascadeReport:
    """Run the deferral rule (1) over a test workload and measure what (3) predicts.

    test_probs: (N, K, L); costs: (N, K) price of calling each arm, cumulative over the
    tiers a query actually visits -- an escalated query pays for both calls, which is
    the honest accounting and the reason a cascade is not free.
    """
    test_probs = np.asarray(test_probs, dtype=float)
    test_labels = np.asarray(test_labels, dtype=int)
    n, _, n_labels = test_probs.shape
    m = len(tiers)

    masks = [prediction_sets(test_probs[:, t.arm, :], t.calibration) for t in tiers]
    sizes = [mk.sum(axis=1) for mk in masks]
    singleton = [sz == 1 for sz in sizes]

    answering = np.full(n, m - 1, dtype=int)
    decided = np.zeros(n, dtype=bool)
    reach = []
    accept = []
    escal = []
    spend = np.zeros(n)
    for i in range(m):
        arriving = ~decided
        reach.append(float(arriving.mean()))
        if costs is not None:
            spend[arriving] += costs[arriving, tiers[i].arm]
        take = arriving & singleton[i] if i < m - 1 else arriving
        answering[take] = i
        accept.append(float(take.mean()))
        escal.append(float((arriving & ~take).sum() / max(arriving.sum(), 1)))
        decided |= take

    rows = np.arange(n)
    final_mask = np.stack([masks[i][rows, :] for i in range(m)])[answering, rows, :]
    hit = final_mask[rows, test_labels]
    final_size = final_mask.sum(axis=1)

    per_tier_miss = tuple(
        float(1.0 - masks[i][rows, test_labels].mean()) for i in range(m))

    is_singleton = final_size == 1
    accepted_acc = float(hit[is_singleton].mean()) if is_singleton.any() else float("nan")

    ub = float(sum(t.alpha for t in tiers))
    miss = float(1.0 - hit.mean())
    return CascadeReport(
        alphas=tuple(t.alpha for t in tiers),
        union_bound=ub,
        empirical_miscoverage=miss,
        slack=ub - miss,
        answering_tier=answering,
        accept_rate=tuple(accept),
        escalation_rate=tuple(escal),
        reach_rate=tuple(reach),
        mean_set_size=float(final_size.mean()),
        accepted_accuracy=accepted_acc,
        mean_cost=float(spend.mean()) if costs is not None else float("nan"),
        per_tier_miscoverage=per_tier_miss,
    )


def split_budget(alpha: float, n_tiers: int) -> tuple[float, ...]:
    """The simplest legal split, alpha_i = alpha/m, so that sum_i alpha_i = alpha.

    Equal splitting is not optimal -- the tiers that answer more queries deserve more of
    the budget, because slack spent on a tier that is rarely reached is wasted. It is
    used here because it makes the comparison across alpha in eval/coverage.py clean,
    and because optimising the split against the test distribution would be a second
    use of the test data and would void the guarantee it is meant to demonstrate.
    """
    return tuple([alpha / n_tiers] * n_tiers)
