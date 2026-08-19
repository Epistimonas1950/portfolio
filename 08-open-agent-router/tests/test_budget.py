"""The budget constraint, and the structural claim that one price is enough."""

import unittest

import numpy as np

from eval.workload import run_fleet
from src.features import N_FEATURES
from src.fleet.simulator import make_workload
from src.routers.budgeted import (BudgetedRouter, best_single_price,
                                  offline_knapsack_dp)
from src.routers.linucb import LinUCB


class TestBudgetIsRespected(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # The constrained policy must stay inside B and the unconstrained one must not. The
    # second half is what makes the first half a result: a budget that the unconstrained
    # policy already satisfies constrains nothing, so the test constructs B from the
    # unconstrained policy's own spend and then asserts the separation.
    def test_budgeted_stays_under_B_and_unconstrained_does_not(self):
        for seed in (401, 402, 403):
            w = make_workload(12_000, seed=seed)
            unconstrained = run_fleet(LinUCB(3, N_FEATURES, alpha=1.0), w)
            budget = 0.70 * unconstrained.total_cost
            router = BudgetedRouter(3, N_FEATURES, budget=budget, horizon=len(w),
                                    eta0=2.0, alpha=1.0)
            run = run_fleet(router, w, reward_override=w.success.astype(float))
            self.assertLessEqual(
                run.total_cost, budget,
                f"seed {seed}: budgeted router spent {run.total_cost:.1f} against a "
                f"budget of {budget:.1f}")
            self.assertGreater(unconstrained.total_cost, budget,
                               "the budget does not bind, so the test proves nothing")
            # And it must actually use the budget, not trivially satisfy it by always
            # choosing the cheapest arm.
            self.assertGreater(run.total_cost, 0.90 * budget,
                               f"seed {seed}: spent only {run.total_cost / budget:.3f} "
                               "of the budget; the dual is not tracking the line")

    def test_without_the_reserve_the_realized_spend_crosses_B_every_time(self):
        """The honest version of the guarantee: the 1% reserve is doing real work.

        With reserve = 0 the dual drives spend onto the budget line and sits just above
        it -- a proportional controller tracking a ramp has a steady-state lag, so the
        overshoot is systematic rather than a coin flip. Asserted on four seeds in both
        directions: without the reserve it crosses B every time, with the reserve it
        never does. The docstring in budgeted.py quotes the range this measures.
        """
        for seed in (404, 405, 406, 407):
            w = make_workload(12_000, seed=seed)
            budget = 0.70 * run_fleet(LinUCB(3, N_FEATURES, alpha=1.0), w).total_cost
            bare = BudgetedRouter(3, N_FEATURES, budget=budget, horizon=len(w),
                                  eta0=2.0, alpha=1.0, reserve=0.0)
            ratio = run_fleet(bare, w,
                              reward_override=w.success.astype(float)).total_cost / budget
            self.assertGreater(ratio, 1.0,
                               f"seed {seed}: spend/B = {ratio:.5f} did not cross B, so "
                               "the reserve is not buying anything and should be removed")
            self.assertLess(ratio, 1.005,
                            f"seed {seed}: spend/B = {ratio:.5f}; the overshoot is far "
                            "larger than the tracking lag, so the hard cap is broken")

            held = BudgetedRouter(3, N_FEATURES, budget=budget, horizon=len(w),
                                  eta0=2.0, alpha=1.0, reserve=0.01)
            kept = run_fleet(held, w,
                             reward_override=w.success.astype(float)).total_cost / budget
            self.assertLessEqual(kept, 1.0, f"seed {seed}: spend/B = {kept:.5f}")
            self.assertGreater(kept, 0.98, f"seed {seed}: spend/B = {kept:.5f}; the "
                                           "reserve is costing more than it should")

    def test_the_dual_price_tracks_the_budget_line(self):
        w = make_workload(12_000, seed=405)
        budget = 0.6 * run_fleet(LinUCB(3, N_FEATURES, alpha=1.0), w).total_cost
        router = BudgetedRouter(3, N_FEATURES, budget=budget, horizon=len(w),
                                eta0=2.0, alpha=1.0)
        run_fleet(router, w, reward_override=w.success.astype(float))
        spend = np.array(router.spend_history)
        line = budget * np.arange(1, len(w) + 1) / len(w)
        # After the transient, realized spend must stay close to the straight line.
        tail = slice(len(w) // 4, None)
        rel = np.abs(spend[tail] - line[tail]) / line[tail]
        self.assertLess(float(rel.max()), 0.05,
                        f"spend departs from the budget line by up to {rel.max():.3f}")
        self.assertGreater(router.price, 0.0, "the dual price never left zero")

    def test_tighter_budget_buys_less_quality(self):
        w = make_workload(12_000, seed=406)
        full = run_fleet(LinUCB(3, N_FEATURES, alpha=1.0), w).total_cost
        accs = []
        for frac in (0.45, 0.70, 0.95):
            r = BudgetedRouter(3, N_FEATURES, budget=frac * full, horizon=len(w),
                               eta0=2.0, alpha=1.0)
            accs.append(run_fleet(r, w,
                                  reward_override=w.success.astype(float)).success_rate)
        self.assertTrue(accs[0] < accs[1] < accs[2],
                        f"success rates {accs} are not increasing in the budget")

    def test_bad_construction_raises(self):
        with self.assertRaises(ValueError):
            BudgetedRouter(3, 4, budget=10.0, horizon=0)
        with self.assertRaises(ValueError):
            BudgetedRouter(3, 4, budget=10.0, horizon=100, reserve=1.5)


class TestSinglePriceIsOptimal(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # The structural result behind the whole budgeted router: dualising the single
    # coupling constraint decouples the problem, so the optimal policy has the form
    # "greedy on quality - p * cost" for some price p. That is a statement about a
    # multiple-choice knapsack, and it is checkable: solve a small instance exactly by
    # dynamic programming, then sweep p over the single-price family and compare.
    #
    # The LP relaxation's integrality gap for a multiple-choice knapsack is at most one
    # item, so the single-price policy must come within one query's quality of the exact
    # optimum. That is the tolerance -- it comes from the theorem, not from what made the
    # test pass.
    def test_single_price_matches_the_offline_knapsack_optimum(self):
        rng = np.random.default_rng(3)
        for trial in range(6):
            n, k = 40, 3
            quality = np.clip(rng.beta(2, 2, size=(n, k))
                              + np.array([0.0, 0.12, 0.24]), 0, 1)
            cost = np.array([0.5, 1.1, 2.8])[None, :] * rng.lognormal(0, 0.2, (n, k))
            budget = 0.55 * cost[:, 2].sum()

            dp_value, dp_arms = offline_knapsack_dp(quality, cost, budget, n_units=4_000)
            sp_value, price, sp_arms = best_single_price(quality, cost, budget)

            rows = np.arange(n)
            self.assertLessEqual(cost[rows, dp_arms].sum(), budget * 1.001)
            self.assertLessEqual(cost[rows, sp_arms].sum(), budget)
            self.assertGreaterEqual(price, 0.0)

            one_item = float(quality.max())
            self.assertGreaterEqual(
                sp_value, dp_value - one_item,
                f"trial {trial}: single price {sp_value:.4f} is more than one item "
                f"({one_item:.4f}) below the DP optimum {dp_value:.4f}")
            # And it must be genuinely close, not merely inside a loose bound.
            self.assertGreater(sp_value / dp_value, 0.985,
                               f"trial {trial}: single price captured only "
                               f"{sp_value / dp_value:.4f} of the optimum")

    def test_zero_price_spends_everything_and_a_high_price_spends_nothing(self):
        rng = np.random.default_rng(8)
        n = 30
        quality = rng.random((n, 3)) + np.array([0.0, 0.1, 0.2])
        cost = np.array([0.5, 1.0, 2.5])[None, :] * np.ones((n, 1))
        rows = np.arange(n)
        cheap = np.argmax(quality - 1e6 * cost, axis=1)
        rich = np.argmax(quality - 0.0 * cost, axis=1)
        self.assertTrue((cost[rows, cheap] == 0.5).all())
        self.assertGreaterEqual(cost[rows, rich].sum(), cost[rows, cheap].sum())

    def test_dp_respects_the_budget_exactly(self):
        rng = np.random.default_rng(12)
        quality = rng.random((25, 3))
        cost = np.array([0.4, 0.9, 2.0])[None, :] * np.ones((25, 1))
        budget = 20.0
        value, arms = offline_knapsack_dp(quality, cost, budget, n_units=2_000)
        self.assertLessEqual(float(cost[np.arange(25), arms].sum()), budget + 1e-9)
        self.assertAlmostEqual(float(quality[np.arange(25), arms].sum()), value, places=6)


if __name__ == "__main__":
    unittest.main()
