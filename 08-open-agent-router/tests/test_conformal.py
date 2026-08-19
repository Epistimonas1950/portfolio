"""The coverage guarantee, its composition across tiers, and the premise it rests on."""

import unittest

import numpy as np

from src.conformal.calibrate import (conformal_quantile, covered,
                                     empirical_coverage, nonconformity,
                                     prediction_sets, set_sizes, split_conformal)
from src.conformal.cascade import (build_cascade, run_cascade, split_budget)
from src.fleet.simulator import make_workload

ALPHAS = (0.01, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)


class TestSplitConformalCoverage(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # Split conformal's promise is P(Y in C(X)) >= 1 - alpha in finite samples, for any
    # distribution, assuming only exchangeability. It is asserted here across the whole
    # alpha sweep, for all three arms, on held-out data, averaged over 12 random
    # calibration/test partitions.
    #
    # The averaging is the part that makes this a test of the mathematics rather than of
    # one lucky draw: a single split has empirical coverage with standard error
    # sqrt(alpha(1-alpha)/n_test) ~ 0.004 at alpha = 0.1 and n_test = 6000, so a bare
    # >= 1 - alpha assertion on one split fails about half the time by construction. The
    # slack below is 0.006, which is about 3 standard errors of the 12-split mean, and it
    # is stated rather than tuned until green.
    #
    # The upper bound is asserted too. Coverage must not merely exceed 1 - alpha; under
    # continuous scores it must sit inside [1 - alpha, 1 - alpha + 1/(n+1)]. A method
    # that returned the full label set every time would pass the lower bound and be
    # useless, so the two-sided check is what rules that out.
    def test_empirical_coverage_meets_the_finite_sample_bound(self):
        w = make_workload(24_000, seed=31_337)
        rng = np.random.default_rng(4)
        n_cal, n_splits = 4_000, 12
        slack = 0.006
        for k, arm in enumerate(w.arms):
            for alpha in ALPHAS:
                cov = []
                for _ in range(n_splits):
                    perm = rng.permutation(len(w))
                    cal, test = perm[:n_cal], perm[n_cal:]
                    c = split_conformal(w.probs[cal, k, :], w.label[cal], alpha)
                    self.assertFalse(c.degenerate)
                    cov.append(empirical_coverage(w.probs[test, k, :],
                                                  w.label[test], c))
                mean_cov = float(np.mean(cov))
                self.assertGreaterEqual(
                    mean_cov, 1.0 - alpha - slack,
                    f"arm {arm.name} at alpha={alpha}: coverage {mean_cov:.4f} is below "
                    f"the guaranteed {1 - alpha:.4f} by more than the stated slack")
                self.assertLessEqual(
                    mean_cov, 1.0 - alpha + 1.0 / (n_cal + 1) + slack,
                    f"arm {arm.name} at alpha={alpha}: coverage {mean_cov:.4f} exceeds "
                    "the upper bound; the sets are wider than the guarantee requires")

    def test_quantile_is_an_order_statistic_not_an_interpolation(self):
        """q_hat must be a calibration score itself; interpolating voids the bound."""
        rng = np.random.default_rng(1)
        scores = rng.random(200)
        cal = conformal_quantile(scores, 0.1)
        self.assertIn(cal.q_hat, set(scores.tolist()))
        k = int(np.ceil(201 * 0.9))
        self.assertEqual(cal.rank, k)
        self.assertEqual(cal.q_hat, float(np.sort(scores)[k - 1]))
        self.assertGreaterEqual(cal.guaranteed_coverage, 0.9)

    def test_too_few_calibration_points_is_flagged_not_faked(self):
        """k = ceil((n+1)(1-alpha)) > n has no valid threshold. Say so."""
        rng = np.random.default_rng(2)
        cal = conformal_quantile(rng.random(50), 0.01)   # needs n >= 99
        self.assertTrue(cal.degenerate)
        self.assertEqual(cal.q_hat, float("inf"))
        mask = prediction_sets(rng.random((10, 8)), cal)
        self.assertTrue(mask.all(), "degenerate calibration must give the full label set")
        self.assertEqual(cal.guaranteed_coverage, 1.0)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            conformal_quantile(np.array([]), 0.1)
        with self.assertRaises(ValueError):
            conformal_quantile(np.random.default_rng(0).random(10), 1.5)
        with self.assertRaises(ValueError):
            nonconformity(np.zeros((2, 3, 4)), np.zeros(2, dtype=int))


class TestExchangeabilityIsThePremise(unittest.TestCase):

    # The other half of the guarantee: it is conditional on exchangeability, and when
    # that fails the bound fails -- loudly. Calibrating on easy queries and testing on
    # hard ones must break coverage. If it did not, coverage would not be coming from
    # the mathematics claimed, and the guarantee would be a coincidence of this
    # simulator.
    def test_coverage_breaks_under_deliberate_distribution_shift(self):
        cal_w = make_workload(8_000, seed=61, difficulty_shift=-0.35)
        test_w = make_workload(8_000, seed=62, difficulty_shift=0.25)
        for k, arm in enumerate(test_w.arms):
            for alpha in (0.05, 0.10, 0.20):
                c = split_conformal(cal_w.probs[:, k, :], cal_w.label, alpha)
                cov = empirical_coverage(test_w.probs[:, k, :], test_w.label, c)
                self.assertLess(cov, 1.0 - alpha - 0.05,
                                f"arm {arm.name} at alpha={alpha}: coverage {cov:.3f} "
                                "survived a shift it should not have; either the shift "
                                "is too small or the sets are trivially wide")

    def test_the_same_calibrator_is_fine_without_the_shift(self):
        """The control: same code, exchangeable data, coverage holds."""
        w = make_workload(8_000, seed=63, difficulty_shift=-0.35)
        for k in range(3):
            c = split_conformal(w.probs[:4_000, k, :], w.label[:4_000], 0.10)
            cov = empirical_coverage(w.probs[4_000:, k, :], w.label[4_000:], c)
            self.assertGreater(cov, 0.88)


class TestCascadeComposition(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # The end-to-end guarantee of the multi-tier cascade. A miss by the cascade is a miss
    # by whichever tier answered, so it is contained in the union of the per-tier miss
    # events, and the union bound gives miscoverage <= sum_i alpha_i. Asserted across the
    # alpha sweep, on both budget splits and both workloads.
    def test_composed_miscoverage_never_exceeds_the_union_bound(self):
        for shift, tag in ((0.0, "nominal"), (-0.35, "easy")):
            cal_w = make_workload(8_000, seed=71, difficulty_shift=shift)
            test_w = make_workload(12_000, seed=72, difficulty_shift=shift)
            for alpha in ALPHAS:
                for split in (split_budget(alpha, 3),
                              (0.7 * alpha, 0.2 * alpha, 0.1 * alpha)):
                    self.assertAlmostEqual(sum(split), alpha, places=12)
                    tiers = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2), split)
                    rep = run_cascade(test_w.probs, test_w.label, tiers,
                                      costs=test_w.cost_matrix())
                    self.assertLessEqual(
                        rep.empirical_miscoverage, rep.union_bound + 1e-12,
                        f"{tag} alpha={alpha} split={split}: end-to-end miscoverage "
                        f"{rep.empirical_miscoverage:.4f} exceeded the union bound "
                        f"{rep.union_bound:.4f}")
                    self.assertGreaterEqual(rep.slack, 0.0)

    def test_the_union_bound_is_loose_and_the_slack_is_measured(self):
        """Not a smoke test: the bound must be loose, and for the reason claimed.

        Miss events are strongly positively correlated across tiers (a hard query is
        missed by everyone), so the sum of the per-tier miscoverages must exceed the
        miscoverage of their union by a wide margin.
        """
        cal_w = make_workload(8_000, seed=73)
        test_w = make_workload(12_000, seed=74)
        tiers = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2), split_budget(0.2, 3))
        rep = run_cascade(test_w.probs, test_w.label, tiers,
                          costs=test_w.cost_matrix())
        self.assertGreater(rep.slack, 0.05,
                           f"slack {rep.slack:.4f}: the union bound is unexpectedly "
                           "tight, which would mean the tiers fail independently")
        # And the per-tier misses really are overlapping, not disjoint.
        self.assertLess(rep.empirical_miscoverage, sum(rep.per_tier_miscoverage))

    def test_more_error_budget_at_a_tier_lets_it_answer_more(self):
        """The structural condition: a tier can only answer when alpha_i exceeds its
        error rate, because otherwise the (1 - alpha_i) calibration quantile lands inside
        the scores of its own wrong answers and every set comes back wide."""
        cal_w = make_workload(8_000, seed=75, difficulty_shift=-0.35)
        test_w = make_workload(12_000, seed=76, difficulty_shift=-0.35)
        err0 = float(1.0 - test_w.success[:, 0].mean())
        below = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2),
                              (0.4 * err0, 0.1, 0.1))
        above = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2),
                              (2.5 * err0, 0.1, 0.1))
        r_below = run_cascade(test_w.probs, test_w.label, below)
        r_above = run_cascade(test_w.probs, test_w.label, above)
        self.assertGreater(r_above.accept_rate[0], r_below.accept_rate[0] + 0.2,
                           f"tier-0 acceptance {r_below.accept_rate[0]:.3f} -> "
                           f"{r_above.accept_rate[0]:.3f} across its own error rate "
                           f"{err0:.3f}: the structural condition is not visible")

    def test_escalated_queries_pay_for_every_tier_they_visit(self):
        cal_w = make_workload(4_000, seed=77)
        test_w = make_workload(4_000, seed=78)
        costs = test_w.cost_matrix()
        tiers = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2), split_budget(0.2, 3))
        rep = run_cascade(test_w.probs, test_w.label, tiers, costs=costs)
        self.assertGreater(rep.mean_cost, costs[:, 2].mean(),
                           "a cascade that reaches the last tier for most queries must "
                           "cost more than calling the last tier directly")

    def test_mismatched_tier_and_alpha_counts_raise(self):
        w = make_workload(500, seed=79)
        with self.assertRaises(ValueError):
            build_cascade(w.probs, w.label, (0, 1, 2), (0.1, 0.1))


class TestSetsAreUsable(unittest.TestCase):

    def test_set_size_shrinks_with_capability(self):
        w = make_workload(8_000, seed=81)
        sizes = []
        for k in range(3):
            c = split_conformal(w.probs[:4_000, k, :], w.label[:4_000], 0.10)
            sizes.append(float(set_sizes(prediction_sets(w.probs[4_000:, k, :],
                                                         c)).mean()))
        self.assertTrue(sizes[0] > sizes[1] > sizes[2],
                        f"mean set sizes {sizes} are not decreasing in capability")

    def test_covered_agrees_with_the_score_threshold(self):
        w = make_workload(2_000, seed=82)
        c = split_conformal(w.probs[:1_000, 1, :], w.label[:1_000], 0.1)
        mask = prediction_sets(w.probs[1_000:, 1, :], c)
        direct = nonconformity(w.probs[1_000:, 1, :], w.label[1_000:]) <= c.q_hat
        np.testing.assert_array_equal(covered(mask, w.label[1_000:]), direct)


if __name__ == "__main__":
    unittest.main()
