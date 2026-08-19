"""The statistics half of this project's anchor.

See the header of `tests/test_vad.py` for why the anchor is signal processing and
statistics rather than the deployment claim: the deployment claim needs a Raspberry
Pi 3, and there isn't one. What can be held to a real standard is that the harness
which will one day report the board's numbers computes its statistics correctly.
"""

import unittest

import numpy as np

from bench.latency import (NoCompletedTurns, percentile,
                          quantile_standard_error, run, summarise)
from src.stages import (LatencyModel, PLACEHOLDER_MODELS, SimulatedStage, Z_P50,
                        Z_P95)


def normal_pdf(z: float) -> float:
    return float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))


def lognormal_density_at_quantile(model: LatencyModel, z: float) -> float:
    """f_T(Q(p)) for T = floor + median * exp(sigma * Z).

    Y = log((T - floor)/median) ~ N(0, sigma^2), so by change of variables
    f_T(t) = phi(y/sigma) / (sigma * (t - floor)), evaluated at y = sigma*z.
    """
    return normal_pdf(z) / (model.sigma_log * model.median_ms *
                            float(np.exp(model.sigma_log * z)))


class TestPercentileArithmetic(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # A latency budget is only as trustworthy as its percentile function, and a
    # percentile function is easy to get subtly wrong: off-by-one in the order
    # statistic, the wrong interpolation convention, or silently sorting in place. So
    # the harness's p50 and p95 are checked against distributions whose quantiles are
    # known in closed form, with a tolerance derived from the asymptotic standard
    # error of a sample quantile,
    #
    #     SE = sqrt( p (1-p) / n ) / f(Q(p))
    #
    # rather than from a constant that happened to make the assertion pass. Three
    # distributions, chosen for different reasons:
    #   uniform      f is constant, so an interpolation-convention error has nowhere
    #                to hide in the density
    #   exponential  right-skewed with a thin density at p95, where the SE is large
    #                and an off-by-one is easiest to mistake for sampling noise
    #   lognormal    the distribution `SimulatedStage` actually draws from, so this
    #                also validates the simulator against its own closed form
    def test_percentiles_match_closed_form_quantiles(self):
        n = 200_000
        k_se = 5.0  # 5 standard errors: a false failure roughly once in 3 million

        # --- uniform(0, 1): Q(p) = p, f(Q(p)) = 1 ------------------------------
        rng = np.random.default_rng(20240817)
        u = rng.uniform(0.0, 1.0, n)
        for p in (0.5, 0.95):
            se = quantile_standard_error(p, n, density_at_quantile=1.0)
            got = percentile(u, 100 * p)
            self.assertLess(abs(got - p), k_se * se,
                            f"uniform p{100*p:g}: {got:.6f} vs {p} "
                            f"({abs(got - p)/se:.2f} SE)")

        # --- exponential(rate 1): Q(p) = -ln(1-p), f(Q(p)) = 1-p ---------------
        e = rng.exponential(1.0, n)
        for p in (0.5, 0.95):
            truth = float(-np.log1p(-p))
            se = quantile_standard_error(p, n, density_at_quantile=1.0 - p)
            got = percentile(e, 100 * p)
            self.assertLess(abs(got - truth), k_se * se,
                            f"exponential p{100*p:g}: {got:.5f} vs {truth:.5f} "
                            f"({abs(got - truth)/se:.2f} SE)")

        # --- the simulator's own lognormal -------------------------------------
        model = PLACEHOLDER_MODELS["llm_decode"]
        draws = model.draw(np.random.default_rng(7), n)
        for p, z in ((0.5, Z_P50), (0.95, Z_P95)):
            truth = model.quantile(z)
            se = quantile_standard_error(
                p, n, density_at_quantile=lognormal_density_at_quantile(model, z))
            got = percentile(draws, 100 * p)
            self.assertLess(abs(got - truth), k_se * se,
                            f"lognormal p{100*p:g}: {got:.3f} ms vs {truth:.3f} ms "
                            f"({abs(got - truth)/se:.2f} SE)")

    def test_percentile_interpolation_on_a_hand_computed_sample(self):
        """No sampling noise anywhere: these are the type-7 values by definition."""
        a = [0.0, 1.0, 2.0, 3.0, 4.0]      # n = 5, so h = 4p
        self.assertEqual(percentile(a, 0), 0.0)
        self.assertEqual(percentile(a, 100), 4.0)
        self.assertEqual(percentile(a, 50), 2.0)        # h = 2.0, exact order stat
        self.assertAlmostEqual(percentile(a, 95), 3.8)  # h = 3.8 -> 3 + 0.8*(4-3)
        self.assertAlmostEqual(percentile(a, 25), 1.0)
        # Unsorted input must give the same answer, and must not reorder the caller's
        # list -- a harness that sorts its input in place corrupts the sample it was
        # handed and every later statistic computed from it.
        shuffled = [3.0, 0.0, 4.0, 1.0, 2.0]
        self.assertAlmostEqual(percentile(shuffled, 95), 3.8)
        self.assertEqual(shuffled, [3.0, 0.0, 4.0, 1.0, 2.0])

    def test_agrees_with_numpy_on_random_samples(self):
        """A cross-check, not the proof: numpy's default is the same convention."""
        rng = np.random.default_rng(3)
        for n in (2, 3, 17, 1000):
            a = rng.standard_normal(n)
            for q in (0.0, 1.0, 25.0, 50.0, 95.0, 99.0, 100.0):
                self.assertAlmostEqual(percentile(a, q), float(np.percentile(a, q)),
                                       places=10, msg=f"n={n} q={q}")

    def test_degenerate_inputs_are_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], 101)
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], -1)
        with self.assertRaises(ValueError):
            quantile_standard_error(1.0, 100, 0.5)
        self.assertEqual(percentile([4.2], 95), 4.2)


class TestLatencyModel(unittest.TestCase):

    def test_closed_form_quantiles_match_the_draws(self):
        model = LatencyModel(median_ms=100.0, sigma_log=0.4, floor_ms=10.0)
        self.assertAlmostEqual(model.p50, 110.0)
        draws = model.draw(np.random.default_rng(1), 100_000)
        self.assertGreaterEqual(float(draws.min()), model.floor_ms)
        self.assertAlmostEqual(percentile(draws, 50), model.p50, delta=1.0)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            LatencyModel(median_ms=0.0)
        with self.assertRaises(ValueError):
            LatencyModel(median_ms=10.0, sigma_log=-0.1)


class TestHarness(unittest.TestCase):

    N_TURNS = 200

    @classmethod
    def setUpClass(cls):
        # setUpClass, not setUp: `run` executes a full pipeline per turn and there is
        # no reason to pay for it once per test method. 200 turns keeps the p95
        # standard error small enough for the tolerances below (which are computed
        # from this n, not hardcoded) while keeping the suite a few seconds long.
        cls.rows, cls.meta = run(n_turns=cls.N_TURNS, seed=0)
        cls.by_stage = {r["stage"]: r for r in cls.rows}

    def test_harness_percentiles_match_the_generating_distribution(self):
        """End-to-end check of the whole pipeline of measurement, not just the maths:
        the numbers that come out of 400 orchestrated turns must agree with the
        closed-form quantiles of the distributions the stages were configured with."""
        n = self.N_TURNS
        for name, model in PLACEHOLDER_MODELS.items():
            row = self.by_stage[name]
            for p, z, col in ((0.5, Z_P50, "median_ms"), (0.95, Z_P95, "p95_ms")):
                truth = model.quantile(z)
                se = quantile_standard_error(
                    p, n, density_at_quantile=lognormal_density_at_quantile(model, z))
                self.assertLess(
                    abs(row[col] - truth), 4.0 * se,
                    f"{name} {col}: {row[col]:.1f} vs analytic {truth:.1f} "
                    f"({abs(row[col]-truth)/se:.2f} SE over {n} turns)")

    def test_percentiles_do_not_add(self):
        """Summing per-stage p95s overstates the end-to-end p95 -- for THIS model.

        Not asserted as a theorem: sample quantiles are not subadditive in general
        (that failure is the standard argument against value-at-risk as a coherent
        risk measure), and comonotone or heavy-tailed stages can make the naive sum an
        understatement instead. What is asserted is the measured property of the
        independent lognormal model the harness runs, because the practical lesson --
        that a latency budget totalled by adding a p95 column is wrong and needs
        measuring end to end -- survives either way.
        """
        overstatement = self.meta["p95_overstatement"]
        self.assertGreater(overstatement, 1.0,
                           "sum of per-stage p95 did not exceed the end-to-end p95")
        self.assertLess(overstatement, 1.5,
                        f"overstatement {overstatement} is implausibly large; check "
                        "the stages are actually independent draws")
        # The medians, by contrast, nearly do add, because the median of a sum of
        # symmetric-ish independent terms is close to the sum of the medians. The
        # contrast between the two lines is the whole point.
        self.assertLess(
            abs(self.meta["sum_of_stage_median_ms"] / self.meta["e2e_median_ms"] - 1.0),
            0.05)

    def test_every_simulated_row_is_labelled_simulated(self):
        """The honesty rule, enforced by the suite rather than by good intentions."""
        for row in self.rows:
            if row["stage"] in PLACEHOLDER_MODELS or row["stage"] == "end_to_end":
                self.assertEqual(row["source"], "SIMULATED", f"unlabelled: {row}")
        for name in ("capture", "vad"):
            self.assertEqual(self.by_stage[name]["source"], "measured")

    def test_harness_overhead_is_measured_and_small(self):
        """The overhead column is a real measurement of this host and is the
        resolution floor of the instrument: a future per-stage figure from a Pi is
        only meaningful well above it."""
        overhead = self.meta["harness_overhead_median_ms"]
        self.assertGreater(overhead, 0.0, "a drawn stage still costs real dispatch "
                                          "time; zero means it was not measured")
        self.assertLess(overhead, 1.0,
                        f"{overhead} ms of harness overhead would contaminate any "
                        "sub-second stage measurement")

    def test_no_turns_failed(self):
        self.assertEqual(self.meta["failures"], 0)

    def test_summarise_reports_median_and_p95_not_only_a_mean(self):
        s = summarise([1.0, 2.0, 3.0, 100.0], "x")
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["median_ms"], 2.5)
        self.assertGreater(s["mean_ms"], s["median_ms"])  # the reason means are out


class TestRealHarnessSeam(unittest.TestCase):

    def test_real_mode_exists_and_fails_by_naming_what_is_missing(self):
        """`bench/latency.py --real` is the command that fills
        `results/latency_budget.md`, and both `setup/install.sh` and that file tell
        the reader to run it. So it has to exist, and on a machine with no inference
        binaries it has to fail in a way that says why -- not with an argparse error
        and not with a division by zero on an empty sample.

        This is also what makes the README's claim that pointing the harness at real
        binaries is "a constructor change and not a rewrite" checkable: the same
        `run()` with the same statistics is what raises below.
        """
        with self.assertRaises(NoCompletedTurns) as ctx:
            run(n_turns=2, real=True)
        message = str(ctx.exception)
        self.assertIn("all 2 turns failed", message)
        self.assertIn("setup/install.sh", message)
        # It must say which stage died, not merely that something did.
        self.assertIn("stage '", message)


class TestSimulatedStageIsLabelled(unittest.TestCase):

    def test_stage_result_carries_the_simulated_flag_to_the_record(self):
        stage = SimulatedStage("asr", LatencyModel(100.0), seed=0)
        rec = stage.run(None).to_record()
        self.assertEqual(rec["source"], "SIMULATED")
        self.assertTrue(rec["meta_SIMULATED"])
        self.assertEqual(stage.label, "SIMULATED asr")

    def test_drawn_latency_is_not_the_wall_clock(self):
        """With sleep=False the drawn latency and the real elapsed time must be
        different numbers in different fields. Conflating them is exactly the mistake
        that would turn a simulation into a fake measurement."""
        stage = SimulatedStage("llm_decode", LatencyModel(5000.0), seed=0)
        result = stage.run(None)
        self.assertGreater(result.latency_ms, 1000.0)      # drawn: seconds
        self.assertLess(result.harness_overhead_ms, 5.0)   # measured: microseconds
        self.assertTrue(result.simulated)


if __name__ == "__main__":
    unittest.main()
