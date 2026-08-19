"""The instrument itself: if the simulator is wrong, nothing measured on it is right."""

import unittest

import numpy as np

from src.cost import CallCost, ComputeCostModel, TokenPriceCostModel
from src.features import (FEATURE_NAMES, N_FEATURES, SERVING_AVAILABLE, featurize)
from src.fleet.client import OllamaClient, load_fleet_spec
from src.fleet.simulator import (B0, BETA, DEFAULT_FLEET, LAMBDA, N_LABELS,
                                 cheapest_sufficient, expected_reward_matrix,
                                 make_workload, oracle_expected, oracle_hindsight,
                                 success_probability)


class TestCapabilityLadder(unittest.TestCase):

    def test_success_probability_is_increasing_in_capability_everywhere(self):
        """A ladder, not a coin flip: p_small(d) < p_mid(d) < p_large(d) for every d."""
        d = np.linspace(0.0, 1.0, 501)
        p = success_probability(d, DEFAULT_FLEET)
        self.assertTrue((np.diff(p, axis=1) > 0).all(),
                        "the arms are not strictly ordered by capability at some "
                        "difficulty; there would be nothing to route")

    def test_success_probability_is_decreasing_in_difficulty(self):
        d = np.linspace(0.0, 1.0, 501)
        p = success_probability(d, DEFAULT_FLEET)
        self.assertTrue((np.diff(p, axis=0) < 0).all())

    def test_cost_is_increasing_in_capability(self):
        base = [a.base_seconds for a in DEFAULT_FLEET]
        rate = [a.seconds_per_token for a in DEFAULT_FLEET]
        mem = [a.peak_memory_gb for a in DEFAULT_FLEET]
        for series in (base, rate, mem):
            self.assertTrue(all(x < y for x, y in zip(series, series[1:])), series)

    def test_realized_success_matches_the_stated_marginals(self):
        """The correlated coupling must not move the marginals.

        The uniform mixture is only legitimate if every arm still succeeds with
        probability exactly p_k(d). If it shifted them, the oracle, the regret and the
        Pareto table would all be computed against the wrong quantity.
        """
        w = make_workload(60_000, seed=17)
        realized = w.success.mean(axis=0)
        stated = w.p_success.mean(axis=0)
        # 60k draws: standard error of the mean is under 0.002 per arm.
        np.testing.assert_allclose(realized, stated, atol=0.006)

    def test_failures_are_correlated_across_arms_at_two_separate_levels(self):
        """Two distinct sources of dependence, and rho controls only the second.

        Even at rho = 0 the arms' failures are positively correlated *marginally*,
        because they share the query's difficulty: hard queries are hard for everyone.
        Conditioning on a narrow difficulty band removes that channel, and what is left
        at rho = 0 is genuine independence -- which is the check below. rho then adds a
        second, conditional dependence on top: a shared uniform draw, i.e. arms failing
        together on the *same* query even at fixed difficulty.

        Getting this distinction wrong is easy and consequential: an unconditional
        independence check on this simulator fails at rho = 0 (measured ratio 4.2), and
        reading that as a bug would mean 'fixing' the shared-difficulty structure that
        the whole routing problem is built on.
        """
        def conditional_ratio(w, n_bins=40):
            """P(both arms fail | difficulty bin) / product of the per-arm rates.

            Measured on the small/mid pair, not on all three arms: the all-three-fail
            event has probability ~0.3 * 0.13 * 0.02 = 8e-4, so within a bin its
            estimate is a handful of counts and the ratio is dominated by its own noise.
            The pair event is ~4e-2 and estimates cleanly.
            """
            edges = np.quantile(w.difficulty, np.linspace(0, 1, n_bins + 1))
            bins = np.clip(np.digitize(w.difficulty, edges[1:-1]), 0, n_bins - 1)
            ratios = []
            for b in range(n_bins):
                fails = ~w.success[bins == b][:, :2]
                joint = float(fails.all(axis=1).mean())
                prod = float(np.prod(fails.mean(axis=0)))
                if prod > 1e-3:
                    ratios.append(joint / prod)
            return float(np.mean(ratios))

        indep = make_workload(60_000, seed=18, rho=0.0)
        corr = make_workload(60_000, seed=18, rho=0.9)
        self.assertLess(abs(conditional_ratio(indep) - 1.0), 0.12,
                        "at rho=0 the arms must be conditionally independent given "
                        "difficulty")
        self.assertGreater(conditional_ratio(corr), 2.5,
                           "rho=0.9 did not add conditional dependence on top of the "
                           "shared-difficulty channel")

        # And the marginal channel is present regardless of rho -- this is the fact the
        # compounding experiment turns on, so it is pinned here.
        marginal = float((~indep.success).all(axis=1).mean()) / float(
            np.prod((~indep.success).mean(axis=0)))
        self.assertGreater(marginal, 2.0,
                           "shared difficulty is not producing marginal correlation")

    def test_routing_problem_is_non_trivial(self):
        """Every arm must be the oracle's choice on a real share of queries.

        A degenerate fleet -- one arm optimal everywhere -- would make every policy in
        the repo look identical and every comparison meaningless. This is a property of
        the instrument and it is asserted rather than assumed.
        """
        w = make_workload(20_000, seed=19)
        shares = np.bincount(oracle_expected(w), minlength=3) / len(w)
        self.assertTrue((shares > 0.10).all(),
                        f"oracle arm shares {shares.round(3)} are degenerate")

    def test_hindsight_oracle_is_the_cheapest_sufficient_arm(self):
        """The two phrasings of the oracle must coincide when lambda*(c_max-c_min) < 1."""
        w = make_workload(20_000, seed=20)
        costs = w.cost_matrix()
        spread = float(costs.max(axis=1).mean() - costs.min(axis=1).mean())
        self.assertLess(LAMBDA * spread, 1.0)
        agree = float((oracle_hindsight(w) == cheapest_sufficient(w)).mean())
        self.assertGreater(agree, 0.999,
                           f"argmax-realized-reward and cheapest-sufficient agree on "
                           f"only {agree:.4f} of queries")

    def test_emitted_argmax_agrees_with_the_success_indicator(self):
        """Accuracy and confidence come from one draw, so they cannot disagree."""
        w = make_workload(5_000, seed=21)
        emitted = w.probs.argmax(axis=2)
        np.testing.assert_array_equal(emitted == w.label[:, None], w.success)

    def test_probabilities_are_normalised(self):
        w = make_workload(2_000, seed=22)
        np.testing.assert_allclose(w.probs.sum(axis=2), 1.0, atol=1e-12)
        self.assertEqual(w.probs.shape[2], N_LABELS)

    def test_difficulty_shift_makes_the_workload_easier(self):
        easy = make_workload(10_000, seed=23, difficulty_shift=-0.35)
        hard = make_workload(10_000, seed=23, difficulty_shift=0.25)
        self.assertTrue((easy.success.mean(axis=0) > hard.success.mean(axis=0)).all())

    def test_seeding_is_reproducible_and_seed_dependent(self):
        a = make_workload(1_000, seed=24)
        b = make_workload(1_000, seed=24)
        c = make_workload(1_000, seed=25)
        np.testing.assert_array_equal(a.success, b.success)
        self.assertFalse(np.array_equal(a.success, c.success))


class TestFeatures(unittest.TestCase):

    def test_every_feature_is_available_at_serving_time(self):
        self.assertEqual(set(SERVING_AVAILABLE), set(FEATURE_NAMES))
        self.assertTrue(all(SERVING_AVAILABLE.values()))
        self.assertNotIn("difficulty", FEATURE_NAMES)
        self.assertNotIn("label", FEATURE_NAMES)

    def test_featurize_shape_and_scaling(self):
        w = make_workload(3_000, seed=26)
        x = featurize(w)
        self.assertEqual(x.shape, (3_000, N_FEATURES))
        self.assertTrue(np.isfinite(x).all())
        # Every column order one, so a single ridge constant means the same thing in
        # every direction.
        self.assertLess(float(np.abs(x).max()), 4.0)
        np.testing.assert_allclose(x[:, 0], 1.0)
        np.testing.assert_allclose(x[:, 2:5].sum(axis=1), 1.0)

    def test_difficulty_score_is_the_only_view_of_difficulty(self):
        w = make_workload(20_000, seed=27)
        corr = float(np.corrcoef(w.difficulty, w.difficulty_score)[0, 1])
        self.assertGreater(corr, 0.85, "the classifier carries too little signal")
        self.assertLess(corr, 0.999, "the classifier is noiseless, which would make "
                                     "the routing problem trivially solvable")
        self.assertEqual(FEATURE_NAMES[6], "difficulty_score")


class TestCostModel(unittest.TestCase):

    def test_compute_cost_reduces_to_seconds_by_default(self):
        c = CallCost(seconds=1.5, peak_memory_gb=18.6, prompt_tokens=100,
                     completion_tokens=50)
        self.assertAlmostEqual(ComputeCostModel().price(c), 1.5)
        self.assertGreater(ComputeCostModel(memory_weight=0.05).price(c), 1.5)

    def test_token_price_model_swaps_in_behind_the_same_interface(self):
        model = TokenPriceCostModel(prompt_usd_per_1k={"mid": 0.2},
                                    completion_usd_per_1k={"mid": 0.6},
                                    arm_name="mid")
        c = CallCost(seconds=99.0, peak_memory_gb=5.4, prompt_tokens=1_000,
                     completion_tokens=500)
        # Seconds are irrelevant to a per-token price; that is the point of the swap.
        self.assertAlmostEqual(model.price(c), 0.2 + 0.3)

    def test_cost_matrix_agrees_with_the_scalar_path(self):
        w = make_workload(200, seed=28)
        model = ComputeCostModel(memory_weight=0.03)
        matrix = w.cost_matrix(model)
        for t in (0, 17, 199):
            for k in range(3):
                self.assertAlmostEqual(matrix[t, k], model.price(w.call_cost(t, k)),
                                       places=12)

    def test_expected_cost_is_the_noise_free_part_of_realized_cost(self):
        w = make_workload(20_000, seed=29)
        ratio = (w.cost_matrix() / w.expected_cost_matrix()).mean(axis=0)
        # Lognormal(0, 0.15) has mean exp(0.15^2/2) = 1.0113.
        np.testing.assert_allclose(ratio, np.exp(0.15 ** 2 / 2), atol=0.01)

    def test_expected_reward_uses_expected_not_realized_cost(self):
        w = make_workload(500, seed=30)
        mu = expected_reward_matrix(w, LAMBDA)
        np.testing.assert_allclose(mu, w.p_success - LAMBDA * w.expected_cost_matrix())


class TestRealFleetClient(unittest.TestCase):

    def test_ollama_client_fails_loudly_and_names_what_it_needs(self):
        client = OllamaClient(("small", "mid", "large"))
        with self.assertRaises(NotImplementedError) as ctx:
            client.generate("hello", 0)
        msg = str(ctx.exception)
        for token in ("ollama", "serve/fleet.yaml", "simulator"):
            self.assertIn(token, msg,
                          f"the error message does not mention {token!r}: {msg}")

    def test_fleet_spec_parses_and_declares_a_ladder(self):
        import pathlib
        spec = load_fleet_spec(str(pathlib.Path(__file__).resolve().parents[1]
                                   / "serve" / "fleet.yaml"))
        tiers = [a["tier"] for a in spec["arms"]]
        self.assertEqual(tiers, ["small", "mid", "large"])
        self.assertIn(spec["config"], ("A", "B"))


class TestConstantsAreTheOnesDocumented(unittest.TestCase):

    def test_ladder_constants(self):
        self.assertEqual((BETA, B0), (6.0, 2.2))
        self.assertEqual(LAMBDA, 0.12)
        self.assertEqual([a.name for a in DEFAULT_FLEET], ["small", "mid", "large"])


if __name__ == "__main__":
    unittest.main()
