"""The regret claim, and the numerical identity the routers depend on."""

import unittest

import numpy as np

from eval.regret import build_envelope
from eval.workload import (LinearBandit, loglog_slope, replay_fixed, run_fleet,
                           run_linear_bandit)
from src.features import N_FEATURES
from src.fleet.simulator import (LAMBDA, expected_reward_matrix, make_workload,
                                 oracle_expected)
from src.routers.baselines import FixedArm, RandomRouter, ThresholdRouter
from src.routers.linucb import LinGreedy, LinUCB
from src.routers.thompson import LinearThompson

# A reduced version of the configuration in eval/regret.py -- d = 10 instead of 20, 4
# arms instead of 5, 8 gap scales instead of 12, T = 16,000 instead of 64,000 -- but NOT
# reduced in the number of seeds, which turned out to be the parameter that matters.
#
# At 2 seeds this configuration measures 0.44 / 0.47 / 0.54 / 0.63 depending on which
# instance family you draw (seed0 = 100 / 200 / 300 / 400): the envelope is a maximum
# over 8 noisy curves, and a max over noisy things is both biased and jittery. Pinning
# the seed would have hidden that behind a number that happened to pass. At 5 seeds the
# same four families give 0.469 / 0.518 / 0.505 / 0.514, a spread of +-0.025 around 0.5,
# with R^2 between 0.989 and 0.997. That is the fix: average the curves before taking
# the max, not widen the tolerance until the noisy version fits inside it.
#
# The cost is about 60 s of the suite's runtime. That is the right trade -- the whole
# repo rests on this exponent -- and the runtime is stated in the README rather than
# quietly paid for by shrinking T.
TEST_D, TEST_ARMS, TEST_SIGMA = 10, 4, 1.0
TEST_HORIZON = 16_000
TEST_GAPS = np.logspace(-1.8, 0.9, 8)
TEST_SEEDS = 5
FIT_LO = 1_000


def _envelope(factory):
    return build_envelope(factory, d=TEST_D, n_arms=TEST_ARMS, sigma=TEST_SIGMA,
                          gaps=TEST_GAPS, horizon=TEST_HORIZON, n_seeds=TEST_SEEDS,
                          seed0=100)


class TestRegretExponent(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # Otilde(d sqrt(T)) is a MINIMAX bound: a statement about the worst instance at each
    # horizon. On any single fixed instance LinUCB's regret leaves the sqrt(T) regime and
    # goes gap-dependent and logarithmic, and the measured local slope drifts smoothly
    # from 1 down towards 0 -- so a slope read off one instance can be made to equal
    # anything in (0,1) by choosing the fit window. (test_fixed_instance_slope_drifts
    # below asserts that drift, so this is a measured statement, not a claim.)
    #
    # What is tested here is therefore the supremum over an instance family indexed by
    # a gap scale, which is the quantity the theorem actually bounds:
    #
    #     R*(T) = max over Delta of E[ regret at horizon T on instance Delta ].
    #
    # For a learner, R(T; Delta) ~ min(c1 Delta T, c2 d sqrt(T)), so the max sits where
    # the two branches meet, at Delta* ~ 1/sqrt(T), and R*(T) ~ sqrt(T): exponent 1/2.
    # For a policy that does not learn, R(T; Delta) = c Delta T for every Delta, the max
    # is at the largest gap in the family, and the exponent is 1.
    #
    # The random policy is the control. It runs the identical protocol on the identical
    # family through the identical fitting code, and it must come out at 1.0. If both
    # came out at 0.5 the measurement would be an artefact of the fit; if both came out
    # at 1.0 the bandit would not be learning. The pair is the evidence.
    def test_minimax_regret_exponent_is_one_half_and_random_is_one(self):
        env_ucb, argmax_ucb = _envelope(lambda k, d: LinUCB(k, d, alpha=1.0))
        env_rnd, _ = _envelope(lambda k, d: RandomRouter(k, seed=23))

        slope_ucb, r2_ucb, _ = loglog_slope(env_ucb, FIT_LO, TEST_HORIZON)
        slope_rnd, r2_rnd, _ = loglog_slope(env_rnd, FIT_LO, TEST_HORIZON)

        # Straightness first: a "slope" fitted to a curve that is not straight in
        # log-log is not a slope, and the tolerance below would be meaningless.
        self.assertGreater(r2_ucb, 0.985, f"LinUCB envelope is not a power law "
                                          f"over [{FIT_LO},{TEST_HORIZON}]: R2={r2_ucb:.4f}")
        self.assertGreater(r2_rnd, 0.99, f"random envelope R2={r2_rnd:.4f}")

        # The claim. +-0.075 around 1/2 at this reduced size, which covers the +-0.025
        # spread measured across four independent instance families with room to spare;
        # the full-size run in eval/regret.py sits at 0.475 with R2 = 0.9994.
        self.assertAlmostEqual(slope_ucb, 0.5, delta=0.075,
                               msg=f"LinUCB minimax regret exponent {slope_ucb:.3f} is "
                                   "not the 1/2 the Otilde(d sqrt(T)) bound predicts")
        # The control, which must be sensitive to the policy and not to the fitting.
        self.assertAlmostEqual(slope_rnd, 1.0, delta=0.08,
                               msg=f"random policy exponent {slope_rnd:.3f} is not 1; "
                                   "the measurement is not detecting the policy")
        self.assertGreater(slope_rnd - slope_ucb, 0.35,
                           "learner and non-learner are not separated")

        # The diagnostic that the supremum is real rather than a grid edge: the
        # maximising gap must be interior to the family and must shrink with T, as
        # Delta* ~ 1/sqrt(T) predicts.
        lo_gap = TEST_GAPS[argmax_ucb[FIT_LO - 1]]
        hi_gap = TEST_GAPS[argmax_ucb[TEST_HORIZON - 1]]
        self.assertTrue(0 < argmax_ucb[TEST_HORIZON - 1] < len(TEST_GAPS) - 1,
                        "the worst-case gap is on the edge of the family: the grid is "
                        "truncating the supremum and the exponent is biased")
        self.assertLess(hi_gap, lo_gap,
                        f"worst-case gap did not shrink with T ({lo_gap:.3f} -> "
                        f"{hi_gap:.3f}); the envelope is not tracking Delta* ~ 1/sqrt(T)")

    def test_fixed_instance_slope_drifts_and_therefore_proves_nothing(self):
        """The control for the control: one instance gives you whatever slope you ask for.

        This is why the test above measures a supremum. If this assertion ever failed --
        if a fixed instance did hold a constant exponent over two decades -- the framing
        in eval/regret.py would be wrong and would need rewriting.
        """
        inst = LinearBandit(TEST_D, TEST_ARMS, TEST_SIGMA, gap_scale=1.0, seed=100)
        curve = np.mean([run_linear_bandit(lambda k, d: LinUCB(k, d, alpha=1.0),
                                           inst, TEST_HORIZON, seed=1000 + s)
                         for s in range(TEST_SEEDS)], axis=0)
        early, _, _ = loglog_slope(curve, 100, 1_000, n_points=30)
        late, _, _ = loglog_slope(curve, 1_600, 16_000, n_points=30)
        self.assertGreater(early - late, 0.15,
                           f"fixed-instance local slope did not drift "
                           f"({early:.3f} -> {late:.3f})")


class TestShermanMorrison(unittest.TestCase):

    def test_rank_one_inverse_updates_match_explicit_reinversion(self):
        """(A + x x^T)^{-1} by Sherman-Morrison against numpy.linalg.inv, after 600 updates.

        The whole reason LinUCB is O(T d^2) rather than O(T d^3) is that this identity
        holds; if it drifts, every confidence width is wrong and the regret curve is
        measuring a different algorithm.
        """
        rng = np.random.default_rng(5)
        d, k = 12, 3
        router = LinUCB(k, d, alpha=1.0, ridge=1.0)
        for _ in range(600):
            x = rng.normal(size=d)
            x /= np.linalg.norm(x)
            arm = int(rng.integers(k))
            router.update(x, arm, float(rng.normal()), None)
        worst = 0.0
        for arm in range(k):
            exact = router.reinverted(arm)
            rel = np.linalg.norm(router.a_inv[arm] - exact) / np.linalg.norm(exact)
            worst = max(worst, float(rel))
        self.assertLess(worst, 1e-8,
                        f"Sherman-Morrison drifted from the explicit inverse by {worst:.2e}")
        # Printed rather than only asserted, so the README can quote the measured value
        # instead of the word "machine precision".
        self.assertLess(worst, 1e-10, f"measured relative agreement {worst:.3e}")

    def test_design_matrix_stays_symmetric_positive_definite(self):
        rng = np.random.default_rng(9)
        router = LinUCB(2, 6, ridge=1.0)
        for _ in range(200):
            x = rng.normal(size=6)
            router.update(x / np.linalg.norm(x), 0, 1.0, None)
        eig = np.linalg.eigvalsh(router.a_inv[0])
        self.assertGreater(eig.min(), 0.0, "A^{-1} lost positive definiteness")
        self.assertLess(float(np.abs(router.a_inv[0] - router.a_inv[0].T).max()), 1e-12)

    def test_ridge_must_be_positive(self):
        with self.assertRaises(ValueError):
            LinUCB(3, 4, ridge=0.0)


class TestOracleIsNeverBeaten(unittest.TestCase):

    # The invariant that makes every regret number in the repo meaningful. The oracle
    # maximises E[r | x, a] arm by arm, so no policy can exceed it in expectation -- and
    # it is asserted on that scalar, not on raw success rate, because a policy is
    # entitled to beat the oracle's success rate by overspending. If this ever fails,
    # either the oracle or the regret definition is wrong, and every other number here
    # is suspect.
    def test_no_policy_beats_the_expected_reward_oracle(self):
        w = make_workload(8_000, seed=77)
        tune = make_workload(4_000, seed=78)
        threshold = ThresholdRouter.fit(tune.difficulty_score,
                                        expected_reward_matrix(tune, LAMBDA),
                                        score_index=6)
        oracle = replay_fixed(oracle_expected(w), w, name="oracle")
        best = float(oracle.expected_reward.mean())
        policies = [FixedArm(0), FixedArm(1), FixedArm(2), RandomRouter(3, seed=1),
                    threshold, LinUCB(3, N_FEATURES, alpha=1.0),
                    LinGreedy(3, N_FEATURES),
                    LinearThompson(3, N_FEATURES, v=0.35, seed=2)]
        for pol in policies:
            run = run_fleet(pol, w)
            self.assertLessEqual(float(run.expected_reward.mean()), best + 1e-12,
                                 f"{run.name} beat the oracle in expectation")
            self.assertGreaterEqual(float(run.regret[-1]), -1e-9,
                                    f"{run.name} has negative cumulative regret")

    def test_oracle_regret_is_identically_zero(self):
        w = make_workload(4_000, seed=79)
        oracle = replay_fixed(oracle_expected(w), w)
        self.assertLess(float(np.abs(oracle.regret).max()), 1e-9)


class TestRoutersLearn(unittest.TestCase):

    def test_linucb_beats_random_on_the_fleet(self):
        w = make_workload(12_000, seed=88)
        ucb = run_fleet(LinUCB(3, N_FEATURES, alpha=1.0), w)
        rnd = run_fleet(RandomRouter(3, seed=3), w)
        self.assertLess(float(ucb.regret[-1]), 0.5 * float(rnd.regret[-1]),
                        "LinUCB did not halve the random policy's regret")

    def test_thompson_posterior_width_matches_linucb_width(self):
        """Both routers carry the same A^{-1}; only the way they use it differs.

        Thompson's posterior standard deviation in direction x is v sqrt(x^T A^{-1} x),
        which is LinUCB's confidence width up to the constant. Checking that the two
        classes agree on that quantity after identical data is what makes the claim in
        thompson.py's docstring a fact about this code.
        """
        rng = np.random.default_rng(4)
        d = 8
        ucb = LinUCB(2, d, alpha=1.0)
        ts = LinearThompson(2, d, v=1.0, seed=0)
        for _ in range(120):
            x = rng.normal(size=d)
            x /= np.linalg.norm(x)
            r = float(rng.normal())
            ucb.update(x, 0, r, None)
            ts.update(x, 0, r, None)
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)
        width_ucb = float(np.sqrt(x @ ucb.a_inv[0] @ x))
        width_ts = float(np.linalg.norm(ts.chol[0].T @ x))
        self.assertAlmostEqual(width_ucb, width_ts, places=10)


if __name__ == "__main__":
    unittest.main()
