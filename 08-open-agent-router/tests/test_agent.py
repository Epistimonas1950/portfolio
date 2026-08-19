"""The tools do real work, the loop is a real loop, and p^n is checked against it."""

import pathlib
import unittest

import numpy as np

from src.agent.loop import (ensure_corpus, make_episode, run_episode)
from src.agent.tools import (ToolCall, ToolError, calculate, dispatch, read_file,
                             search_text)
from src.features import N_FEATURES
from src.fleet.simulator import DEFAULT_FLEET
from src.routers.baselines import FixedArm

ROOT = ensure_corpus()


class TestToolsAreReal(unittest.TestCase):

    def test_calculator_computes(self):
        self.assertAlmostEqual(calculate("(3 + 4) * 5"), 35.0)
        self.assertAlmostEqual(calculate("2 ** 10"), 1024.0)
        self.assertAlmostEqual(calculate("-7 + 2.5"), -4.5)
        self.assertAlmostEqual(calculate("17 % 5"), 2.0)

    def test_calculator_refuses_code_execution(self):
        """It walks an AST whitelist rather than calling eval(). That is the test.

        An agent tool handed model-emitted text is an arbitrary-code-execution surface
        if it uses eval(). The fact that this repo's 'model' is a random number
        generator is not a reason to write the pattern down.
        """
        for hostile in ("__import__('os').system('true')", "open('/etc/passwd').read()",
                        "[].__class__", "lambda: 1", "x + 1", "print(1)"):
            with self.assertRaises(ToolError, msg=hostile):
                calculate(hostile)

    def test_calculator_refuses_denial_of_service_and_zero_division(self):
        with self.assertRaises(ToolError):
            calculate("9 ** 9 ** 9")
        with self.assertRaises(ToolError):
            calculate("1 / 0")
        with self.assertRaises(ToolError):
            calculate("1 +")

    def test_file_tools_read_real_bytes(self):
        name = sorted(p.name for p in ROOT.glob("notes_*.txt"))[0]
        text = read_file(name, ROOT)
        self.assertGreater(len(text.splitlines()), 10)
        expected = sum(1 for line in text.splitlines() if "router" in line)
        self.assertEqual(search_text("router", name, ROOT), expected)

    def test_sandbox_escape_is_refused(self):
        with self.assertRaises(ToolError):
            read_file("../../../../etc/passwd", ROOT)
        with self.assertRaises(ToolError):
            read_file("does_not_exist.txt", ROOT)

    def test_dispatch_rejects_unknown_tools(self):
        with self.assertRaises(ToolError):
            dispatch(ToolCall("no_such_tool", {}), ROOT)

    def test_corpus_is_deterministic(self):
        a = ensure_corpus()
        names = sorted(p.name for p in a.glob("notes_*.txt"))
        self.assertGreaterEqual(len(names), 6)
        self.assertEqual(read_file(names[0], ROOT), read_file(names[0], a))


class TestEpisodicLoop(unittest.TestCase):

    @staticmethod
    def _features(rng):
        def fn(step, i):
            s = float(np.clip(step.difficulty + rng.normal(0, 0.12), 0, 1))
            x = np.zeros(N_FEATURES)
            x[0] = 1.0
            x[6] = s
            x[7] = s * s
            expected = np.array([a.base_seconds + a.seconds_per_token * 90
                                 for a in DEFAULT_FLEET])
            return x, expected
        return fn

    def test_a_perfect_arm_would_solve_every_episode(self):
        """The loop's plumbing must not lose episodes by itself.

        With a hypothetical arm that never mis-emits a call, every episode must succeed
        -- otherwise the measured end-to-end rate is contaminated by tool bugs rather
        than by model failures, and the compounding numbers mean nothing.
        """
        from dataclasses import replace
        perfect = tuple(replace(a, skill=50.0) for a in DEFAULT_FLEET)
        rng = np.random.default_rng(0)
        for _ in range(60):
            ep = make_episode(rng, 5, ROOT, 0.9, independent_steps=False)
            res = run_episode(ep, FixedArm(0), self._features(rng), perfect, rng, ROOT)
            self.assertTrue(res.succeeded, "the loop lost an episode with a perfect arm")

    def test_a_useless_arm_solves_nothing(self):
        from dataclasses import replace
        useless = tuple(replace(a, skill=-50.0) for a in DEFAULT_FLEET)
        rng = np.random.default_rng(1)
        wins = 0
        for _ in range(60):
            ep = make_episode(rng, 4, ROOT, 0.5, independent_steps=False)
            wins += run_episode(ep, FixedArm(2), self._features(rng), useless, rng,
                                ROOT).succeeded
        self.assertEqual(wins, 0)

    # === THE TEST THAT MATTERS ===
    # The compounding claim, and the control that makes it credible. With per-step
    # difficulties drawn independently, the steps really are independent and p^n is the
    # right prediction -- so the measured end-to-end rate must land on it. With a shared
    # per-episode difficulty the steps are positively dependent and Jensen's inequality
    # applied to the convex map q -> q^n forces the measured rate ABOVE p^n. Both signs
    # are asserted, because either one alone could be produced by a broken loop.
    def test_independent_steps_match_p_to_the_n_and_correlated_steps_exceed_it(self):
        def block(independent, n_steps, episodes=800, seed=7):
            rng = np.random.default_rng(seed)
            ok_steps = total_steps = wins = 0
            for _ in range(episodes):
                d = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
                ep = make_episode(rng, n_steps, ROOT, d, independent)
                res = run_episode(ep, FixedArm(0), self._features(rng), DEFAULT_FLEET,
                                  rng, ROOT)
                ok_steps += sum(res.step_ok)
                total_steps += len(res.step_ok)
                wins += res.succeeded
            p = ok_steps / total_steps
            return p, wins / episodes, p ** n_steps

        n = 6
        p_i, measured_i, predicted_i = block(True, n)
        # The control: independence holds, so the formula must be right. Tolerance is
        # 3 binomial standard errors of the measured rate at 800 episodes.
        se = float(np.sqrt(max(predicted_i, 1e-9) * (1 - predicted_i) / 800))
        self.assertLess(abs(measured_i - predicted_i), max(3 * se, 0.02),
                        f"independent steps: measured {measured_i:.4f} vs p^n "
                        f"{predicted_i:.4f} (p={p_i:.4f}); the loop is not measuring "
                        "what the formula predicts")

        p_c, measured_c, predicted_c = block(False, n)
        self.assertGreater(measured_c, predicted_c * 1.5,
                           f"correlated steps: measured {measured_c:.4f} is not clearly "
                           f"above p^n {predicted_c:.4f}; Jensen's inequality says it "
                           "must be")

    def test_gap_grows_with_the_horizon(self):
        """The correlated/independent gap must widen with n -- the convexity is in q^n."""
        rng_seed = 21

        def ratio(n_steps):
            rng = np.random.default_rng(rng_seed)
            ok = tot = wins = 0
            for _ in range(600):
                d = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
                ep = make_episode(rng, n_steps, ROOT, d, independent_steps=False)
                res = run_episode(ep, FixedArm(0), self._features(rng), DEFAULT_FLEET,
                                  rng, ROOT)
                ok += sum(res.step_ok)
                tot += len(res.step_ok)
                wins += res.succeeded
            p = ok / tot
            return (wins / 600) / (p ** n_steps)

        self.assertGreater(ratio(8), ratio(3),
                           "the correlated-vs-independent gap did not grow with the "
                           "number of steps")


if __name__ == "__main__":
    unittest.main()
