"""The translation layer -- the only part of this system that can be wrong.

Everything downstream of `translate` is proved optimal, so these are the tests that
bound the *system's* error rate rather than the search's.  They assert the parser's
stated grammar exactly (every request inside it must parse exactly right), assert that
the parser flags what it cannot represent, and assert that the language-model backend
fails loudly instead of silently falling back.
"""

import unittest

from src.domain import instance_from_spec, load_gazetteer, plan_cost, spec_from_instance
from src.explain import explain_plan
from src.search import heuristics as H
from src.search.astar import astar
from src.translate import (OllamaBackend, RuleBackend, TranslationError, canonical,
                           field_report, load_corpus, parse_clock)


class TestClock(unittest.TestCase):

    def test_time_expressions(self):
        for text, minutes in (("09:30", 570), ("9am", 540), ("3 pm", 900),
                              ("12:00", 720), ("noon", 720), ("midday", 720),
                              ("07:05", 425)):
            self.assertEqual(parse_clock(text), minutes, text)

    def test_unparsable_time_raises(self):
        with self.assertRaises(TranslationError):
            parse_clock("teatime")


class TestRuleBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = RuleBackend()
        cls.corpus = load_corpus()
        cls.plain = [r for r in cls.corpus["requests"] if r["difficulty"] == "plain"]
        cls.hard = [r for r in cls.corpus["requests"] if r["difficulty"] == "hard"]

    def test_corpus_is_large_enough_to_mean_something(self):
        self.assertGreaterEqual(len(self.corpus["requests"]), 25)
        self.assertGreaterEqual(len(self.hard), 8)

    # The grammar in translate.RuleBackend's docstring is a promise. Every request
    # inside it must parse exactly -- not approximately, not on average. If this drops
    # below 100% the promise is what changed, and the README's headline number with it.
    def test_every_request_inside_the_grammar_parses_exactly(self):
        misses = []
        for r in self.plain:
            res = self.backend.translate(r["text"])
            if not field_report(r["gold"], res.spec)["exact"]:
                misses.append((r["id"], res.spec))
        self.assertEqual(misses, [], f"{len(misses)} in-grammar request(s) mis-parsed")

    def test_hard_subset_is_actually_hard(self):
        # If the deliberately out-of-grammar requests all passed, the corpus would be
        # measuring nothing and the honest headline would be a different number.
        exact = sum(field_report(r["gold"], self.backend.translate(r["text"]).spec)["exact"]
                    for r in self.hard)
        self.assertLess(exact, len(self.hard))

    def test_every_failure_is_self_flagged(self):
        # The claim that makes the error rate deployable: when the parser is wrong, it
        # said so. A wrong-and-confident parse is the expensive kind, and this test is
        # what stops one appearing unnoticed.
        for r in self.corpus["requests"]:
            res = self.backend.translate(r["text"])
            if not field_report(r["gold"], res.spec)["exact"]:
                self.assertTrue(res.warnings or res.error,
                                f"{r['id']} parsed wrongly with no warning and no error")

    def test_missing_capacity_is_an_error_not_a_default(self):
        res = self.backend.translate("One crate to Baker Street before 11:00.")
        self.assertFalse(res.ok)
        self.assertIn("capacity", res.error)

    def test_unknown_place_is_an_error_not_a_guess(self):
        res = self.backend.translate("Two crates to the docks. The van holds four crates.")
        self.assertFalse(res.ok)
        self.assertIn("gazetteer", res.error)

    def test_repeated_mention_is_one_stop(self):
        res = self.backend.translate(
            "Two crates to Baker Street. Baker Street must be done before 11:00. "
            "Capacity four.")
        self.assertEqual(len(res.spec["stops"]), 1)
        self.assertEqual(res.spec["stops"][0]["demand"], 2)
        self.assertEqual(res.spec["stops"][0]["latest"], 660)

    def test_canonical_form_is_order_insensitive(self):
        a = self.backend.translate(
            "One crate to Baker Street and one to Elm Avenue. Capacity two.").spec
        b = self.backend.translate(
            "One crate to Elm Avenue and one to Baker Street. Capacity two.").spec
        self.assertEqual(canonical(a), canonical(b))


class TestOllamaBackend(unittest.TestCase):

    def test_stub_raises_and_names_what_it_needs(self):
        with self.assertRaises(NotImplementedError) as ctx:
            OllamaBackend().translate("two crates to Baker Street, van holds four")
        msg = str(ctx.exception)
        for token in ("Ollama", "qwen", "STATUS.md"):
            self.assertIn(token, msg)

    def test_stub_does_not_silently_fall_back_to_the_rule_backend(self):
        # A silent fallback would make every number attributed to the LLM path a
        # measurement of the rule parser instead.
        self.assertNotIsInstance(OllamaBackend(), RuleBackend)


class TestEndToEnd(unittest.TestCase):

    def test_request_to_optimal_validated_plan_to_english(self):
        backend = RuleBackend()
        gaz = load_gazetteer()
        corpus = load_corpus()
        solved = 0
        for r in corpus["requests"]:
            res = backend.translate(r["text"])
            if not res.ok:
                continue
            inst = instance_from_spec(res.spec, gaz, name=r["id"])
            out = astar(inst, H.make("h2"))
            if out.cost is None:
                continue                       # proved infeasible; also a valid answer
            self.assertEqual(plan_cost(inst, out.plan), out.cost,
                             f"{r['id']}: A*'s plan does not re-validate")
            text = explain_plan(inst, out.plan, out.cost, "h2-mst",
                                out.stats.expansions)
            for stop in res.spec["stops"]:
                self.assertIn(stop["name"], text)
            self.assertIn("proved", text)
            solved += 1
        self.assertGreaterEqual(solved, 20)

    def test_spec_round_trips_through_an_instance(self):
        backend = RuleBackend()
        gaz = load_gazetteer()
        spec = backend.translate(
            "The van holds four crates. Two crates to Baker Street before 11:00 and "
            "one to Elm Avenue after 09:00.").spec
        inst = instance_from_spec(spec, gaz)
        self.assertEqual(canonical(spec_from_instance(inst)), canonical(spec))

    def test_gazetteer_gap_is_reported_not_guessed(self):
        with self.assertRaises(KeyError):
            instance_from_spec({"capacity": 2, "stops": [
                {"name": "atlantis", "demand": 1, "earliest": 480, "latest": 1080}]},
                load_gazetteer())


if __name__ == "__main__":
    unittest.main()
