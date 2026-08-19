"""The committed CSVs, and the column contract analysis/plot_results.py depends on.

matplotlib is not installed here, so plot_results.py cannot be run and a mistyped
column name in it would be invisible until someone with matplotlib tried. These tests
close that gap without needing matplotlib: they assert that every column the plotting
code indexes exists in the committed CSV, and that the numbers in those CSVs are
internally consistent.
"""

import csv
import pathlib
import unittest

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

#: file -> columns that analysis/plot_results.py reads by name.
PLOT_COLUMNS = {
    "convergence.csv": {"mode", "prior", "grid", "sampler", "h_max", "error",
                        "fitted_slope"},
    "nfe_quality.csv": {"sampler", "nfe", "w1", "traj_rmse"},
    "sde_vs_ode.csv": {"section", "sampler", "prob_level", "conditioning_time",
                       "mode_entropy_bits"},
    "stability.csv": {"section", "prior_variance", "err_euler", "err_heun",
                      "err_dpm1", "err_dpm2"},
}


def load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise AssertionError(f"{path} is missing -- run `make results`.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


class TestCommittedResults(unittest.TestCase):
    def test_every_plotted_column_exists(self):
        for name, needed in PLOT_COLUMNS.items():
            rows = load(name)
            self.assertTrue(rows, f"{name} is empty")
            missing = needed - set(rows[0])
            self.assertFalse(missing, f"{name} is missing {sorted(missing)}")

    def test_the_plot_filters_select_something(self):
        # Each figure filters its CSV down to a subset; an empty subset would draw a
        # blank axis rather than raise, so the filters are checked here instead.
        conv = load("convergence.csv")
        self.assertTrue([r for r in conv if r["mode"] == "ode_trajectory"
                         and r["prior"] == "canonical" and r["grid"] == "uniform_logsnr"])
        self.assertTrue([r for r in load("nfe_quality.csv") if r["sampler"] == "exact_map"])
        self.assertTrue([r for r in load("sde_vs_ode.csv")
                         if r["section"] == "diversity"
                         and r["sampler"] == "euler_maruyama"
                         and r["prob_level"] == "0.5"])
        self.assertTrue([r for r in load("stability.csv") if r["section"] == "sharpness"])

    def test_the_headline_slopes_are_the_ones_the_readme_quotes(self):
        # If someone changes the experiment without updating the README, this fails.
        want = {"euler_ode": 1.0, "heun": 2.0, "exponential_1": 1.0, "exponential_2": 2.0}
        rows = [r for r in load("convergence.csv")
                if r["mode"] == "ode_trajectory" and r["prior"] == "canonical"
                and r["grid"] == "uniform_logsnr"]
        for sampler, order in want.items():
            sub = [r for r in rows if r["sampler"] == sampler]
            self.assertTrue(sub, sampler)
            self.assertAlmostEqual(float(sub[0]["fitted_slope"]), order, delta=0.1,
                                   msg=sampler)
            self.assertLess(float(sub[0]["fit_residual_decades"]), 0.02, sampler)

    def test_euler_maruyama_strong_order_row_is_one(self):
        rows = [r for r in load("convergence.csv") if r["mode"] == "sde_strong"]
        self.assertTrue(rows)
        self.assertAlmostEqual(float(rows[0]["fitted_slope"]), 1.0, delta=0.15)

    def test_the_exponential_amplification_never_leaves_its_interval(self):
        rows = [r for r in load("stability.csv") if r["section"] == "amplification"]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["exponential_within_exact"], "1", f"N={r['n_steps']}")

    def test_no_result_row_is_blank_where_it_claims_a_number(self):
        for name in PLOT_COLUMNS:
            for r in load(name):
                for key, value in r.items():
                    if value == "":
                        continue
                    self.assertNotIn(value.lower(), {"nan", "inf", "-inf"},
                                     f"{name}: {key} = {value}")


if __name__ == "__main__":
    unittest.main()
