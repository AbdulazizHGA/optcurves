"""Two side experiments behind supporting sentences of the README. Writes results/side.json.

    python -m optcurves.side

* worst_case_pep: the PEP maximum of v_1 over ||x_0 - x*|| <= 1 (pep.evaluate) against the closed form
  h^2 (-q) / (2 D) of worstcase.py, at 20 points: kappa = 1.1, 1.5, 2, 10, 100 and h = h_1(kappa) + 0.01,
  (h_1(kappa) + 2) / 2, 1.95, 1.99. The closed form is proved, so this checks the PEP pipeline.
* one_dimensional_search: the max-slack local search of counterexample.py (200 random starts, seed 0) in R^1, for
  two consecutive violations at h = 1.9 and three at h = 1.99. A margin above 1e-9 (the feasibility tolerance of
  the search) would be a strictly feasible example; for comparison, the same search in R^2 gave the margins stored
  with the two-dimensional counterexamples. A search that finds nothing is not a proof that R^1 is impossible.
"""
import json

from . import counterexample, figures, pep, study

KAPPAS = (1.1, 1.5, 2, 10, 100)
SEARCHES = (("1.9", 2, "n2_h19_10_d2.json"), ("1.99", 3, "n3_h199_100_d2.json"))
STARTS = 200


def worst_case_pep():
    rows = []
    for kappa in KAPPAS:
        mu = 1 / kappa
        h1 = figures.h1_closed_form(kappa)
        for h in (h1 + 0.01, (h1 + 2) / 2, 1.95, 1.99):
            q = 2 * mu * h ** 2 - 4 * (1 + mu) * h + mu + 7
            exact = -h ** 2 * q / (2 * (2 + h ** 2 * (1 - mu)))
            pep.reset_stats()
            value, reliable = pep.evaluate(h, 1, mu=mu)
            rows.append({"kappa": kappa, "h": h, "pep": value, "closed_form": exact,
                         "relative_error": abs(value - exact) / exact, "reliable": reliable,
                         "fallbacks": pep.STATS["fallbacks"]})
    return {"points": rows, "max_relative_error": max(r["relative_error"] for r in rows)}


def one_dimensional_search():
    rows = []
    for h, n, two_dim in SEARCHES:
        found = counterexample._nlp(float(h), n, 1, trials=STARTS, seed=0)
        stored = json.loads((counterexample.ROOT / "counterexamples" / two_dim).read_text())
        rows.append({"h": h, "n": n, "starts": STARTS, "seed": 0,
                     "best_margin_R1": None if found is None else found[2],
                     "margin_R2": stored["numeric_slack"], "R2_example": two_dim})
    return rows


def main():
    out = {"meta": study._meta(), "worst_case_pep": worst_case_pep(), "one_dimensional_search": one_dimensional_search()}
    print(f"worst case: PEP against the closed form at {len(out['worst_case_pep']['points'])} points, largest relative "
          f"error {out['worst_case_pep']['max_relative_error']:.2e}", flush=True)
    for r in out["one_dimensional_search"]:
        print(f"R^1 search, h = {r['h']}, n = {r['n']}: best margin {r['best_margin_R1']} (R^2: {r['margin_R2']:.2e})",
              flush=True)
    study._write("side.json", out)


if __name__ == "__main__":
    main()
