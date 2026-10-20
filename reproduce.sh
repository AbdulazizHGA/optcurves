#!/usr/bin/env bash
# Regenerate every result, certificate, counterexample, Lean file and figure, then run the tests.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONPATH=src
"$PY" -m optcurves.study consecutive     # results/consecutive.json
"$PY" -m optcurves.study strong          # results/strong.json
"$PY" -m optcurves.certificate           # certificates/*.json
"$PY" -m optcurves.counterexample        # counterexamples/*.json
"$PY" -m optcurves.side                  # results/side.json (PEP vs worst case at 20 points; search in R^1)
"$PY" -m optcurves.theorem               # symbolic proof of h_1(kappa) (checks only, writes nothing)
"$PY" -m optcurves.worstcase             # symbolic proof of the worst case above h_1 (checks only, writes nothing)
"$PY" -m optcurves.lean_gen              # lean/Optcurves/Certificates.lean
"$PY" -m optcurves.figures               # figures/*.png
"$PY" -m pytest -q
