#!/usr/bin/env bash
# Regenerate every result, certificate, counterexample, Lean file and figure, then run the tests.
set -euo pipefail
cd "$(dirname "$0")"
# Interpreter: $PYTHON if set, else python3, else python (on old systems `python` can be Python 2).
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "reproduce.sh: neither python3 nor python is on PATH; set PYTHON=/path/to/python" >&2
    exit 1
fi
# Python >= 3.11 (pyproject.toml); the check itself also runs under Python 2.
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "reproduce.sh: interpreter not found: $PY" >&2
    exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "reproduce.sh: $PY is $("$PY" -V 2>&1); Python >= 3.11 is needed (set PYTHON=/path/to/python)" >&2
    exit 1
fi
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
