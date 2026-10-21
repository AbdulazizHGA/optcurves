#!/usr/bin/env bash
# Regenerate every result, certificate, counterexample, Lean file and figure, then run the tests.
set -euo pipefail
cd "$(dirname "$0")"
# Interpreter: $PYTHON if set; else the first of these with Python >= 3.11 (pyproject.toml). `python` comes last
# because it can be Python 2 on old systems; the version check itself also runs under Python 2.
CANDIDATES="python3 python3.14 python3.13 python3.12 python3.11 python"
version_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; }
deps_ok() { "$1" -c 'import PEPit, cvxpy, clarabel, scs, sympy, numpy, scipy, matplotlib, pytest' >/dev/null 2>&1; }
if [ -n "${PYTHON:-}" ]; then
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "reproduce.sh: interpreter not found: $PYTHON" >&2
        exit 1
    fi
    if ! version_ok "$PYTHON"; then
        echo "reproduce.sh: $PYTHON is $("$PYTHON" -V 2>&1); Python >= 3.11 is needed" >&2
        exit 1
    fi
    PY="$PYTHON"
else
    # Prefer the first candidate that also has the packages; else the first with a right version (for the message).
    PY="" FIRST_OK=""
    for candidate in $CANDIDATES; do
        if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
            [ -z "$FIRST_OK" ] && FIRST_OK="$candidate"
            if deps_ok "$candidate"; then
                PY="$candidate"
                break
            fi
        fi
    done
    PY="${PY:-$FIRST_OK}"
    if [ -z "$PY" ]; then
        echo "reproduce.sh: no Python >= 3.11 found (tried $CANDIDATES); set PYTHON=/path/to/python" >&2
        exit 1
    fi
fi
echo "reproduce.sh: using $PY ($("$PY" -V 2>&1))"
# A right version can still lack the packages: say so before the first step fails on an import.
if ! deps_ok "$PY"; then
    echo "reproduce.sh: $PY lacks the dependencies; install them with" >&2
    echo "  $PY -m pip install -r requirements-lock.txt && $PY -m pip install -e . --no-deps" >&2
    echo "or set PYTHON to an interpreter that has them" >&2
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
