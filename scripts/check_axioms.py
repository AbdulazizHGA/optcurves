"""Fail unless every `#print axioms` line in a Lean log lists exactly the three standard axioms.

    python3 scripts/check_axioms.py lean/axioms.log lean/Optcurves/Certificates.lean
"""
import re
import sys

STANDARD = "[propext, Classical.choice, Quot.sound]"

log = open(sys.argv[1]).read()
expected = len(re.findall(r"^#print axioms ", open(sys.argv[2]).read(), flags=re.M))
found = re.findall(r"'([\w.]+)' depends on axioms: (\[[^\]]*\])", log.replace("\n", " "))
bad = [(name, ax) for name, ax in found if ax != STANDARD]
for name, ax in found:
    print(f"{name}: {ax}")
if "error" in log.lower() or "sorry" in log.lower():
    sys.exit("Lean log contains an error or sorry")
if len(found) != expected or bad:
    sys.exit(f"expected {expected} theorems depending only on {STANDARD}; found {len(found)} '#print axioms' "
             f"lines, these with other axioms: {bad}")
print(f"OK: {expected} theorems depend only on {STANDARD}")
