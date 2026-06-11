#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# reproduce.sh -- one-command reproduction of every figure, number, and check.
#
# Runs the test suites, then every analysis script in dependency order, then
# tools/manifest.py (which fails if any headline number quoted in the paper
# no longer matches the ledger output/data/key_numbers.json).
#
# NOTE on runtime: all heavy scripts checkpoint per-section to
# output/data/*.json (keyed by parameter signatures) and resume, so a full
# pass over the committed checkpoints is mostly cache reads (minutes).
# Delete a script's checkpoint JSON in output/data/ to force genuine
# recomputation of that section (hours for the largest scans).
#
# Environment: python with numpy/scipy/matplotlib (see requirements.txt);
# without scipy the validated compat/ shim is picked up automatically.
# Override the interpreter with:  PYTHON=/path/to/python ./reproduce.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "venv/bin/python" ]; then PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY="python3"
else PY="python"
fi
echo "Using interpreter: $PY"

run() { echo; echo "== $* =="; "$PY" "$@"; }

# 1. test suites (every exact claim is pinned here)
run tests.py
run compat/test_shim.py

# 2. analysis scripts, dependency order
run run_analysis.py --fig all      # figs 1-11 + key_numbers tags fig1..fig11
run transport.py                   # fig12, transport_numbers.json
run transport_valley.py            # valley/Anderson transport (transport_numbers.json: valley)
run qp_poisoning.py                # QP poisoning bounds (key_numbers: fig8.qp_poisoning)
run convergence.py                 # fig13 + convergence tables (key_numbers: convergence)
run realism.py                     # dynamic self-energy / Dynes / disorder controls
run orbital.py                     # Peierls orbital control
run pairing_mix.py                 # inter/intra-valley pairing mix control
run morphology.py                  # miscut / terrace / bunching ensembles
run kp6_holes.py                   # six-band k.p fin model ([100])
run kp6_110.py                     # [110] channel rotation
run poisson2d.py                   # tri-gate Poisson + SCF harness
run kp6_sc.py                      # self-consistent six-band + Poisson
run platform110.py                 # platform operating points with [110] parameters

# 3. any round-6 add-on scripts present
for f in r6_*.py; do
  [ -e "$f" ] || continue
  run "$f"
done

# 4. integrity manifest (fails on any paper-number / ledger mismatch)
run tools/manifest.py

echo
echo "ALL REPRODUCED"
