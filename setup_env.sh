#!/usr/bin/env bash
# One-shot environment setup (run on a machine with PyPI access).
# Windows (PowerShell):  py -m venv venv; .\venv\Scripts\Activate.ps1;
#                        pip install -r requirements.txt
set -e
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python tests.py                      # must end ALL TESTS PASS
python compat/test_shim.py           # shim cross-validation (optional)
echo "Environment ready. Regenerate everything with:"
echo "  python run_analysis.py --fig all && python transport.py && \\"
echo "  python transport_valley.py && python qp_poisoning.py && \\"
echo "  python convergence.py"
