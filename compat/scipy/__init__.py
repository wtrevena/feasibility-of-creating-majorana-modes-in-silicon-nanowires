"""Minimal pure-numpy scipy shim (sandbox fallback; see compat/README.md).

Provides exactly the scipy.sparse API surface used by majorana_sim.py for
1D chain Hamiltonians, backed by an explicit block-tridiagonal (BT)
representation, plus a shift-invert Lanczos eigsh. Validated against dense
numpy.linalg.eigh by compat/test_shim.py. NOT a general scipy replacement.
"""
__version__ = "0.0.shim"
from . import sparse  # noqa: F401
