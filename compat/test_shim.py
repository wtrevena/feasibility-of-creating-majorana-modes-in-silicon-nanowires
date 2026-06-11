"""Validate the compat scipy shim against dense numpy.linalg.eigh.

Builds small wires through majorana_sim (with the shim active), converts the
BT matrix to dense in the ORIGINAL ordering, and compares the k lowest-|E|
eigenvalues from the shim's shift-invert Lanczos with dense eigh, plus
Hermiticity and eigenvector residual checks. Run: python compat/test_shim.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
try:
    import scipy  # noqa: F401
    print("note: sys.path[0] is compat/, so this import found the shim")
except ImportError:
    pass
sys.path.insert(0, HERE)   # shim first
sys.path.insert(0, ROOT)

from majorana_sim import (UEV, build_wire, build_wire_two_valley_iv,
                          solve_lowest)


def check(tag, H, k=6):
    E, V = solve_lowest(H, k=k)
    M = H.todense()
    assert np.max(np.abs(M - M.conj().T)) < 1e-30, "not Hermitian"
    Ed = np.linalg.eigvalsh(M)
    Ed = Ed[np.argsort(np.abs(Ed))][:k]
    a = np.sort(np.abs(E)) / UEV
    b = np.sort(np.abs(Ed)) / UEV
    scale = max(b[-1], 1e-3)
    err = np.max(np.abs(a - b)) / scale
    # eigenvector residual in original ordering
    r = np.max([np.linalg.norm(M @ V[:, i] - E[i] * V[:, i]) /
                max(np.abs(E[i]), 1e-28 * np.linalg.norm(M, 2))
                for i in range(k)])
    print(f"{tag:38s} |E|/ueV shim={np.round(a,4)} dense={np.round(b,4)} "
          f"rel_err={err:.2e}")
    assert err < 1e-6, f"{tag}: eigenvalue mismatch {err:.2e}"
    return err


def main():
    rng = np.random.default_rng(3)
    # 1) plain BdG wire, topological point, with disorder
    H = build_wire(80, 5e-9, 20.0, 1.2, 40.0, 0.05, 0.19, 2.0,
                   disorder_ueV=60.0, rng=rng)
    check("build_wire topo + disorder", H)
    # 2) two-valley inter-valley pairing, clean wedge
    H = build_wire_two_valley_iv(70, 5e-9, 35.0, 1.5, 50.0, 0.05, 0.19, 2.0,
                                 vo_profile_ueV=75.0 * np.ones(70, complex))
    check("two-valley iv clean", H)
    # 3) two-valley with a 0.85*pi step (the fig9b observable)
    vo = 75.0 * np.exp(1j * np.where(np.arange(70) < 35, 0.0, 0.85 * np.pi))
    H = build_wire_two_valley_iv(70, 5e-9, 35.0, 1.5, 50.0, 0.05, 0.19, 2.0,
                                 vo_profile_ueV=vo)
    check("two-valley iv single step", H)
    # 4) dresselhaus phase-locked SOC path (exercises LilBT)
    H = build_wire_two_valley_iv(60, 5e-9, 35.0, 1.5, 50.0, 0.05, 0.19, 2.0,
                                 vo_profile_ueV=vo[:60],
                                 soc_mode="dresselhaus")
    check("two-valley dresselhaus", H)
    print("shim self-test: ALL PASS")


if __name__ == "__main__":
    main()
