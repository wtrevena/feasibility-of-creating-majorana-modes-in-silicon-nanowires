"""
platform110.py — hole-platform operating points re-evaluated with the
six-band [110]-channel parameters (kp6_110, the experimentally relevant
orientation): m* ~ 0.19-0.21, alpha = 0.052 (Ez=10 MV/m) / 0.073 (30 MV/m)
eV*A, wire-axis g_x' = 1.55 / 1.66 (nearly Ez-independent), g_z = 2.6 / 3.4.

Key change vs the old empirical/4-band treatment: the wire-axis g is now
usable for IN-PLANE-field designs (film parents are in-plane-locked), and
the harmful alpha-g_x covariation is largely absent in [110].

Sections (checkpoint output/data/platform110.json, ledger tag platform110):
  A: static optimizer (_best_gap_hole, 10x10x20 grid) bare + renormalized,
     three parents x two [110] parameter points (Ez=10, 30 MV/m), using the
     WIRE-AXIS g (in-plane along wire — the conservative film-parent case).
  B: same points with the tilted/out-of-plane g_z (orientation upper bound,
     thick-SiB-style isotropic-Bc parent only).
  C: dynamic self-energy (realism.dynamic_gap_ueV) Gamma-scan at the
     section-A operating points — the corrected central estimates.
"""
import argparse, json, os, sys, time
import numpy as np
try:
    import scipy.sparse  # noqa
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compat"))
from run_analysis import (DATA, save_numbers, _best_gap_hole, _parent_gap_ueV,
                          SIB_DELTA0, SIB_BC2_MEAS)
import run_analysis as ra
from realism import dynamic_gap_ueV

CKPT = os.path.join(DATA, "platform110.json")
# [110] six-band parameter points (kp6_110 section B)
PTS = {
    "Ez10": dict(m=0.194, al=0.052, gx=1.549, gz=2.632),
    "Ez30": dict(m=0.204, al=0.073, gx=1.657, gz=3.441),
}
PARENTS = ("SiB_meas", "SiB_pauli", "Al")

def _load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}

def _save(d):
    json.dump(d, open(CKPT, "w"), indent=2)

def sec_A(res):
    out = {}
    for tag, p in PTS.items():
        for parent in PARENTS:
            for ren in (False, True):
                g, arg = _best_gap_hole(p["al"], p["gx"], parent,
                                        m_rel=p["m"], nB=10, nD=10, nmu=20,
                                        renormalize=ren)
                key = f"{tag}_{parent}" + ("_renorm" if ren else "")
                out[key] = dict(gap=round(g, 2), B=round(float(arg[0]), 3),
                                Dind=round(float(arg[1]), 1))
                print("A", key, out[key], flush=True)
    res["A_inplane_gx"] = out

def sec_B(res):
    out = {}
    for tag, p in PTS.items():
        for ren in (False, True):
            g, arg = _best_gap_hole(p["al"], p["gz"], "SiB_meas",
                                    m_rel=p["m"], nB=10, nD=10, nmu=20,
                                    renormalize=ren)
            key = f"{tag}_SiB_meas_gz" + ("_renorm" if ren else "")
            out[key] = dict(gap=round(g, 2), B=round(float(arg[0]), 3))
            print("B", key, out[key], flush=True)
    res["B_tilted_gz"] = out

def sec_C(res):
    out = {}
    pp = dict(SiB_meas=lambda B: SIB_DELTA0 * max(1 - (B / SIB_BC2_MEAS)**2, 0),
              SiB_pauli=lambda B: max(SIB_DELTA0 - 57.883818060 * B, 0.0),
              Al=lambda B: 200.0 * max(1 - (B / 2.0)**2, 0.0))
    for tag, p in PTS.items():
        for parent in PARENTS:
            # operating B from section A (bare)
            Bop = res["A_inplane_gx"][f"{tag}_{parent}"]["B"]
            Dp = pp[parent](Bop)
            best = (0.0, None)
            row = {}
            for G in (3, 6, 10, 15, 20, 30, 45, 70, 100):
                gap = dynamic_gap_ueV(0.0, Bop, Dp, G, p["al"], p["m"],
                                      p["gx"])
                row[str(G)] = round(gap, 2)
                if gap > best[0]:
                    best = (gap, G)
            out[f"{tag}_{parent}"] = dict(Bop=Bop, Dp=round(Dp, 1),
                                          gap_vs_Gamma=row,
                                          best=round(best[0], 2),
                                          Gamma_star=best[1])
            print("C", tag, parent, "best", best, flush=True)
    res["C_dynamic"] = out

SECS = dict(A=sec_A, B=sec_B, C=sec_C)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sec", default="all")
    a = ap.parse_args()
    todo = list(SECS) if a.sec == "all" else a.sec.split(",")
    res = _load()
    for s_ in todo:
        if any(k.startswith(s_ + "_") for k in res):
            print("cached", s_, flush=True); continue
        SECS[s_](res); _save(res)
    if all(any(k.startswith(s_ + "_") for k in res) for s_ in SECS):
        save_numbers("platform110", res)
