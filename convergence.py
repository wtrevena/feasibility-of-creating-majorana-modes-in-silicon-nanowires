"""
convergence.py — numerical convergence and sensitivity studies (reviewer demand 5).

Sections (each checkpointed to output/data/convergence.json, resumable):
  A  dx     : lattice-spacing convergence of the clean two-valley wedge gap and the
              single-step (0.85*pi, pi) bound states at fixed L = 2.5 um.
  B  L      : wire-length convergence of the same observables + the 50-nm Poisson
              same-sign step-ensemble median (14 seeds), L = 1.25 / 2.5 / 5 um.
  C  seeds  : disorder-seed convergence of the 50-nm ensemble median at baseline
              geometry: running median vs n and a 56-seed bootstrap 95% CI.
  D  mu     : chemical-potential sensitivity of clean gap and single-step state.
  E  ev     : valley-splitting sensitivity (VO amplitude lam, E_v = 2*lam).
  F  nk     : k-grid convergence of the bulk-gap formula at the three headline
              hole operating points.
  G  grid   : optimizer-grid convergence of _best_gap_hole at the empirical-center
              hole point for both Si:B parents (bare + renormalized).
  H  parent : parent-gap-model sensitivity (Delta0 +/-20%, Bc2 0.3/0.4 T) of the
              measured-field Si:B headline number.

Usage: python convergence.py [--sec A,B,...|all] ; figure: fig13_convergence.png
(If scipy/matplotlib are unavailable the compat/ BT shim is used; fig13 is
then deferred and regenerated later from convergence.json.)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import scipy.sparse.linalg  # noqa: F401  (the /tmp stub lacks .sparse)
    import matplotlib.pyplot    # noqa: F401
except Exception:               # sandbox without scipy: use the BT shim
    sys.path.insert(0, os.path.join(_HERE, "compat"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MPL_STUB = getattr(matplotlib, "_IS_STUB", False)

from majorana_sim import (
    UEV, bulk_gap_ueV, build_wire_two_valley_iv, solve_lowest,
)
from run_analysis import (
    M_SI, G_SI, DELTA, DX, ALPHA_DEMO, OUT, DATA, save_numbers,
    _make_vo_profile, _best_gap_hole, _parent_gap_ueV, M_HOLE,
)

CKPT = os.path.join(DATA, "convergence.json")

# fig9 baseline (electron valley sector)
B0, MU0, LAM0 = 1.5, 35.0, 75.0
N0 = 500                      # L = 2.5 um at DX = 5 nm
# fig8 empirical-center hole point
AL_H, G_H = 0.06, 2.2


def _load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def _save(d):
    with open(CKPT, "w") as f:
        json.dump(d, f, indent=2)


def solve_iv(N, dx, vo, mu=MU0, B=B0, Delta=DELTA, al=ALPHA_DEMO):
    H = build_wire_two_valley_iv(N, dx, mu, B, Delta, al, M_SI, G_SI,
                                 vo_profile_ueV=vo)
    E, _ = solve_lowest(H, k=6)
    Ea = np.sort(np.abs(E)) / UEV
    return float(Ea[0]), float(Ea[2])


def vo_clean(N, lam=LAM0):
    return lam * np.ones(N, complex)


def vo_step(N, dphi, lam=LAM0):
    half = N // 2
    return lam * np.exp(1j * np.where(np.arange(N) < half, 0.0, dphi))


def sec_A(res):
    out = {}
    for dx_nm in (10.0, 5.0, 2.5):
        dx = dx_nm * 1e-9
        N = int(round(2.5e-6 / dx))
        row = dict(clean=solve_iv(N, dx, vo_clean(N))[1],
                   step085=solve_iv(N, dx, vo_step(N, 0.85 * np.pi))[1],
                   step100_E0=solve_iv(N, dx, vo_step(N, np.pi))[0])
        out[f"dx={dx_nm:g}nm"] = {k: round(v, 3) for k, v in row.items()}
        print("A", dx_nm, row, flush=True)
    res["A_dx"] = out


def sec_B(res):
    out = {}
    for L_um in (1.25, 2.5, 5.0):
        N = int(round(L_um * 1e-6 / DX))
        meds = []
        for s in range(14):
            rng = np.random.default_rng([71, 1, 50, s])   # fig9 seed family
            vo = _make_vo_profile(N, DX, 50e-9, "poisson", "fixed", rng)
            meds.append(solve_iv(N, DX, vo)[1])
        out[f"L={L_um}um"] = dict(
            clean=round(solve_iv(N, DX, vo_clean(N))[1], 3),
            step085=round(solve_iv(N, DX, vo_step(N, 0.85 * np.pi))[1], 3),
            ens50_median=round(float(np.median(meds)), 3),
            ens50_iqr=[round(float(np.percentile(meds, q)), 3)
                       for q in (25, 75)])
        print("B", L_um, out[f"L={L_um}um"], flush=True)
    res["B_L"] = out


def sec_C(res):
    vals = []
    for s in range(56):
        rng = np.random.default_rng([71, 1, 50, s])
        vo = _make_vo_profile(N0, DX, 50e-9, "poisson", "fixed", rng)
        vals.append(solve_iv(N0, DX, vo)[1])
        if (s + 1) % 14 == 0:
            print("C", s + 1, "median",
                  round(float(np.median(vals)), 3), flush=True)
    vals = np.array(vals)
    running = {str(n): round(float(np.median(vals[:n])), 3)
               for n in (7, 14, 28, 56)}
    rng = np.random.default_rng(7)
    boot = [float(np.median(rng.choice(vals, len(vals)))) for _ in range(2000)]
    res["C_seeds"] = dict(
        running_median=running,
        median_56=round(float(np.median(vals)), 3),
        bootstrap95=[round(float(np.percentile(boot, q)), 3)
                     for q in (2.5, 97.5)],
        values=[round(float(v), 3) for v in vals])


def sec_D(res):
    out = {}
    for mu in (15.0, 25.0, 35.0, 45.0, 60.0):
        out[f"mu={mu:g}"] = dict(
            clean=round(solve_iv(N0, DX, vo_clean(N0), mu=mu)[1], 3),
            step085=round(solve_iv(N0, DX, vo_step(N0, 0.85 * np.pi),
                                   mu=mu)[1], 3))
        print("D", mu, out[f"mu={mu:g}"], flush=True)
    res["D_mu"] = out


def sec_E(res):
    out = {}
    for lam in (50.0, 75.0, 112.5, 150.0):
        out[f"Ev={2*lam:g}ueV"] = dict(
            clean=round(solve_iv(N0, DX, vo_clean(N0, lam))[1], 3),
            step085=round(solve_iv(N0, DX, vo_step(N0, 0.85 * np.pi,
                                                   lam))[1], 3))
        print("E", lam, out[f"Ev={2*lam:g}ueV"], flush=True)
    res["E_ev"] = out


# headline hole operating points (mu, B, Dind) from key_numbers fig8:
# center_SiB_measured_args = [0.33, 11.3, 0.0]; center_a006_g22 = B=1.0, D=33.0
HOLE_PTS = dict(
    SiB_meas=dict(mu=0.0, B=0.33, Dind=11.3, al=AL_H, g=G_H),
    SiB_pauli=dict(mu=0.0, B=1.0, Dind=33.0, al=AL_H, g=G_H),
)


def sec_F(res):
    out = {}
    pts = dict(HOLE_PTS)
    _, argAl = _best_gap_hole(AL_H, G_H, "Al")          # deterministic
    pts["Al"] = dict(mu=float(argAl[2]), B=float(argAl[0]),
                     Dind=float(argAl[1]), al=AL_H, g=G_H)
    for tag, p in pts.items():
        out[tag] = {str(nk): round(bulk_gap_ueV(p["mu"], p["B"], p["Dind"],
                                                p["al"], M_HOLE, p["g"],
                                                nk=nk), 4)
                    for nk in (501, 1001, 2501, 6001, 12001)}
        print("F", tag, out[tag], flush=True)
    res["F_nk"] = out


def sec_G(res):
    out = {}
    for parent in ("SiB_meas", "SiB_pauli"):
        for ren in (False, True):
            key = parent + ("_renorm" if ren else "")
            out[key] = {}
            for nB, nD, nmu in ((5, 5, 10), (7, 7, 14), (10, 10, 20),
                                (14, 14, 28)):
                g, arg = _best_gap_hole(AL_H, G_H, parent, nB=nB, nD=nD,
                                        nmu=nmu, renormalize=ren)
                out[key][f"{nB}x{nD}x{nmu}"] = dict(
                    gap=round(g, 3),
                    B=round(float(arg[0]), 3), Dind=round(float(arg[1]), 2),
                    mu=round(float(arg[2]), 2))
            print("G", key, out[key], flush=True)
    res["G_grid"] = out


def sec_H(res):
    import run_analysis as ra
    out = {}
    base_D0, base_Bc2 = ra.SIB_DELTA0, ra.SIB_BC2_MEAS
    for fD in (0.8, 1.0, 1.2):
        for Bc2 in (0.3, 0.4):
            ra.SIB_DELTA0, ra.SIB_BC2_MEAS = base_D0 * fD, Bc2
            g, arg = _best_gap_hole(AL_H, G_H, "SiB_meas")
            gr, _ = _best_gap_hole(AL_H, G_H, "SiB_meas", renormalize=True)
            out[f"D0x{fD:g}_Bc2={Bc2:g}T"] = dict(
                gap=round(g, 2), gap_renorm=round(gr, 2),
                B=round(float(arg[0]), 3))
            print("H", fD, Bc2, out[f"D0x{fD:g}_Bc2={Bc2:g}T"], flush=True)
    ra.SIB_DELTA0, ra.SIB_BC2_MEAS = base_D0, base_Bc2
    res["H_parent"] = out


def make_figure(res):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    a = ax[0, 0]
    dxs = [10, 5, 2.5]
    A = res["A_dx"]
    a.plot(dxs, [A[f"dx={d:g}nm"]["clean"] for d in dxs], "o-",
           label="clean wedge gap")
    a.plot(dxs, [A[f"dx={d:g}nm"]["step085"] for d in dxs], "s-",
           label=r"single step $\Delta\varphi=0.85\pi$")
    a.axvline(5, color="gray", ls=":", lw=1, label="production dx")
    a.set_xlabel("dx (nm)")
    a.set_ylabel(r"E$_2$ ($\mu$eV)")
    a.set_title("(a) lattice-spacing convergence (L = 2.5 µm)")
    a.invert_xaxis(); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    Ls = [1.25, 2.5, 5.0]
    Bv = res["B_L"]
    a.plot(Ls, [Bv[f"L={L}um"]["clean"] for L in Ls], "o-", label="clean")
    a.plot(Ls, [Bv[f"L={L}um"]["step085"] for L in Ls], "s-",
           label="single step")
    a.plot(Ls, [Bv[f"L={L}um"]["ens50_median"] for L in Ls], "d-",
           label="50-nm ensemble median")
    a.axvline(2.5, color="gray", ls=":", lw=1, label="production L")
    a.set_xlabel("L (µm)")
    a.set_ylabel(r"E$_2$ ($\mu$eV)")
    a.set_title("(b) length convergence")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    C = res["C_seeds"]
    ns = sorted(int(k) for k in C["running_median"])
    a.plot(ns, [C["running_median"][str(n)] for n in ns], "o-",
           label="running median")
    a.fill_between([min(ns), max(ns)], *([v] for v in C["bootstrap95"]),
                   color="C0", alpha=0.15, label="56-seed bootstrap 95% CI")
    a.set_xlabel("number of disorder seeds")
    a.set_ylabel(r"median E$_2$ ($\mu$eV)")
    a.set_title("(c) seed convergence, 50-nm same-sign ensemble")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    G = res["G_grid"]
    labels = list(next(iter(G.values())).keys())
    xs = np.arange(len(labels))
    for key, m in G.items():
        a.plot(xs, [m[l]["gap"] for l in labels], "o-", label=key, ms=4)
    a.set_xticks(xs); a.set_xticklabels(labels, fontsize=8)
    a.set_xlabel(r"optimizer grid $n_B \times n_\Delta \times n_\mu$")
    a.set_ylabel(r"best gap ($\mu$eV)")
    a.set_title("(d) hole-platform optimizer-grid convergence")
    a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig13_convergence.png"), dpi=150,
                bbox_inches="tight")
    print("figure written", flush=True)


SECS = dict(A=sec_A, B=sec_B, C=sec_C, D=sec_D, E=sec_E, F=sec_F, G=sec_G,
            H=sec_H)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sec", default="all")
    args = p.parse_args()
    todo = list(SECS) if args.sec == "all" else args.sec.split(",")
    res = _load()
    t0 = time.time()
    for s in todo:
        key = [k for k in res if k.startswith(s + "_")]
        if key:
            print(f"section {s}: cached, skipping", flush=True)
            continue
        print(f"=== section {s} (t={time.time()-t0:.0f}s)", flush=True)
        SECS[s](res)
        _save(res)
    if all(any(k.startswith(s + "_") for k in res) for s in SECS):
        if MPL_STUB:
            print("matplotlib stub active: fig13 deferred (data complete in "
                  "convergence.json)", flush=True)
        else:
            make_figure(res)
        save_numbers("convergence", dict(
            {k: v for k, v in res.items() if k != "C_seeds"},
            C_seeds={kk: vv for kk, vv in res["C_seeds"].items()
                     if kk != "values"}))
