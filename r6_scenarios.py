"""
r6_scenarios.py — review-round-6 item A1: three reviewer-demanded controls.

Sections (fine-grained checkpoints in output/data/r6_scenarios.json; ledger
tag "r6_scenarios" written when all three sections complete):

  T  Tilted-field DYNAMIC self-energy. platform110 sec B treated the
     g_z-riding tilted measured-Si:B case statically only (12.5 / 17.1 ueV
     bare at theta=90 for the [110] Ez=10/30 tensors). Re-run here with the
     validated frequency-dependent solver (realism.dynamic_gap_ueV).
     Method (scalar-g_eff): at tilt angle theta from the in-plane wire axis
     x' toward out-of-plane z, the wire is a single channel with scalar
       g_eff(theta) = sqrt((g_x' cos th)^2 + (g_z sin th)^2).
     For the [110] tensors the SOC axis is y' (kp6_110 nso = +-y), so a
     field tilted in the x'-z plane stays perpendicular to the SOC axis and
     the scalar reduction is exact at this level. For the [100]-empirical
     tensor (nso ~ 0.41 y + 0.91 z) the scalar form NEGLECTS the
     pair-breaking Zeeman component parallel to the SOC axis, so those
     tilted numbers are upper bounds. Parent: thick measured-class Si:B,
     isotropic GL Bc2 = 0.4 T, direction-independent and FROZEN (no
     feedback of the wire on the parent). Static counterpart computed on
     the identical (Gamma, B) grid at matched Dind = Gamma*Dp/(Gamma+Dp).

  G  Gate-tuning protocol widths at the three platform110 sec-C dynamic
     optima ([110] Ez=10 parameters m=0.194, alpha=0.052 eV*A, g=1.549):
     window (gap >= 50% of the operating-point value, linear interp) as
     mu (+-30 ueV), B (+-30%), Gamma (x0.5..x2) vary one at a time.

  D  Al-specific disorder. The paper previously inferred Al-scenario
     disorder margins from Si:B-class checks; here finite-wire ensembles
     run AT the Al operating points: N=1200, dx=2.5 nm; iid onsite
     disorder W in {100..1600} ueV and correlated GRF mu(x) (lc=50 nm,
     RMS {25,50,100} ueV), 10 seeds each, for the [110] Ez=10 parameters
     (B=1.753 T, Dind=42 from platform110 A) and the old empirical set
     (m=0.25, al=0.06, g=2.2, B=1.7, Dind=60).

Usage: python r6_scenarios.py [--sec T,G,D|all]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import scipy.sparse  # noqa: F401
except Exception:
    sys.path.insert(0, os.path.join(_HERE, "compat"))

from majorana_sim import topological_gap_ueV, build_wire
from run_analysis import DATA, save_numbers, _parent_gap_ueV
from realism import dynamic_gap_ueV, build_wire_sitewise, grf, _e2

CKPT = os.path.join(DATA, "r6_scenarios.json")


def _load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def _save(d):
    json.dump(d, open(CKPT, "w"), indent=2)


# ------------------------------------------------------- T: tilted + dynamic
TENSORS = {
    "t110_Ez10": dict(m=0.194, al=0.052, gx=1.549, gz=2.632),
    "t110_Ez30": dict(m=0.204, al=0.073, gx=1.657, gz=3.441),
    "emp_100":   dict(m=0.25,  al=0.06,  gx=2.1,   gz=2.7),
}
THETAS = (0, 20, 39, 60, 90)
GAMMAS = (3, 6, 10, 15, 20, 30, 45)
BGRID = np.linspace(0.1, 0.38, 6)        # coarse, up to 0.95*Bc2


def sec_T(res):
    out = res.setdefault("T_tilted_dynamic", {})
    for tens, p in TENSORS.items():
        for th in THETAS:
            key = f"{tens}_th{th}"
            if key in out:
                continue
            t0 = time.time()
            rad = np.deg2rad(th)
            geff = float(np.hypot(p["gx"] * np.cos(rad),
                                  p["gz"] * np.sin(rad)))
            row, srow = {}, {}
            best, sbest = (0.0, None, None), (0.0, None, None)
            for G in GAMMAS:
                gG, sG = 0.0, 0.0
                for B in BGRID:
                    Dp = _parent_gap_ueV(float(B), "SiB_meas")
                    gap = dynamic_gap_ueV(0.0, float(B), Dp, G,
                                          p["al"], p["m"], geff)
                    gG = max(gG, gap)
                    if gap > best[0]:
                        best = (gap, G, float(B))
                    Dind = G * Dp / (G + Dp)
                    sgap = topological_gap_ueV(0.0, float(B), Dind,
                                               p["al"], p["m"], geff)
                    sG = max(sG, sgap)
                    if sgap > sbest[0]:
                        sbest = (sgap, G, float(B))
                row[str(G)] = round(gG, 2)
                srow[str(G)] = round(sG, 2)
            ent = dict(g_eff=round(geff, 3),
                       dyn_gap_vs_Gamma=row, dyn_best=round(best[0], 2),
                       Gamma_star=best[1], B_star=best[2],
                       static_matched_vs_Gamma=srow,
                       static_best=round(sbest[0], 2),
                       static_Gamma_star=sbest[1], static_B_star=sbest[2],
                       runtime_s=round(time.time() - t0, 1))
            if best[1] is not None:
                DpS = _parent_gap_ueV(best[2], "SiB_meas")
                Dm = best[1] * DpS / (best[1] + DpS)
                ent["Dp_at_Bstar"] = round(DpS, 1)
                ent["Dind_matched"] = round(Dm, 1)
                ent["static_at_dyn_opt"] = round(
                    topological_gap_ueV(0.0, best[2], Dm, p["al"], p["m"],
                                        geff), 2)
            out[key] = ent
            print("T", key, "geff", ent["g_eff"], "dyn", ent["dyn_best"],
                  "static", ent["static_best"], flush=True)
            _save(res)
    for tens in TENSORS:
        skey = f"{tens}_summary"
        if skey in out:
            continue
        bt = max(THETAS, key=lambda t: out[f"{tens}_th{t}"]["dyn_best"])
        e = out[f"{tens}_th{bt}"]
        out[skey] = dict(theta_star=bt, g_eff=e["g_eff"],
                         dyn_best=e["dyn_best"], Gamma_star=e["Gamma_star"],
                         B_star=e["B_star"],
                         static_best_at_theta_star=e["static_best"])
        print("T summary", tens, out[skey], flush=True)
    out["_done"] = True
    _save(res)


# ------------------------------------------------- G: gate-tuning windows
# operating points from the platform110 C-section ledger ([110] Ez=10)
G_OPS = {
    "SiB_meas":  dict(B=0.38,  Gamma=6,  parent="SiB_meas"),
    "SiB_pauli": dict(B=1.0,   Gamma=20, parent="SiB_pauli"),
    "Al":        dict(B=1.753, Gamma=30, parent="Al"),
}
P110 = dict(m=0.194, al=0.052, g=1.549)


def _window(xs, ys, i0):
    """Contiguous window around index i0 where ys >= 0.5*ys[i0] (linear
    interpolation at the edges). None if ys[i0] <= 0."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if ys[i0] <= 0:
        return None
    thr = 0.5 * ys[i0]
    i = i0
    while i > 0 and ys[i - 1] >= thr:
        i -= 1
    if i == 0:
        lo, clo = float(xs[0]), True
    else:
        lo = float(xs[i - 1] + (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
                   * (xs[i] - xs[i - 1]))
        clo = False
    j, n = i0, len(xs)
    while j < n - 1 and ys[j + 1] >= thr:
        j += 1
    if j == n - 1:
        hi, chi = float(xs[-1]), True
    else:
        hi = float(xs[j] + (ys[j] - thr) / (ys[j] - ys[j + 1])
                   * (xs[j + 1] - xs[j]))
        chi = False
    return lo, hi, clo, chi


def _fmt_win(w, unit):
    if w is None:
        return dict(note="gap at operating point is 0")
    lo, hi, clo, chi = w
    return dict(lo=round(lo, 3), hi=round(hi, 3), width=round(hi - lo, 3),
                unit=unit, censored_low=clo, censored_high=chi)


def sec_G(res):
    out = res.setdefault("G_tuning", {})
    for tag, op in G_OPS.items():
        if tag in out:
            continue
        t0 = time.time()
        p = P110
        B0, G0 = op["B"], op["Gamma"]
        Dp0 = _parent_gap_ueV(B0, op["parent"])
        gap0 = dynamic_gap_ueV(0.0, B0, Dp0, G0, p["al"], p["m"], p["g"])
        d = dict(op=dict(B=B0, Gamma=G0, Dp=round(Dp0, 1),
                         gap=round(gap0, 2)))
        # mu knob (+-30 ueV)
        mus = np.linspace(-30, 30, 7)
        gm = [dynamic_gap_ueV(float(mu), B0, Dp0, G0, p["al"], p["m"],
                              p["g"]) for mu in mus]
        d["mu_scan_ueV"] = {f"{mu:+.0f}": round(g_, 2)
                            for mu, g_ in zip(mus, gm)}
        d["mu_window"] = _fmt_win(_window(mus, gm, 3), "ueV")
        # B knob (+-30%)
        Bs = np.linspace(0.7 * B0, 1.3 * B0, 7)
        gB = [dynamic_gap_ueV(0.0, float(B),
                              _parent_gap_ueV(float(B), op["parent"]),
                              G0, p["al"], p["m"], p["g"]) for B in Bs]
        d["B_scan_T"] = {f"{B:.3f}": round(g_, 2) for B, g_ in zip(Bs, gB)}
        w = _window(Bs, gB, 3)
        d["B_window"] = _fmt_win(w, "T")
        if w is not None:
            d["B_window"]["width_pct_of_B0"] = round(
                100 * (w[1] - w[0]) / B0, 1)
        # Gamma knob (x0.5 .. x2, log-spaced)
        Gs = G0 * 2.0 ** np.linspace(-1, 1, 7)
        gG = [dynamic_gap_ueV(0.0, B0, Dp0, float(Gm), p["al"], p["m"],
                              p["g"]) for Gm in Gs]
        d["Gamma_scan_ueV"] = {f"{Gm:.2f}": round(g_, 2)
                               for Gm, g_ in zip(Gs, gG)}
        w = _window(Gs, gG, 3)
        d["Gamma_window"] = _fmt_win(w, "ueV")
        if w is not None:
            d["Gamma_window"]["factor_span"] = round(w[1] / max(w[0], 1e-9),
                                                     2)
        d["runtime_s"] = round(time.time() - t0, 1)
        out[tag] = d
        print("G", tag, "op gap", d["op"]["gap"],
              "mu", d["mu_window"], "B", d["B_window"],
              "Gamma", d["Gamma_window"], flush=True)
        _save(res)
    out["_done"] = True
    _save(res)


# ----------------------------------------------- D: Al-specific disorder
D_SETS = {
    "Al_110_Ez10": dict(m=0.194, al=0.052, g=1.549, B=1.753, Dind=42.0),
    "Al_emp_100":  dict(m=0.25,  al=0.06,  g=2.2,   B=1.7,   Dind=60.0),
}
N_W, DX_W = 1200, 2.5e-9
IID_WS = (100, 200, 400, 800, 1600)
GRF_RMS = (25, 50, 100)
NSEED = 10


def _pct(es):
    return dict(p5=round(float(np.percentile(es, 5)), 2),
                p50=round(float(np.median(es)), 2),
                p95=round(float(np.percentile(es, 95)), 2))


def sec_D(res):
    out = res.setdefault("D_al_disorder", {})
    for idx, (sname, p) in enumerate(D_SETS.items()):
        d = out.setdefault(sname, {})
        if "clean_E2" not in d:
            H = build_wire(N_W, DX_W, 0.0, p["B"], p["Dind"], p["al"],
                           p["m"], p["g"])
            d["clean_E2"] = round(_e2(H, N_W), 2)
            d["bulk_gap"] = round(
                topological_gap_ueV(0.0, p["B"], p["Dind"], p["al"],
                                    p["m"], p["g"]), 2)
            print("D", sname, "clean E2", d["clean_E2"], "bulk",
                  d["bulk_gap"], flush=True)
            _save(res)
        iid = d.setdefault("iid", {})
        for W in IID_WS:
            k = f"W{W}"
            if k in iid:
                continue
            t0 = time.time()
            es = []
            for s in range(NSEED):
                rng = np.random.default_rng([61, idx, W, s])
                H = build_wire(N_W, DX_W, 0.0, p["B"], p["Dind"], p["al"],
                               p["m"], p["g"], disorder_ueV=float(W),
                               rng=rng)
                es.append(_e2(H, N_W))
            iid[k] = {**_pct(es), "runtime_s": round(time.time() - t0, 1)}
            print("D iid", sname, W, iid[k], flush=True)
            _save(res)
        if "W_half_ueV" not in d:
            med = np.array([iid[f"W{W}"]["p50"] for W in IID_WS], float)
            half = 0.5 * d["clean_E2"]
            Whalf = None
            for i in range(1, len(IID_WS)):
                if med[i - 1] >= half > med[i]:
                    f = (med[i - 1] - half) / (med[i - 1] - med[i])
                    Whalf = float(np.exp(
                        np.log(IID_WS[i - 1])
                        + f * (np.log(IID_WS[i]) - np.log(IID_WS[i - 1]))))
                    break
            if Whalf is not None:
                d["W_half_ueV"] = round(Whalf, 0)
            else:
                d["W_half_ueV"] = (f">{IID_WS[-1]}" if med[-1] >= half
                                   else f"<{IID_WS[0]}")
            print("D", sname, "W_half", d["W_half_ueV"], flush=True)
            _save(res)
        gd = d.setdefault("grf_mu_lc50nm", {})
        ones = np.ones(N_W)
        for rms in GRF_RMS:
            k = f"rms{rms}"
            if k in gd:
                continue
            t0 = time.time()
            es = []
            for s in range(NSEED):
                rng = np.random.default_rng([62, idx, rms, s])
                mux = float(rms) * grf(N_W, DX_W, 50e-9, rng)
                H = build_wire_sitewise(N_W, DX_W, mux, p["B"],
                                        p["Dind"] * ones, p["al"] * ones,
                                        p["m"], p["g"])
                es.append(_e2(H, N_W))
            gd[k] = {**_pct(es), "runtime_s": round(time.time() - t0, 1)}
            print("D grf", sname, rms, gd[k], flush=True)
            _save(res)
    out["_done"] = True
    _save(res)


SECS = dict(T=sec_T, G=sec_G, D=sec_D)
SECKEY = dict(T="T_tilted_dynamic", G="G_tuning", D="D_al_disorder")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all")
    a = ap.parse_args()
    todo = list(SECS) if a.sec == "all" else a.sec.split(",")
    res = _load()
    t0 = time.time()
    for s_ in todo:
        if res.get(SECKEY[s_], {}).get("_done"):
            print("section", s_, "cached", flush=True)
            continue
        print(f"=== {s_} (t={time.time()-t0:.0f}s)", flush=True)
        SECS[s_](res)
        _save(res)
    if all(res.get(k, {}).get("_done") for k in SECKEY.values()):
        save_numbers("r6_scenarios", res)
        print("ALL DONE", flush=True)
