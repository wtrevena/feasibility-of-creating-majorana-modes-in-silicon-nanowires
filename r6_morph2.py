"""r6_morph2.py -- review-round-6 item A3: fab-facing extensions of the
step-morphology program (reviewer demands, round 6).

Three sections (checkpointed per parameter point to output/data/r6_morph2.json,
summaries merged into key_numbers.json under tag "r6_morph2"):

  Y  Wafer-spec device yields. Converts the step ensembles of morphology.py
     into yield numbers for COMMERCIAL Si(001) wafer miscut specs:
     premium +-0.05 deg, standard +-0.1 deg. Within-spec miscut is sampled
     conservatively as theta_m ~ U[0.2*spec, spec] (4 stratified values =
     quartile midpoints; the measure-zero theta=0 region is excluded), wire
     vs step-edge angle chi in {90 (unaligned), 30 (coarse), 10 (good)} deg,
     gamma terraces kgam=4, abrupt 0.85*pi jumps, electron wedge-point
     baseline (fig9: mu=35 ueV, B=1.5 T, alpha=0.05 eV*A, lam=75 ueV,
     N=500, dx=5 nm), 10 seeds per (theta, chi) cell, pooled to 40
     realizations per (spec, chi).
     Yield = fraction with E2 >= 4 ueV (the single-0.85pi-step ceiling is
     4.03 ueV at this operating point, so this threshold means "the device
     is at least as good as a wire with one damaging step" -- a usable-
     device proxy). Fraction with E2 >= 1 ueV also reported.

  V  Finite valley splitting x smooth steps. Single-step (0.85*pi, tanh
     ramp width w) E2 and clean-wire E2 on the grid lam in {40,75,150,300}
     ueV x w in {0,2,5,10,20} nm, fine grid (N=1000, dx=2.5 nm) so ramps
     are resolved. mu is re-centered per lam:
         mu(lam) = max(lam - 40, mu_crit + 20 - lam),  mu_crit = sqrt(EZ^2
         - Delta^2) ~ 71 ueV,
     i.e. the lower valley band sits at mu_eff = -40 ueV (the fig9 baseline)
     whenever that also leaves the UPPER band trivial by >= 20 ueV; at
     lam=40 the second branch takes over (mu ~ 51 ueV, lower band at +11),
     because with mu = lam-40 BOTH bands would be topological (two MZM
     pairs, E2 meaningless). At lam=40 the upper band's trivial gap
     (~17 ueV) caps the clean E2 -- an intrinsic small-valley-splitting
     penalty, not step physics. Topology is verified per point via
     E0 << E2 (flag stored).

  C  Electrostatics-calibrated correlated disorder. Replaces the abstract
     25/50/100-ueV-RMS GRF with disorder derived from interface trap
     charges. 2D PROXY (documented): a single trap at the Si/oxide
     interface of the fin cross-section is modeled in poisson2d.FinPoisson
     as a LINE charge q = +-e/(2*lc) at one interface node (the point trap
     smeared over its along-wire action length 2*lc); the device is
     re-converged with the full SCF loop (toy density, n_l=2e7 /m,
     V_g=-1.2 V) so the mobile channel charge screens the trap; the
     potential shift dphi at the (unperturbed) channel density centroid
     gives the per-trap amplitude A_trap = e*dphi (ueV). A_trap is averaged
     over 3 interface sites (top center, sidewall middle, top corner) and
     both signs. Along-wire mu(x): trap positions Poisson-distributed with
     areal density n_trap in {1e10, 5e10, 2e11} cm^-2 times the gated fin
     perimeter P = 34 nm (two sidewalls + top; Si/BOX interface excluded),
     each trap contributing s_i * A_trap * exp(-|x-x_i|/lc) with random
     sign s_i and lc = 15 nm (~ the fin half-perimeter; the observed
     cross-section decay length is also recorded). The resulting mu(x)
     feeds realism.build_wire_sitewise at the [110] SiB_pauli point
     (m=0.204, alpha=0.073 eV*A, g=1.657, B=1.0 T, Dind=24.7 ueV, N=1200,
     dx=2.5 nm), 10 seeds per density, plus GRF references (RMS 25/50 ueV,
     lc=50 nm) at the SAME point. Calibration output: which n_trap
     reproduces the paper's 25/50 ueV RMS levels (shot-noise formula
     RMS^2 = A_trap^2 * n_trap * P * lc, cross-checked against measured
     RMS).

  PROXY CAVEATS (section C): (i) the 2D line-charge smearing is exact only
  up to a logarithmic geometry factor -- A_trap is an order-of-magnitude
  calibration, not an exact map; (ii) the toy SCF density is a documented
  stand-in for the k.p density (poisson2d docstring); (iii) total channel
  charge is held fixed during screening (canonical, not grand-canonical);
  (iv) the along-wire kernel exp(-|x|/lc) with a single lc ignores the
  trap-depth distribution; (v) trap signs are taken random +-e with zero
  mean.

Usage: python r6_morph2.py --sec Y|V|C|all
"""
import argparse
import json
import os
import time

import numpy as np

import morphology as mo
import poisson2d as p2d
from majorana_sim import (build_wire_two_valley_iv, solve_lowest, UEV, EZ_J,
                          QE)
from realism import build_wire_sitewise, grf
from run_analysis import (save_numbers, DATA, M_SI, G_SI, DELTA, ALPHA_DEMO)

CKPT = os.path.join(DATA, "r6_morph2.json")


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def save_ckpt(ck):
    with open(CKPT, "w") as f:
        json.dump(ck, f, indent=1)


def pct(a, q):
    return round(float(np.percentile(np.asarray(a, float), q)), 3)


# ================================================================ section Y
SPECS = {"premium_0.05deg": 0.05, "standard_0.10deg": 0.10}
CHIS = [90, 30, 10]
NSEED_Y = 10
THR_HI, THR_LO = 4.0, 1.0      # ueV; see module docstring


def theta_values(spec):
    """4 stratified values of theta_m ~ U[0.2*spec, spec] (quartile
    midpoints): spec * (0.3, 0.5, 0.7, 0.9)."""
    lo, hi = 0.2 * spec, spec
    return [lo + (hi - lo) * f for f in (0.125, 0.375, 0.625, 0.875)]


def secY():
    t0 = time.time()
    ck = load_ckpt()
    ck.setdefault("Y", {})
    for sname, spec in SPECS.items():
        for chi in CHIS:
            for th in theta_values(spec):
                key = f"{sname}_chi{chi}_th{th:.4f}deg"
                if key in ck["Y"]:
                    continue
                sm = mo.s_mean_from_miscut(th)
                E0s, E2s, nst = [], [], []
                for s in range(NSEED_Y):
                    rng = np.random.default_rng(
                        [101, int(spec * 1000), chi, int(th * 1e4), s])
                    pos = mo.gen_step_positions(mo.N0 * mo.DX0, sm, 4.0,
                                                chi, rng)
                    vo = mo.morph_profile(mo.N0, mo.DX0, pos, mo.DPHI1)
                    e0, e2 = mo.solve_E0_E2(vo, mo.N0, mo.DX0)
                    E0s.append(round(e0, 6))
                    E2s.append(round(e2, 6))
                    nst.append(int(len(pos)))
                ck["Y"][key] = dict(spec=sname, chi=chi,
                                    theta_deg=round(th, 5),
                                    s_mean_nm=round(sm * 1e9, 1),
                                    n_steps=nst, E0=E0s, E2=E2s)
                save_ckpt(ck)
                print(f"  [Y/{key}] med E2 {np.median(E2s):.2f} ueV  "
                      f"t={time.time()-t0:.0f}s", flush=True)
    # ---- pooled summary per (spec, chi)
    summ = {"_yield_def": f"yield = frac(E2 >= {THR_HI} ueV) "
                          "(single-0.85pi-step ceiling = 4.03 ueV at this "
                          "operating point); also frac(E2 >= 1 ueV)",
            "_sampling": "theta_m stratified U[0.2*spec, spec] x 4, kgam=4, "
                         "chi = wire vs step-edge angle, 10 seeds/cell, "
                         "40 realizations per (spec, chi)"}
    for sname in SPECS:
        for chi in CHIS:
            recs = [r for r in ck["Y"].values()
                    if isinstance(r, dict) and r.get("spec") == sname
                    and r.get("chi") == chi]
            E2 = np.concatenate([r["E2"] for r in recs])
            E0 = np.concatenate([r["E0"] for r in recs])
            ns = np.concatenate([r["n_steps"] for r in recs])
            summ[f"{sname}_chi{chi}deg"] = dict(
                yield_E2_ge_4ueV=round(float(np.mean(E2 >= THR_HI)), 3),
                frac_E2_ge_1ueV=round(float(np.mean(E2 >= THR_LO)), 3),
                p5=pct(E2, 5), p50=pct(E2, 50), p95=pct(E2, 95),
                frac_zero_steps=round(float(np.mean(ns == 0)), 3),
                mean_steps=round(float(np.mean(ns)), 1),
                E0_max=round(float(np.max(E0)), 3), n=int(len(E2)))
    save_numbers("r6_morph2", {"secY_wafer_yields": summ})
    print(json.dumps(summ, indent=1), flush=True)


# ================================================================ section V
LAMS = [40.0, 75.0, 150.0, 300.0]
WS_NM = [0, 2, 5, 10, 20]
BV = 1.5
EZ_UEV = EZ_J(G_SI, BV) / UEV
MU_CRIT = float(np.sqrt(EZ_UEV**2 - DELTA**2))     # ~70.99 ueV
MARGIN = 20.0


def mu_of_lam(lam):
    """Recentering rule (see module docstring): lower band at mu_eff=-40
    when possible, while keeping the upper band trivial by >= MARGIN."""
    return max(lam - 40.0, MU_CRIT + MARGIN - lam)


def solve_mu(vo, mu):
    H = build_wire_two_valley_iv(mo.NF, mo.DXF, mu, BV, DELTA, ALPHA_DEMO,
                                 M_SI, G_SI, vo_profile_ueV=vo)
    E, _ = solve_lowest(H, k=6)
    Ea = np.sort(np.abs(E)) / UEV
    return float(Ea[0]), float(Ea[2])


def secV():
    t0 = time.time()
    ck = load_ckpt()
    ck.setdefault("V", {})
    L = mo.NF * mo.DXF
    for lam in LAMS:
        mu = mu_of_lam(lam)
        ckey = f"lam{lam:.0f}_clean"
        if ckey not in ck["V"]:
            vo = mo.morph_profile(mo.NF, mo.DXF, [], mo.DPHI1, lam0=lam)
            e0, e2 = solve_mu(vo, mu)
            ck["V"][ckey] = dict(lam=lam, mu=round(mu, 2),
                                 mu_eff_lower=round(mu - lam, 2),
                                 mu_eff_upper=round(mu + lam, 2),
                                 E0=round(e0, 6), E2=round(e2, 6))
            save_ckpt(ck)
            print(f"  [V/{ckey}] E2 {e2:.2f}  t={time.time()-t0:.0f}s",
                  flush=True)
        for w in WS_NM:
            key = f"lam{lam:.0f}_w{w}nm"
            if key in ck["V"]:
                continue
            vo = mo.morph_profile(mo.NF, mo.DXF, [L / 2], mo.DPHI1,
                                  w_ramp=w * 1e-9, lam0=lam)
            e0, e2 = solve_mu(vo, mu)
            ck["V"][key] = dict(lam=lam, w_nm=w, mu=round(mu, 2),
                                E0=round(e0, 6), E2=round(e2, 6))
            save_ckpt(ck)
            print(f"  [V/{key}] E2 {e2:.2f}  t={time.time()-t0:.0f}s",
                  flush=True)
    # ---- summary: E2_step/E2_clean ratio grid
    summ = {"_rule": "mu(lam) = max(lam-40, mu_crit+20-lam), mu_crit = "
                     f"{MU_CRIT:.1f} ueV; lower band at mu_eff=-40 for "
                     "lam>=75; at lam=40 mu=51 keeps the upper band "
                     "trivial (its ~17 ueV gap caps clean E2)",
            "_grid": "N=1000, dx=2.5 nm, single 0.85pi step at L/2, "
                     "tanh ramp width w"}
    for lam in LAMS:
        cl = ck["V"][f"lam{lam:.0f}_clean"]
        row = dict(mu_ueV=cl["mu"], E2_clean=round(cl["E2"], 3),
                   E0_clean=round(cl["E0"], 4))
        for w in WS_NM:
            r = ck["V"][f"lam{lam:.0f}_w{w}nm"]
            row[f"ratio_w{w}nm"] = round(r["E2"] / max(cl["E2"], 1e-9), 3)
            row[f"E2_w{w}nm"] = round(r["E2"], 3)
            row[f"topo_ok_w{w}nm"] = bool(r["E0"] < 0.5 and
                                          r["E0"] < 0.25 * max(r["E2"], 1e-9))
        summ[f"lam{lam:.0f}ueV"] = row
    save_numbers("r6_morph2", {"secV_valley_x_ramp": summ})
    print(json.dumps(summ, indent=1), flush=True)


# ================================================================ section C
V_G, N_L = -1.2, 2e7
LC = 15e-9                      # along-wire trap kernel length (~ half-perim)
PERIM = 34e-9                   # gated Si/oxide perimeter: 2*12 + 10 nm
Q_LINE = QE / (2 * LC)          # 2D line-charge proxy for one trap
PT110 = dict(m=0.204, al=0.073, g=1.657, B=1.0, Dind=24.7,
             N=1200, dx=2.5e-9)
DENS_CM2 = {"1e10": 1e10, "5e10": 5e10, "2e11": 2e11}
NSEED_C = 10


class _TrapFin(p2d.FinPoisson):
    """FinPoisson with an additional fixed background charge (the trap),
    included in every Poisson solve of the SCF loop so the mobile channel
    charge screens it self-consistently."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rho_extra = None

    def solve(self, V_g, rho=None):
        if self.rho_extra is not None:
            rho = (self.rho_extra if rho is None
                   else rho + self.rho_extra)
        return super().solve(V_g, rho)


def _calibrate_trap(ck):
    """A_trap (ueV) from SCF-screened single-trap solves."""
    if "calibration" in ck["C"]:
        return ck["C"]["calibration"]["A_trap_ueV"]
    t0 = time.time()
    dev = _TrapFin()
    phi0, nit0, _ = dev.scf(p2d.toy_density, n_l=N_L, V_g=V_G,
                            tol=1e-6, max_iter=600)
    pf0 = dev.phi_fin(phi0)
    n2d0 = dev._last_n2d
    wsum = (n2d0 * dev.wf).sum()
    ybar = float((n2d0 * dev.wf * dev.yf[:, None]).sum() / wsum)
    zbar = float((n2d0 * dev.wf * dev.zf[None, :]).sum() / wsum)
    Wy, Wz = dev.Wy, dev.Wz
    sites = {"top_center": (0.0, Wz), "sidewall_mid": (Wy / 2, Wz / 2),
             "top_corner": (Wy / 2, Wz)}
    y, z = dev.core.y, dev.core.z
    res = {}
    dphi_tc_pos = None
    for sname, (yt, zt) in sites.items():
        iy = int(np.argmin(np.abs(y - yt)))
        iz = int(np.argmin(np.abs(z - zt)))
        for sign in (+1, -1):
            rho = np.zeros((dev.core.Ny, dev.core.Nz))
            rho[iy, iz] = sign * Q_LINE / dev.core.area[iy, iz]
            dev.rho_extra = rho
            phi1, nit, _ = dev.scf(p2d.toy_density, n_l=N_L, V_g=V_G,
                                   tol=1e-6, max_iter=600)
            dev.rho_extra = None
            dpf = dev.phi_fin(phi1) - pf0
            dphi = float(p2d.bilinear_interp(dev.yf, dev.zf, dpf,
                                             [ybar], [zbar])[0, 0])
            res[f"{sname}_{'+e' if sign > 0 else '-e'}"] = dict(
                dphi_centroid_ueV=round(dphi * 1e6, 2), scf_iters=nit)
            if sname == "top_center" and sign > 0:
                dphi_tc_pos = dpf
            print(f"  [C/trap {sname} {sign:+d}e] dphi "
                  f"{dphi*1e6:.1f} ueV  t={time.time()-t0:.0f}s", flush=True)
    A = float(np.mean([abs(v["dphi_centroid_ueV"]) for v in res.values()]))
    # observed cross-section decay length (top-center +e, line y=0)
    iy0 = int(np.argmin(np.abs(dev.yf - 0.0)))
    prof = np.abs(dphi_tc_pos[iy0, :])
    d = dev.zf[-1] - dev.zf                       # distance below the trap
    m = (d > 0.5e-9) & (d < 9e-9) & (prof > 0)
    lc_obs = np.nan
    if m.sum() > 3:
        slope = np.polyfit(d[m], np.log(prof[m]), 1)[0]
        lc_obs = -1.0 / slope if slope < 0 else np.inf
    ck["C"]["calibration"] = dict(
        per_site=res, A_trap_ueV=round(A, 2),
        A_def="mean |e*dphi(centroid)| over 3 interface sites x both signs",
        q_line_C_per_m=float(f"{Q_LINE:.4e}"),
        proxy="point trap smeared over 2*lc = 30 nm along wire "
              "(2D line charge), SCF-screened (toy density, canonical)",
        V_g=V_G, n_l=N_L, lc_kernel_nm=LC * 1e9, perimeter_nm=PERIM * 1e9,
        lc_observed_xsec_nm=round(float(lc_obs * 1e9), 2),
        centroid_y_nm=round(ybar * 1e9, 2), centroid_z_nm=round(zbar * 1e9, 2),
        baseline_scf_iters=nit0)
    save_ckpt(ck)
    return A


def _e0e2_sitewise(mu_arr):
    p = PT110
    ones = np.ones(p["N"])
    H = build_wire_sitewise(p["N"], p["dx"], mu_arr, p["B"],
                            p["Dind"] * ones, p["al"] * ones,
                            p["m"], p["g"])
    E, _ = solve_lowest(H, k=6)
    Ea = np.sort(np.abs(E)) / UEV
    return float(Ea[0]), float(Ea[2])


def secC():
    t0 = time.time()
    ck = load_ckpt()
    ck.setdefault("C", {})
    A = _calibrate_trap(ck)
    N, dx = PT110["N"], PT110["dx"]
    L = N * dx
    x = np.arange(N) * dx
    if "wire_clean" not in ck["C"]:
        e0, e2 = _e0e2_sitewise(np.zeros(N))
        ck["C"]["wire_clean"] = dict(E0=round(e0, 6), E2=round(e2, 6))
        save_ckpt(ck)
        print(f"  [C/clean] E2 {e2:.2f}  t={time.time()-t0:.0f}s", flush=True)
    for k, (dk, dens) in enumerate(DENS_CM2.items()):
        key = f"ntrap_{dk}_cm2"
        if key in ck["C"]:
            continue
        n_lin = dens * 1e4 * PERIM            # traps per m of wire
        E0s, E2s, rms = [], [], []
        for s in range(NSEED_C):
            rng = np.random.default_rng([113, k, s])
            cnt = rng.poisson(n_lin * L)
            mu = np.zeros(N)
            if cnt:
                xi = rng.uniform(0, L, cnt)
                sg = rng.integers(0, 2, cnt) * 2.0 - 1.0
                mu = (sg[None, :] * A
                      * np.exp(-np.abs(x[:, None] - xi[None, :]) / LC)
                      ).sum(axis=1)
            e0, e2 = _e0e2_sitewise(mu)
            E0s.append(round(e0, 6))
            E2s.append(round(e2, 6))
            rms.append(round(float(mu.std()), 2))
        ck["C"][key] = dict(dens_cm2=dens, n_lin_per_um=round(n_lin * 1e-6, 2),
                            rms_ueV=rms, E0=E0s, E2=E2s)
        save_ckpt(ck)
        print(f"  [C/{key}] rms~{np.mean(rms):.0f} ueV  med E2 "
              f"{np.median(E2s):.2f}  t={time.time()-t0:.0f}s", flush=True)
    for W in (25.0, 50.0):
        key = f"grf_rms{W:.0f}_lc50nm"
        if key in ck["C"]:
            continue
        E0s, E2s = [], []
        for s in range(NSEED_C):
            rng = np.random.default_rng([114, int(W), s])
            mu = W * grf(N, dx, 50e-9, rng)
            e0, e2 = _e0e2_sitewise(mu)
            E0s.append(round(e0, 6))
            E2s.append(round(e2, 6))
        ck["C"][key] = dict(E0=E0s, E2=E2s)
        save_ckpt(ck)
        print(f"  [C/{key}] med E2 {np.median(E2s):.2f}  "
              f"t={time.time()-t0:.0f}s", flush=True)
    # ---- summary + calibration of the abstract RMS levels
    cal = ck["C"]["calibration"]
    summ = {"_point": "[110] SiB_pauli: m=0.204, al=0.073, g=1.657, B=1.0 T, "
                      "Dind=24.7 ueV, N=1200, dx=2.5 nm",
            "calibration": cal,
            "wire_clean_E2": ck["C"]["wire_clean"]["E2"]}
    for dk in DENS_CM2:
        r = ck["C"][f"ntrap_{dk}_cm2"]
        E2 = np.asarray(r["E2"])
        summ[f"ntrap_{dk}_cm2"] = dict(
            rms_mean_ueV=round(float(np.mean(r["rms_ueV"])), 1),
            rms_theory_ueV=round(
                A * np.sqrt(r["dens_cm2"] * 1e4 * PERIM * LC), 1),
            p5=pct(E2, 5), p50=pct(E2, 50), p95=pct(E2, 95),
            frac_E2_below_1ueV=round(float(np.mean(E2 < 1.0)), 2),
            E0_max=round(float(np.max(r["E0"])), 3))
    for W in (25.0, 50.0):
        r = ck["C"][f"grf_rms{W:.0f}_lc50nm"]
        E2 = np.asarray(r["E2"])
        summ[f"grf_rms{W:.0f}_lc50nm"] = dict(
            p5=pct(E2, 5), p50=pct(E2, 50), p95=pct(E2, 95))
    # density that reproduces the abstract RMS levels (shot-noise formula)
    for W in (25.0, 50.0, 100.0):
        dens = W**2 / (A**2 * PERIM * LC) / 1e4       # cm^-2
        summ[f"ntrap_for_rms{W:.0f}ueV_cm2"] = float(f"{dens:.3g}")
    save_numbers("r6_morph2", {"secC_trap_disorder": summ})
    print(json.dumps(summ, indent=1), flush=True)


SECS = dict(Y=secY, V=secV, C=secC)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all", choices=list(SECS) + ["all"])
    args = ap.parse_args()
    for name in (SECS if args.sec == "all" else [args.sec]):
        print(f"=== section {name} ===", flush=True)
        SECS[name]()
