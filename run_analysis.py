"""
run_analysis.py — generates all figures and key numbers for the feasibility study.

Usage:  python run_analysis.py --fig 1     (or 2..7, or "all")

Figures are written to output/, scan data and key numbers to output/data/.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from majorana_sim import (
    HBAR, ME, QE, MU_B_EV, UEV, KB,
    EZ_J, is_topological, bulk_gap_ueV, topological_gap_ueV,
    build_wire, build_wire_two_valley, build_wire_two_valley_iv,
    build_wire_2d, step_phase_profile, subband_bottoms_ueV,
    solve_lowest, site_density, end_weight, majorana_metrics,
)


def first_crossing(x, y, level, rising=True):
    """First x where y crosses `level` (linear interp). None if never."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    above = y >= level if rising else y <= level
    idx = np.where(above)[0]
    idx = idx[idx > 0] if len(idx) and idx[0] == 0 and not above[0] else idx
    for i in idx:
        if i == 0:
            return float(x[0])
        y0, y1 = y[i-1], y[i]
        if y1 == y0:
            return float(x[i])
        f = (level - y0) / (y1 - y0)
        return float(x[i-1] + f * (x[i] - x[i-1]))
    return None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)

# Baseline silicon parameters
M_SI, G_SI = 0.19, 2.0
DELTA = 50.0            # ueV, induced gap (as in the original notebook)
DX = 5e-9               # lattice spacing
ALPHA_DEMO = 0.05       # eV*A — "engineered SOC", used for demonstration figures
ALPHA_SI_OPT = 1e-3     # eV*A — optimistic upper end for intrinsic Si interfaces
                        # (measured Si/SiGe & MOS Rashba ~ 1e-5..1e-3 eV*A)
ALPHA_INSB = 0.5        # eV*A — InSb-class SOC

UEV_TO_MK = UEV / KB * 1e3   # 1 ueV in mK (~11.6)


def save_numbers(tag, d):
    path = os.path.join(DATA, "key_numbers.json")
    allnum = {}
    if os.path.exists(path):
        with open(path) as f:
            allnum = json.load(f)
    allnum[tag] = {**allnum.get(tag, {}), **d}   # merge, never clobber a tag
    with open(path, "w") as f:
        json.dump(allnum, f, indent=2)
    print(f"[{tag}] " + json.dumps(d))


# ----------------------------------------------------------------- Figure 1
def fig1():
    """Validation: Majorana fan, end-localization (vs the old model's bulk
    state), exponential splitting vs L, finite-wire vs bulk gap."""
    t0 = time.time()
    N, L = 400, 2e-6
    Bs = np.linspace(0.02, 2.0, 40)
    fan, e2s, bulks = [], [], []
    for B in Bs:
        H = build_wire(N, DX, 0.0, B, DELTA, ALPHA_DEMO, M_SI, G_SI)
        E, _ = solve_lowest(H, k=14)
        fan.append(np.sort(E) / UEV)
        Eabs = np.sort(np.abs(E)) / UEV
        e2s.append(Eabs[2])
        bulks.append(bulk_gap_ueV(0.0, B, DELTA, ALPHA_DEMO, M_SI, G_SI))
    fan = np.array(fan)
    Bstar = DELTA * 1e-6 / (0.5 * G_SI * MU_B_EV)

    # (b) corrected MZM vs original notebook's "Majorana"
    H = build_wire(N, DX, 0.0, 1.5, DELTA, ALPHA_DEMO, M_SI, G_SI)
    E, V = solve_lowest(H, k=6)
    dens_new = site_density(V[:, 0], N)
    dens_new /= dens_new.sum()
    e0_new = E[0] / UEV
    # original (broken) model: spinless, EZ added to both blocks, alpha unused
    x = np.arange(N) * DX
    dxo = DX
    T = (-(HBAR**2) / (2 * M_SI * ME * dxo**2)) * (
        np.diag(-2 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
    EZo = 0.5 * G_SI * MU_B_EV * 0.5 * QE          # their B = 0.5 T
    Do = DELTA * UEV
    Ho = np.block([[T + EZo * np.eye(N), Do * np.eye(N)],
                   [Do * np.eye(N), -T + EZo * np.eye(N)]])
    Eo, Vo = np.linalg.eigh(Ho)
    i0 = np.argmin(np.abs(Eo))
    dens_old = np.abs(Vo[:N, i0])**2 + np.abs(Vo[N:, i0])**2
    dens_old /= dens_old.sum()
    e0_old = Eo[i0] / UEV

    # (c) splitting vs L
    Ls = np.linspace(0.4e-6, 4.0e-6, 13)
    splits, ews = [], []
    for Li in Ls:
        Ni = int(Li / DX)
        H = build_wire(Ni, DX, 0.0, 1.5, DELTA, ALPHA_DEMO, M_SI, G_SI)
        e0, _, ew = majorana_metrics(H, Ni)
        splits.append(max(e0, 1e-9))
        ews.append(ew)
    splits = np.array(splits)
    # exponential envelope fit on the decaying part
    mask = (splits > 1e-7) & (Ls > 0.5e-6)
    p = np.polyfit(Ls[mask], np.log(splits[mask]), 1)
    xi_fit = -1.0 / p[0]

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
    a = ax[0, 0]
    for j in range(fan.shape[1]):
        a.plot(Bs, fan[:, j], ".", color="C0", ms=3)
    a.axvline(Bstar, color="r", ls="--", lw=1,
              label=fr"analytic $B^*=\sqrt{{\Delta^2+\mu^2}}/(\frac{{1}}{{2}}g\mu_B)$ = {Bstar:.2f} T")
    a.set_xlabel("B (T)"); a.set_ylabel("E (µeV)")
    a.set_title("(a) Gap closing & reopening with Majorana fan  (µ=0)")
    a.set_ylim(-80, 80); a.legend(loc="upper left", fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(x * 1e9, dens_new, "C2", label=f"corrected model: |E|={abs(e0_new):.4f} µeV (MZM)")
    a.plot(x * 1e9, dens_old, "C3--", label=f"original notebook: E={e0_old:.1f} µeV (bulk state)")
    a.set_xlabel("x (nm)"); a.set_ylabel("probability / site")
    a.set_title("(b) Lowest mode: end-localized MZM vs old model")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.semilogy(Ls * 1e6, splits, "o", color="C0")
    Lf = np.linspace(0.5e-6, 4e-6, 100)
    a.semilogy(Lf * 1e6, np.exp(np.polyval(p, Lf)), "k--", lw=1,
               label=fr"exp fit: $\xi$ = {xi_fit*1e9:.0f} nm")
    a.set_xlabel("wire length L (µm)"); a.set_ylabel("|E$_0$| (µeV)")
    a.set_title("(c) Zero-mode splitting decays exponentially with L")
    a.legend(fontsize=9); a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    a.plot(Bs, e2s, "o", ms=4, label="finite wire: 3rd-smallest |E|")
    a.plot(Bs, bulks, "k-", lw=1.2, label="infinite wire: analytic bulk gap")
    a.axvline(Bstar, color="r", ls="--", lw=1)
    a.set_xlabel("B (T)"); a.set_ylabel("gap (µeV)")
    a.set_title("(d) Finite-wire gap vs analytic bulk gap")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    fig.suptitle(f"Model validation — Si parameters (m*=0.19 m$_e$, g=2), Δ={DELTA:.0f} µeV, "
                 f"α={ALPHA_DEMO} eV·Å (engineered), L={L*1e6:.0f} µm", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_model_validation.png"), dpi=150)
    save_numbers("fig1", dict(
        Bstar_T=round(Bstar, 3), xi_fit_nm=round(xi_fit * 1e9, 1),
        E0_at_2um_ueV=float(f"{abs(e0_new):.2e}"),
        end_weight_at_2um=round(ews[np.argmin(np.abs(Ls - 2e-6))], 3),
        old_model_E0_ueV=round(e0_old, 2),
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 2
def _bulk_gap_grid(mus_ueV, Bs_T, Delta_ueV, alpha_eVA, m_rel, g, nk=6001):
    """Vectorized bulk gap over a (B, mu) grid. Returns array (nB, nmu) in ueV."""
    m = m_rel * ME
    D = Delta_ueV * UEV
    aSI = alpha_eVA * 1e-10 * QE
    mus = mus_ueV[None, :] * UEV                     # (1, nmu)
    out = np.zeros((len(Bs_T), len(mus_ueV)))
    for i, B in enumerate(Bs_T):
        EZ = EZ_J(g, B)
        kF = np.sqrt(2 * m * (np.abs(mus).max() + EZ + D)) / HBAR
        kso = m * aSI / HBAR**2
        k = np.linspace(0, 4 * (kF + kso) + 2e7, nk)[:, None]   # (nk, 1)
        xi = HBAR**2 * k**2 / (2 * m) - mus                      # (nk, nmu)
        a = aSI * k
        root = np.sqrt(xi**2 * a**2 + EZ**2 * (xi**2 + D**2))
        e2 = np.maximum(xi**2 + a**2 + EZ**2 + D**2 - 2 * root, 0.0)
        out[i] = np.sqrt(e2.min(axis=0)) / UEV
    return out


def fig2():
    """Topological-gap phase diagrams in (mu, B): Si SOC vs InSb-class SOC."""
    t0 = time.time()
    mus = np.linspace(-150, 150, 201)
    Bs = np.linspace(0.01, 3.0, 181)
    panels = [(ALPHA_SI_OPT, f"intrinsic Si (optimistic): α = {ALPHA_SI_OPT} eV·Å"),
              (ALPHA_INSB, f"InSb-class SOC: α = {ALPHA_INSB} eV·Å")]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    ims = []
    maxgaps = []
    for a, (alpha, label) in zip(ax, panels):
        gap = _bulk_gap_grid(mus, Bs, DELTA, alpha, M_SI, G_SI)
        EZs = EZ_J(G_SI, Bs)[:, None]
        topo = EZs**2 > (DELTA * UEV)**2 + (mus[None, :] * UEV)**2
        gap_t = np.where(topo, gap, np.nan)
        maxgaps.append(float(np.nanmax(gap_t)))
        im = a.pcolormesh(mus, Bs, np.clip(gap_t, 1e-2, None),
                          norm=LogNorm(vmin=1e-2, vmax=60), cmap="viridis",
                          shading="auto")
        ims.append(im)
        # phase boundary
        mub = np.linspace(-150, 150, 400)
        Bb = np.sqrt((DELTA * UEV)**2 + (mub * UEV)**2) / (0.5 * G_SI * MU_B_EV * QE)
        a.plot(mub, Bb, "r--", lw=1.5, label=r"$E_Z=\sqrt{\Delta^2+\mu^2}$")
        a.set_xlabel("µ (µeV)")
        a.set_title(label, fontsize=10)
        a.legend(loc="lower right", fontsize=8)
        a.set_ylim(0, 3)
    ax[0].set_ylabel("B (T)")
    cb = fig.colorbar(ims[1], ax=ax, label="topological gap (µeV), log scale")
    cb.ax.axhline(2.2, color="w")  # ~25 mK
    ratio = maxgaps[1] / maxgaps[0]
    fig.suptitle(f"Topological gap, Si band parameters (m*=0.19, g=2), Δ={DELTA:.0f} µeV "
                 f"— same phase boundary, {ratio:.0f}× different protection", y=1.0)
    fig.savefig(os.path.join(OUT, "fig2_phase_diagrams.png"), dpi=150,
                bbox_inches="tight")
    save_numbers("fig2", dict(
        max_gap_Si_alpha_ueV=round(maxgaps[0], 2),
        max_gap_InSb_alpha_ueV=round(maxgaps[1], 2),
        ratio=round(maxgaps[1] / maxgaps[0], 1),
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 3
def _best_gap_vs_alpha(alphas, Delta_ueV, Bc_T, mu_max=145.0, nmu=16, nB=18,
                       suppress=False):
    """Max topological gap over mu in [0, mu_max], B in (0, Bc].
    suppress=True applies pair-breaking Delta(B) = Delta0 (1 - (B/Bc)^2)."""
    mus = np.linspace(0, mu_max, nmu)
    Bs = np.linspace(0.3, 0.98 * Bc_T, nB)
    best = np.zeros(len(alphas))
    for ia, al in enumerate(alphas):
        m = 0.0
        for B in Bs:
            D = Delta_ueV * (1 - (B / Bc_T)**2) if suppress else Delta_ueV
            if D <= 1.0:
                continue
            g = _bulk_gap_grid(mus, np.array([B]), D, al, M_SI, G_SI, nk=8001)[0]
            topo = EZ_J(G_SI, B)**2 > (D * UEV)**2 + (mus * UEV)**2
            gt = np.where(topo, g, 0.0)
            m = max(m, gt.max())
        best[ia] = m
    return best


def fig3():
    """The verdict plot: best achievable topological gap vs SOC strength,
    under a parent-superconductor critical-field constraint."""
    t0 = time.time()
    alphas = np.logspace(-5, np.log10(2.0), 70)
    curves = [
        (DELTA, 2.0, False, "Δ=50 µeV const, B ≤ 2 T (idealized)", "C0-"),
        (DELTA, 2.0, True, "Δ(B)=Δ₀[1−(B/B_c)²], Δ₀=50 µeV, B_c=2 T (Al film)", "C3-"),
        (100.0, 2.0, False, "Δ=100 µeV const, B ≤ 2 T", "C1--"),
        (DELTA, 4.0, True, "Δ(B), Δ₀=50 µeV, B_c=4 T (NbTiN-class)", "C2-."),
    ]
    results = {}
    fig, a = plt.subplots(figsize=(9.5, 6.5))
    for D, Bc, sup, label, style in curves:
        best = _best_gap_vs_alpha(alphas, D, Bc, suppress=sup)
        a.loglog(alphas, np.clip(best, 1e-3, None), style, label=label)
        results[label] = best
    # platform bands on the alpha axis
    bands = [(1e-5, 2e-3, "intrinsic Si\n(MOS/SiGe, measured)", "#d62728", 0.12),
             (1e-2, 0.1, "engineered SOC\n(micromagnets etc.)", "#9467bd", 0.10),
             (0.1, 1.0, "Ge/Si holes /\nInAs / InSb", "#2ca02c", 0.10)]
    for x0, x1, lab, c, al in bands:
        a.axvspan(x0, x1, color=c, alpha=al)
        a.text(np.sqrt(x0 * x1), 2.2e-3, lab, ha="center", fontsize=8, color=c)
    a.axhline(2.2, color="gray", ls=":", lw=1)
    a.text(1.3e-5, 2.5, "k$_B$·25 mK", fontsize=8, color="gray")
    a.axhline(20, color="k", ls=":", lw=1)
    a.text(1.3e-5, 23, "robust operation (~20 µeV)", fontsize=8)
    a.set_xlabel("Rashba SOC strength α (eV·Å)")
    a.set_ylabel("best achievable topological gap (µeV)")
    secy = a.secondary_yaxis("right", functions=(lambda e: e * UEV_TO_MK,
                                                 lambda T: T / UEV_TO_MK))
    secy.set_ylabel("equivalent temperature (mK)")
    a.set_title("Feasibility verdict: topological gap vs SOC, Si band parameters (g=2),\n"
                "optimized over µ and B within the parent superconductor's field limit")
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_gap_vs_alpha.png"), dpi=150)

    # key numbers
    base = results[curves[0][3]]
    supp = results[curves[1][3]]
    def gap_at(al):
        return float(np.interp(al, alphas, base))
    alpha_for_20 = first_crossing(alphas, base, 20.0)
    alpha_for_20_supp = first_crossing(alphas, supp, 20.0)
    # the Delta catch-22: max Delta that still allows topology at B<=2T
    Dmax = EZ_J(G_SI, 2.0) / UEV
    # optimal induced Delta at the engineered-SOC working point
    Dscan = np.linspace(10, 114, 27)
    bb = [_best_gap_vs_alpha(np.array([ALPHA_DEMO]), D, 2.0, suppress=True)[0]
          for D in Dscan]
    # high-resolution recompute of the two intrinsic-Si key numbers
    # (k-grid bias at small alpha; review minor item 2)
    def _refined(al):
        best = 0.0
        for B in np.linspace(0.9, 1.98, 16):
            for mu in np.linspace(0, 120, 16):
                if is_topological(mu, B, DELTA, G_SI):
                    best = max(best, bulk_gap_ueV(mu, B, DELTA, al, M_SI,
                                                  G_SI, nk=40001))
        return best
    save_numbers("fig3", dict(
        gap_at_intrinsic_Si_opt_1e3_ueV=round(_refined(1e-3), 4),
        gap_at_intrinsic_Si_typ_1e4_ueV=round(_refined(1e-4), 5),
        gap_at_engineered_005_ueV=round(gap_at(0.05), 2),
        alpha_needed_for_20ueV=round(alpha_for_20, 4),
        alpha_needed_for_20ueV_with_DeltaB=round(alpha_for_20_supp, 4),
        gap_at_1e3_with_DeltaB=round(float(np.interp(1e-3, alphas, supp)), 3),
        shortfall_vs_optimistic_intrinsic=round(alpha_for_20 / 1e-3, 0),
        shortfall_vs_typical_intrinsic=round(alpha_for_20 / 1e-4, 0),
        max_Delta_allowing_topology_at_2T_ueV=round(Dmax, 1),
        optimal_Delta_at_alpha005_ueV=float(Dscan[int(np.argmax(bb))]),
        optimal_Delta_gap_ueV=round(float(np.max(bb)), 3),
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 4
def fig4(rows=None):
    """Two-valley model: zero-mode splitting vs (mu, valley splitting)."""
    t0 = time.time()
    B = 1.5
    N, L = 200, 1e-6
    div = 10.0           # ueV inter-valley mixing
    mus = np.linspace(-150, 150, 31)
    Evs = np.linspace(0, 300, 31)
    ckpt = os.path.join(DATA, "fig4_scan.npz")
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        E0 = z["E0"]; done = int(z["done"])
    else:
        E0 = np.full((len(Evs), len(mus)), np.nan); done = 0
    budget = rows if rows else len(Evs)
    for i in range(done, min(done + budget, len(Evs))):
        for j, mu in enumerate(mus):
            rng = np.random.default_rng(7919 * i + j)
            H = build_wire_two_valley(N, L / N, mu, B, DELTA, ALPHA_DEMO,
                                      M_SI, G_SI, Ev_ueV=Evs[i],
                                      delta_iv_ueV=div, rng=rng)
            E, _ = solve_lowest(H, k=2)
            E0[i, j] = abs(E[0]) / UEV
        np.savez(ckpt, E0=E0, done=i + 1, mus=mus, Evs=Evs)
        print(f"fig4 row {i+1}/{len(Evs)}  t={time.time()-t0:.0f}s", flush=True)
    if int(np.load(ckpt)["done"]) < len(Evs):
        print("fig4: partial — rerun to continue")
        return

    EZ = EZ_J(G_SI, B) / UEV
    win = np.sqrt(EZ**2 - DELTA**2)              # single-valley mu window
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    a = ax[0]
    im = a.pcolormesh(mus, Evs, np.clip(E0, 1e-3, None),
                      norm=LogNorm(vmin=1e-3, vmax=30), cmap="magma_r",
                      shading="auto")
    fig.colorbar(im, ax=a, label="|E$_0$| (µeV), log scale")
    ev = np.linspace(0, 300, 200)
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            a.plot(s1 * win + s2 * ev / 2, ev, "c--", lw=1)
    a.set_xlabel("µ (µeV)"); a.set_ylabel("valley splitting E$_v$ (µeV)")
    a.set_xlim(mus[0], mus[-1]); a.set_ylim(0, 300)
    a.set_title("(a) zero-mode energy; dashed: analytic single-valley-\n"
                "topological (XOR) boundaries µ±E$_v$/2 = ±%.0f µeV" % win)

    a = ax[1]
    jc = np.argmin(np.abs(mus - 35.0))
    mu_cut = float(mus[jc])
    a.semilogy(Evs, np.clip(E0[:, jc], 1e-4, None), "o-")
    a.axvline(2 * (win - mu_cut), color="r", ls="--", lw=1,
              label=f"both→one valley topological: E$_v$={2*(win-mu_cut):.0f} µeV")
    a.axvline(2 * (win + mu_cut), color="purple", ls="--", lw=1,
              label=f"one→zero: E$_v$={2*(win+mu_cut):.0f} µeV")
    a.set_xlabel("valley splitting E$_v$ (µeV)"); a.set_ylabel("|E$_0$| (µeV)")
    a.set_title(f"(b) cut at µ = {mu_cut:.0f} µeV  (random-phase inter-valley\n"
                f"scattering δ$_{{iv}}$ = {div:.0f} µeV)")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.suptitle(f"Valley degeneracy: two-valley toy model, α={ALPHA_DEMO} eV·Å, "
                 f"B={B} T, Δ={DELTA:.0f} µeV, L={L*1e6:.0f} µm", y=1.0)
    fig.savefig(os.path.join(OUT, "fig4_valleys.png"), dpi=150,
                bbox_inches="tight")
    xor = (Evs > 2 * (win - mu_cut) + 15) & (Evs < 2 * (win + mu_cut) - 15)
    both = Evs < 2 * (win - mu_cut) - 15
    save_numbers("fig4", dict(
        single_valley_window_ueV=round(win, 1),
        Ev_threshold_at_mu_cut_ueV=round(2 * (win - mu_cut), 1),
        median_split_both_valleys_ueV=round(float(np.nanmedian(E0[both][:, jc])), 2),
        median_split_single_valley_ueV=round(float(np.nanmedian(E0[xor][:, jc])), 3),
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 5
FIG5_CASES = [
    dict(tag="intrinsic Si: α=0.001 eV·Å", alpha=1e-3, dx=5e-9, L=20e-6,
         Ws=[0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]),
    dict(tag="engineered Si: α=0.05 eV·Å", alpha=0.05, dx=2.5e-9, L=2e-6,
         Ws=[0.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]),
    dict(tag="strong SOC: α=0.15 eV·Å", alpha=0.15, dx=2.5e-9, L=6e-6,
         Ws=[0.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]),
]
FIG5_NREAL = 16


def fig5(chunk=None):
    """Disorder robustness, redone after review: per-case dx/L chosen so that
    k_so*dx < 0.1 (lattice convergence) and L >> coherence length where
    affordable; per-case clean baselines; first-crossing W_half with honest
    clamping. Disorder: onsite iid uniform [-W, W], correlation length = dx
    (quote alongside W; thresholds are lattice-convention dependent)."""
    t0 = time.time()
    B, mu = 1.5, 0.0
    params_sig = json.dumps([(c["alpha"], c["dx"], c["L"], c["Ws"]) for c in FIG5_CASES]) \
        + f"|B={B},mu={mu},D={DELTA},n={FIG5_NREAL}"
    nW = max(len(c["Ws"]) for c in FIG5_CASES)
    ckpt = os.path.join(DATA, "fig5_scan_v2.npz")
    fresh = True
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if str(z["sig"]) == params_sig:
            E0 = z["E0"]; E2 = z["E2"]; done = int(z["done"]); fresh = False
    if fresh:
        E0 = np.full((len(FIG5_CASES), nW, FIG5_NREAL), np.nan)
        E2 = np.full_like(E0, np.nan); done = 0
    cells = [(ic, iw) for ic, c in enumerate(FIG5_CASES) for iw in range(len(c["Ws"]))]
    budget = chunk if chunk else len(cells)
    for idx in range(done, min(done + budget, len(cells))):
        ic, iw = cells[idx]
        c = FIG5_CASES[ic]
        N = int(round(c["L"] / c["dx"]))
        W = c["Ws"][iw]
        nr = 1 if W == 0 else FIG5_NREAL
        for r in range(nr):
            rng = np.random.default_rng([91, ic, iw, r]) if W else None
            H = build_wire(N, c["dx"], mu, B, DELTA, c["alpha"], M_SI, G_SI,
                           disorder_ueV=W, rng=rng)
            e0, e2, _ = majorana_metrics(H, N)
            E0[ic, iw, r], E2[ic, iw, r] = e0, e2
        if W == 0:
            E0[ic, iw, :] = E0[ic, iw, 0]; E2[ic, iw, :] = E2[ic, iw, 0]
        np.savez(ckpt, E0=E0, E2=E2, done=idx + 1, sig=params_sig)
        print(f"fig5 {idx+1}/{len(cells)}  t={time.time()-t0:.0f}s", flush=True)
    if int(np.load(ckpt, allow_pickle=True)["done"]) < len(cells):
        print("fig5: partial — rerun to continue")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    keynum = {}
    for ic, c in enumerate(FIG5_CASES):
        Ws = np.array(c["Ws"]); nWc = len(Ws)
        med0 = np.median(E0[ic, :nWc], axis=1)
        med2 = np.median(E2[ic, :nWc], axis=1)
        q1, q3 = np.percentile(E0[ic, :nWc], [25, 75], axis=1)
        gap_clean = med2[0]
        Wp = np.clip(Ws, Ws[1] / 2, None)        # log-x placement for W=0
        ax[0].loglog(Wp, np.clip(med0, 1e-4, None), "o-", color=f"C{ic}",
                     label=f'{c["tag"]}, L={c["L"]*1e6:.0f} µm')
        ax[0].fill_between(Wp, np.clip(q1, 1e-4, None), np.clip(q3, 1e-4, None),
                           color=f"C{ic}", alpha=0.2)
        ax[1].semilogx(Wp, med2 / gap_clean, "o-", color=f"C{ic}",
                       label=f'{c["tag"]} (clean E$_2$={gap_clean:.1f} µeV)')
        Whalf = first_crossing(Ws, med2, gap_clean / 2, rising=False)
        bulk = bulk_gap_ueV(mu, B, DELTA, c["alpha"], M_SI, G_SI)
        keynum[f"case{ic}"] = dict(
            tag=c["tag"], clean_E0_ueV=round(float(med0[0]), 4),
            clean_E2_ueV=round(float(gap_clean), 2),
            bulk_gap_ueV=round(bulk, 2),
            W_half_ueV=(round(Whalf, 1) if Whalf is not None
                        else f">{Ws[-1]:.0f} (not reached)"),
            dx_nm=c["dx"] * 1e9, L_um=c["L"] * 1e6)
    ax[0].set_ylabel("median |E$_0$| (µeV)")
    ax[0].set_title("(a) zero-mode splitting vs disorder (IQR shaded)")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_ylabel("median E$_2$ / clean E$_2$")
    ax[1].set_title("(b) protecting gap vs disorder, per-case clean baseline")
    ax[1].set_ylim(0, 1.15)
    for a in ax:
        a.set_xlabel("onsite disorder amplitude W (µeV)  [iid, corr. length = dx]")
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.suptitle(f"Disorder robustness (µ=0, B={B} T, Δ={DELTA:.0f} µeV, "
                 f"{FIG5_NREAL} realizations; W=0 point plotted at half of first W)", y=1.0)
    fig.savefig(os.path.join(OUT, "fig5_disorder.png"), dpi=150,
                bbox_inches="tight")
    keynum["runtime_s"] = round(time.time() - t0, 1)
    save_numbers("fig5", keynum)


# ----------------------------------------------------------------- Figure 6
def fig6(chunk=None):
    """Physical valley model: inter-valley pairing with valley-orbit coupling.
    (a) uniform-phase equivalence + wedge survival under interface steps;
    (b) protection vs step density inside the single-valley wedge;
    (c) valley polarization (nu_z, TRS-breaking) as the true pair-breaker."""
    t0 = time.time()
    B, Delta, al = 1.5, DELTA, ALPHA_DEMO
    N, dx = 300, DX
    mu = 35.0
    win = np.sqrt((EZ_J(G_SI, B) / UEV)**2 - Delta**2)
    Evs = np.linspace(0, 300, 13)
    nseed = 8

    def solve_iv(muv, Ev, vo, **kw):
        H = build_wire_two_valley_iv(N, dx, muv, B, Delta, al, M_SI, G_SI,
                                     Ev_ueV=Ev, vo_profile_ueV=vo, **kw)
        E, _ = solve_lowest(H, k=6)
        Ea = np.sort(np.abs(E)) / UEV
        return Ea[0], Ea[2]

    # (a) Ev scan at mu=35
    a_unif, a_100, a_30 = [], [], []
    for Ev in Evs:
        a_unif.append(solve_iv(mu, 0, 0.5 * Ev * np.exp(1.1j) * np.ones(N))[0])
        for dest, Lst in ((a_100, 100e-9), (a_30, 30e-9)):
            vals = []
            for s in range(nseed):
                rng = np.random.default_rng([61, int(Ev), s, int(Lst * 1e9)])
                phi = step_phase_profile(N, dx, Lst, rng)
                vals.append(solve_iv(mu, 0, 0.5 * Ev * np.exp(1j * phi))[0])
            dest.append(vals)
    a_100 = np.array(a_100); a_30 = np.array(a_30)

    # (b) step-density scan inside the wedge (Ev=150), both physical SOC classes
    Lsteps = np.array([1000, 300, 100, 50, 30, 15]) * 1e-9
    modes = ("rashba", "dresselhaus")
    b_e0 = {m_: [] for m_ in modes}; b_e2 = {m_: [] for m_ in modes}
    for mode in modes:
        for Lst in Lsteps:
            v0s, v2s = [], []
            for s in range(nseed):
                rng = np.random.default_rng([62, int(Lst * 1e9), s, len(mode)])
                phi = step_phase_profile(N, dx, Lst, rng)
                e0, e2 = solve_iv(mu, 0, 75 * np.exp(1j * phi), soc_mode=mode)
                v0s.append(e0); v2s.append(e2)
            b_e0[mode].append(v0s); b_e2[mode].append(v2s)

    # L-convergence of the step-disorder gap (review M1)
    Lconv = {}
    for Nc in (300, 600, 1200):
        vals = []
        for s in range(12):
            rng = np.random.default_rng([63, Nc, s])
            phi = step_phase_profile(Nc, dx, 50e-9, rng)
            H = build_wire_two_valley_iv(Nc, dx, mu, B, Delta, al, M_SI, G_SI,
                                         vo_profile_ueV=75 * np.exp(1j * phi))
            E, _ = solve_lowest(H, k=6)
            vals.append(np.sort(np.abs(E))[2] / UEV)
        Lconv[f"L{Nc*dx*1e6:.1f}um"] = (round(float(np.median(vals)), 2),
                                        round(float(np.percentile(vals, 25)), 2),
                                        round(float(np.percentile(vals, 75)), 2))

    # (c) valley polarization at the wedge point (uniform VO)
    Epols = np.linspace(0, 150, 11)
    c_e0, c_e2 = [], []
    for ep in Epols:
        e0, e2 = solve_iv(mu, 150, None, valley_pol_ueV=ep)
        c_e0.append(e0); c_e2.append(e2)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    a = ax[0]
    a.semilogy(Evs, np.clip(a_unif, 1e-4, None), "k-o", ms=4,
               label="uniform VO phase (any φ): exact two-band picture")
    for arr, lab, c in ((a_100, "steps, L$_{step}$=100 nm", "C0"),
                        (a_30, "steps, L$_{step}$=30 nm", "C3")):
        med = np.median(arr, axis=1)
        q1, q3 = np.percentile(arr, [25, 75], axis=1)
        a.semilogy(Evs, np.clip(med, 1e-4, None), "o-", color=c, ms=4, label=lab)
        a.fill_between(Evs, np.clip(q1, 1e-4, None), np.clip(q3, 1e-4, None),
                       color=c, alpha=0.2)
    a.axvline(2 * (win - mu), color="r", ls="--", lw=1)
    a.axvline(2 * (win + mu), color="purple", ls="--", lw=1)
    a.set_xlabel("valley splitting E$_v$ (µeV)"); a.set_ylabel("|E$_0$| (µeV)")
    a.set_title(f"(a) inter-valley pairing, µ={mu:.0f} µeV\n"
                "dashed: two-band boundaries (both→one→zero)")
    a.legend(fontsize=7); a.grid(alpha=0.3, which="both")

    a = ax[1]
    for anti, c, lab in (("rashba", "C0", "valley-scalar SOC (Rashba-like)"),
                         ("dresselhaus", "C2", "phase-locked SOC (Dresselhaus-like)")):
        e0m = np.median(b_e0[anti], axis=1); e2m = np.median(b_e2[anti], axis=1)
        a.loglog(Lsteps * 1e9, np.clip(e0m, 1e-4, None), "o-", color=c,
                 label="|E$_0$|, " + lab)
        a.loglog(Lsteps * 1e9, e2m, "s--", color=c, alpha=0.6,
                 label="E$_2$, " + lab)
    a.set_xlabel("mean step spacing L$_{step}$ (nm)")
    a.set_ylabel("energy (µeV)")
    a.set_title(f"(b) wedge point (E$_v$=150, µ={mu:.0f}):\nprotection vs interface step density")
    a.legend(fontsize=7); a.grid(alpha=0.3, which="both")

    a = ax[2]
    a.semilogy(Epols, np.clip(c_e0, 1e-4, None), "o-", label="|E$_0$|")
    a.semilogy(Epols, c_e2, "s--", label="E$_2$")
    a.set_xlabel("valley polarization E$_{pol}$ (µeV)  [ν$_z$, TRS-breaking]")
    a.set_ylabel("energy (µeV)")
    a.set_title("(c) the true pair-breaker:\nvalley polarization vs inter-valley pairing")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    fig.suptitle(f"Physical valley channel — inter-valley singlet pairing, α={al} eV·Å, "
                 f"B={B} T, Δ={Delta:.0f} µeV, L={N*dx*1e6:.1f} µm, {nseed} seeds", y=1.02)
    fig.savefig(os.path.join(OUT, "fig6_intervalley_pairing.png"), dpi=150,
                bbox_inches="tight")

    both_mask = Evs < 2 * (win - mu) - 10
    xor_mask = (Evs > 2 * (win - mu) + 10) & (Evs < 2 * (win + mu) - 10)
    save_numbers("fig6", dict(
        median_split_bothvalleys_steps100_ueV=round(float(np.median(a_100[both_mask])), 3),
        median_split_singlevalley_steps100_ueV=round(float(np.median(a_100[xor_mask])), 3),
        median_split_singlevalley_steps30_ueV=round(float(np.median(a_30[xor_mask])), 3),
        wedge_E0_at_steps50_ueV=round(float(np.median(b_e0["rashba"][3])), 3),
        wedge_E2_at_steps50_ueV=round(float(np.median(b_e2["rashba"][3])), 2),
        wedge_E2_at_steps50_dresselhaus_ueV=round(float(np.median(b_e2["dresselhaus"][3])), 2),
        wedge_E2_clean_ueV=round(float(np.median(b_e2["rashba"][0])), 2),
        wedge_E2_Lconvergence_median_q1_q3=Lconv,
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 7
def fig7(chunk=None):
    """Multi-subband 2D strip: do the 1D conclusions survive several transverse
    subbands? B is along the wire and in-plane (no flux through plaquettes)."""
    t0 = time.time()
    B, Delta, al = 1.5, DELTA, ALPHA_DEMO
    Nx, Ny, dx, dy = 300, 5, DX, 10e-9
    bots = subband_bottoms_ueV(Ny, dy, M_SI)
    win = np.sqrt((EZ_J(G_SI, B) / UEV)**2 - Delta**2)
    windows = [np.linspace(b - 150, b + 150, 31) for b in bots[:2]]
    res = []
    for mus in windows:
        e0s, e2s = [], []
        for mu in mus:
            H = build_wire_2d(Nx, Ny, dx, dy, mu, B, Delta, al, M_SI, G_SI)
            E, _ = solve_lowest(H, k=6)
            Ea = np.sort(np.abs(E)) / UEV
            e0s.append(Ea[0]); e2s.append(Ea[2])
        res.append((mus, np.array(e0s), np.array(e2s)))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for i, (a, (mus, e0s, e2s)) in enumerate(zip(ax, res)):
        a.axvspan(bots[i] - win, bots[i] + win, color="C2", alpha=0.15,
                  label="1D prediction: |µ−E$_n$| < %.0f µeV" % win)
        a.semilogy(mus, np.clip(e0s, 1e-4, None), "o-", ms=4, label="|E$_0$|")
        a.semilogy(mus, e2s, "s--", ms=4, label="E$_2$")
        a.axvline(bots[i], color="k", ls=":", lw=1)
        a.set_xlabel("µ (µeV)")
        a.set_title(f"subband {i+1} (bottom E_{i+1} = {bots[i]:.0f} µeV)")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    ax[0].set_ylabel("energy (µeV)")
    fig.suptitle(f"Multi-subband strip (W={(Ny+1)*dy*1e9:.0f} nm hard wall, Ny={Ny}; "
                 f"first two subbands scanned): topological domes at each subband bottom — α={al} eV·Å, B={B} T, "
                 f"Δ={Delta:.0f} µeV, L={Nx*dx*1e6:.1f} µm, B in-plane ∥ wire", y=1.02)
    fig.savefig(os.path.join(OUT, "fig7_multisubband.png"), dpi=150,
                bbox_inches="tight")

    k1 = {}
    for i, (mus, e0s, e2s) in enumerate(res):
        inwin = np.abs(mus - bots[i]) < win - 10
        k1[f"subband{i+1}_max_E2_inwindow_ueV"] = round(float(e2s[inwin].max()), 2)
        k1[f"subband{i+1}_median_E0_inwindow_ueV"] = round(float(np.median(e0s[inwin])), 4)
    k1["bottoms_ueV"] = [round(b, 0) for b in bots[:3]]
    k1["window_halfwidth_ueV"] = round(win, 1)
    k1["gap_1D_same_params_ueV"] = round(bulk_gap_ueV(0, B, Delta, al, M_SI, G_SI), 2)
    k1["runtime_s"] = round(time.time() - t0, 1)
    save_numbers("fig7", k1)



# ----------------------------------------------------------------- Figure 8
# The all-silicon stack: p-type Si hole channel + superconducting Si:B parent
M_HOLE = 0.25          # hole effective mass along channel (representative)
SIB_DELTA0 = 91.0      # ueV  (Tc = 0.6 K, BCS 1.764 kB Tc)
SIB_BP = SIB_DELTA0 / (np.sqrt(2) * MU_B_EV * 1e6)   # Pauli limit ~1.11 T


SIB_BC2_MEAS = 0.4     # T — measured-class critical field of Si:B
                       # (reports range ~0.1-0.4 T, orbital-limited, dirty;
                       # Bustarret 2006; PRB 81, 020501(R) 2010). The
                       # Pauli-limited scenario below is a HYPOTHESIS requiring
                       # <20 nm films + parallel alignment, never demonstrated.


def _parent_gap_ueV(B, parent):
    """Parent spectral gap vs field (caricatures; see RESULTS caveats).
    SiB_pauli: hypothetical thin-film Pauli-limited Si:B, Delta0 - muB*B.
    SiB_meas:  measured-class orbital-limited Si:B, GL with Bc2 = 0.4 T.
    Al:        GL orbital caricature, Delta0 = 200 ueV, Bc = 2 T."""
    if parent in ("SiB", "SiB_pauli"):
        return max(SIB_DELTA0 - MU_B_EV * 1e6 * B, 0.0)
    if parent == "SiB_meas":
        return SIB_DELTA0 * max(1 - (B / SIB_BC2_MEAS)**2, 0.0)
    if parent == "Al":
        return 200.0 * max(1 - (B / 2.0)**2, 0.0)
    raise ValueError(parent)


def _best_gap_hole(alpha, g, parent, m_rel=M_HOLE, nB=7, nD=7, nmu=14,
                   nk=2501, mu_max=160.0, renormalize=False):
    """Max topological gap over (B, Delta_ind <= parent gap, mu).
    renormalize=True applies the standard tunneling-model metallization:
    quasiparticle weight Z = 1 - Dind/Dp, alpha -> Z alpha,
    g -> Z g + (1-Z) g_parent (g_parent = 2). Returns (gap, B*, Dind*, mu*)."""
    if parent in ("SiB", "SiB_pauli"):
        Bmax = 0.9 * SIB_BP
    elif parent == "SiB_meas":
        Bmax = 0.95 * SIB_BC2_MEAS
    else:
        Bmax = 1.96
    best, arg = 0.0, (np.nan,) * 3
    mus = np.linspace(0, mu_max, nmu)
    for B in np.linspace(0.1, Bmax, nB):
        Dp = _parent_gap_ueV(B, parent)
        if Dp < 8:
            continue
        for Dind in np.linspace(8, Dp, nD):
            if renormalize:
                Z = 1.0 - Dind / Dp
                a_eff = Z * alpha
                g_eff = Z * g + (1 - Z) * 2.0
                if a_eff < 1e-4:
                    continue
            else:
                a_eff, g_eff = alpha, g
            gaps = _bulk_gap_grid(mus, np.array([B]), Dind, a_eff, m_rel,
                                  g_eff, nk=nk)[0]
            topo = EZ_J(g_eff, B)**2 > (Dind * UEV)**2 + (mus * UEV)**2
            gaps = np.where(topo, gaps, 0.0)
            j = int(np.argmax(gaps))
            if gaps[j] > best:
                best, arg = float(gaps[j]), (B, Dind, float(mus[j]))
    return best, arg


def _hole_wire_check():
    """Finite-wire + disorder check at the fig8 center operating point.
    (In-repo generator for the numbers quoted in RESULTS F8; review round 3.)"""
    mh, gh, al, B, Dind, mu = M_HOLE, 2.2, 0.06, 1.0, 33.0, 0.0
    N, dx = 1200, 2.5e-9
    H = build_wire(N, dx, mu, B, Dind, al, mh, gh)
    e0, e2, ew = majorana_metrics(H, N)
    out = dict(clean=dict(E0=round(e0, 4), E2=round(e2, 2),
                          end_weight=round(ew, 3),
                          bulk=round(bulk_gap_ueV(mu, B, Dind, al, mh, gh), 2)))
    meds = {}
    for W in (50, 100, 200, 400, 800):
        v = []
        for s in range(10):
            rng = np.random.default_rng([81, W, s])
            Hd = build_wire(N, dx, mu, B, Dind, al, mh, gh,
                            disorder_ueV=W, rng=rng)
            v.append(majorana_metrics(Hd, N)[1])
        meds[str(W)] = round(float(np.median(v)), 2)
    out["disorder_W_medE2"] = meds
    return out


def fig8(chunk=None):
    """All-silicon Majorana stack: hole channel + Si:B parent.
    (a) gap map over the measured FinFET-hole (alpha, g) box;
    (b) the transparency design rule (gap vs induced Delta);
    (c) platform comparison."""
    t0 = time.time()
    alphas = np.logspace(np.log10(0.01), np.log10(0.3), 20)
    gs = np.linspace(1.2, 4.0, 20)
    ckpt = os.path.join(DATA, "fig8_scan.npz")
    sig = f"a{alphas[0]:.3f}-{alphas[-1]:.3f}x{len(alphas)},g{gs[0]}-{gs[-1]}x{len(gs)},SiB{SIB_DELTA0}"
    fresh = True
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if str(z["sig"]) == sig:
            gap = z["gap"]; done = int(z["done"]); fresh = False
    if fresh:
        gap = np.full((len(gs), len(alphas)), np.nan); done = 0
    budget = chunk if chunk else len(gs)
    for i in range(done, min(done + budget, len(gs))):
        for j, al in enumerate(alphas):
            gap[i, j] = _best_gap_hole(al, gs[i], "SiB")[0]
        np.savez(ckpt, gap=gap, done=i + 1, sig=sig)
        print(f"fig8 row {i+1}/{len(gs)}  t={time.time()-t0:.0f}s", flush=True)
    if int(np.load(ckpt, allow_pickle=True)["done"]) < len(gs):
        print("fig8: partial — rerun to continue"); return

    # (b) transparency design rule at alpha=0.06
    Dinds = np.linspace(10, SIB_DELTA0 - 2, 24)
    curves = {}
    for g in (1.8, 2.2, 2.8, 3.4):
        ys = []
        for Dind in Dinds:
            Bmax_c = min(0.9 * SIB_BP, (SIB_DELTA0 - Dind) / (MU_B_EV * 1e6))
            best = 0.0
            if Bmax_c > 0.15:
                for B in np.linspace(0.15, Bmax_c, 8):
                    mus = np.linspace(0, 160, 14)
                    gg = _bulk_gap_grid(mus, np.array([B]), Dind, 0.06,
                                        M_HOLE, g, nk=2501)[0]
                    topo = EZ_J(g, B)**2 > (Dind*UEV)**2 + (mus*UEV)**2
                    best = max(best, float(np.where(topo, gg, 0.0).max()))
            ys.append(best)
        curves[g] = np.array(ys)

    # (c) platform comparison
    comp = {
        "Si e⁻ intrinsic\n(α=10⁻³)": 1.55,
        "Si e⁻ engineered\n(α=0.05, Δ(B) Al)": _best_gap_vs_alpha(
            np.array([0.05]), DELTA, 2.0, suppress=True)[0],
        "Si holes + Al\n(α=0.06, g=2.2)": _best_gap_hole(0.06, 2.2, "Al")[0],
        "Si holes + Si:B\nPauli hyp. (α=0.06, g=2.2)": _best_gap_hole(0.06, 2.2, "SiB_pauli")[0],
        "Si holes + Si:B\nMEASURED B$_{c2}$=0.4T": _best_gap_hole(0.06, 2.2, "SiB_meas")[0],
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    im = a.pcolormesh(alphas, gs, np.clip(gap, 0.5, None),
                      norm=LogNorm(vmin=0.5, vmax=60), cmap="viridis",
                      shading="auto")
    fig.colorbar(im, ax=a, label="max topological gap (µeV)")
    cs = a.contour(alphas, gs, gap, levels=[10, 20, 30], colors="w",
                   linewidths=1)
    a.clabel(cs, fmt="%.0f µeV", fontsize=7)
    a.add_patch(plt.Rectangle((0.03, 1.5), 0.12, 1.5, fill=False,
                              edgecolor="r", lw=2))
    a.text(0.062, 1.62, "measured Si FinFET\nhole range", color="r",
           fontsize=8, ha="center")
    a.set_xscale("log"); a.set_xlabel("hole SOC α (eV·Å)")
    a.set_ylabel("hole g-factor (field direction dependent)")
    a.set_title("(a) Si:B parent — HYPOTHETICAL Pauli-limited thin film\n"
                "optimized over B ≤ 0.9B_P, Δ_ind ≤ Δ_parent(B), µ")

    a = ax[1]
    for g, ys in curves.items():
        a.plot(Dinds, ys, "-o", ms=3, label=f"g = {g}")
    a.set_xlabel("induced gap Δ_ind (µeV)")
    a.set_ylabel("max topological gap (µeV)")
    a.set_title("(b) transparency design rule, α=0.06 eV·Å:\n"
                "interior optimum — do not maximize coupling")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2]
    names = list(comp); vals = [float(comp[n]) for n in names]
    bars = a.bar(range(len(names)), vals,
                 color=["C3", "C1", "C0", "C2", "C2"])
    a.bar_label(bars, fmt="%.1f", fontsize=8)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, fontsize=7)
    a.axhline(20, color="k", ls=":", lw=1)
    a.text(-0.4, 21, "robust operation", fontsize=7)
    a.set_ylabel("max topological gap (µeV)")
    a.set_title("(c) platform comparison (same machinery)")
    fig.suptitle("The all-silicon stack: Si hole channel + superconducting Si:B "
                 f"(T_c=0.6 K, B_P={SIB_BP:.2f} T), m*={M_HOLE} m_e — no valleys, "
                 "no micromagnets, CMOS-compatible", y=1.03)
    fig.savefig(os.path.join(OUT, "fig8_allsilicon_holes.png"), dpi=150,
                bbox_inches="tight")

    # box statistics + operating points + key numbers
    abox = (alphas >= 0.03) & (alphas <= 0.15)
    gbox = (gs >= 1.5) & (gs <= 3.0)
    box = gap[np.ix_(gbox, abox)]
    # refined evaluation at the literal box corner (grid never samples it;
    # review round 3) and finite-wire + disorder check at the center point
    g_corner = _best_gap_hole(0.03, 1.5, "SiB", nB=15, nD=15, nmu=29,
                              nk=12001)[0]
    hw = _hole_wire_check()
    g_cons, arg_cons = _best_gap_hole(0.03, 1.8, "SiB")
    g_cent, arg_cent = _best_gap_hole(0.06, 2.2, "SiB")
    g_fav, arg_fav = _best_gap_hole(0.15, 3.0, "SiB")
    save_numbers("fig8", dict(
        box_median_gap_ueV=round(float(np.median(box)), 2),
        box_frac_above_20ueV_grid=round(float((box >= 20).mean()), 2),
        box_corner_a003_g15_refined_ueV=round(g_corner, 1),
        hole_wire_finite_check=hw,
        box_frac_above_10ueV=round(float((box >= 10).mean()), 2),
        conservative_a003_g18=dict(gap=round(g_cons, 1),
                                   B_T=round(arg_cons[0], 2),
                                   Dind_ueV=round(arg_cons[1], 0),
                                   mu_ueV=round(arg_cons[2], 0)),
        center_a006_g22=dict(gap=round(g_cent, 1), B_T=round(arg_cent[0], 2),
                             Dind_ueV=round(arg_cent[1], 0),
                             mu_ueV=round(arg_cent[2], 0)),
        favorable_a015_g30=dict(gap=round(g_fav, 1), B_T=round(arg_fav[0], 2),
                                Dind_ueV=round(arg_fav[1], 0)),
        g2_works=dict(gap=round(_best_gap_hole(0.06, 2.0, "SiB")[0], 1)),
        center_SiB_measured_Bc04=round(_best_gap_hole(0.06, 2.2, "SiB_meas")[0], 1),
        center_SiB_measured_args=[round(v, 2) for v in
                                  _best_gap_hole(0.06, 2.2, "SiB_meas")[1]],
        center_SiB_pauli_renormalized=round(
            _best_gap_hole(0.06, 2.2, "SiB_pauli", renormalize=True)[0], 1),
        center_SiB_meas_renormalized=round(
            _best_gap_hole(0.06, 2.2, "SiB_meas", renormalize=True)[0], 1),
        center_Al_renormalized=round(
            _best_gap_hole(0.06, 2.2, "Al", renormalize=True)[0], 1),
        conservative_SiB_meas_renorm=round(
            _best_gap_hole(0.03, 1.8, "SiB_meas", renormalize=True)[0], 1),
        comparison={k.replace("\n", " "): round(float(v), 2)
                    for k, v in comp.items()},
        BP_T=round(SIB_BP, 3),
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 9
def _make_vo_profile(N, dx, mean_s, kind, phase_mode, rng, lam=75.0,
                     jitter_frac=0.0, dphi=0.85 * np.pi):
    """Valley-orbit profile with controlled step statistics.
    kind: 'poisson' | 'periodic' | 'jitter' (periodic + gaussian jitter).
    phase_mode: 'random' (uniform redraw at each step) | 'fixed'
    (deterministic +dphi per step: same-sign staircase = vicinal miscut) |
    'signrand' (+-dphi, random sign: same jump magnitude, zero net winding —
    the control discriminating winding from per-step junction physics)."""
    L = N * dx
    if kind == "poisson":
        pos, x = [], 0.0
        while True:
            x += rng.exponential(mean_s)
            if x >= L:
                break
            pos.append(x)
    else:
        pos = np.arange(mean_s, L, mean_s)
        if kind == "jitter":
            pos = pos + rng.normal(0, jitter_frac * mean_s, len(pos))
            pos = np.sort(pos[(pos > 0) & (pos < L)])
    phi = np.zeros(N)
    cur = 0.0
    pidx = 0
    pos = list(pos)
    for n in range(N):
        xn = n * dx
        while pidx < len(pos) and pos[pidx] <= xn:
            if phase_mode == "random":
                cur = rng.uniform(0, 2 * np.pi)
            elif phase_mode == "fixed":
                cur = cur + dphi
            elif phase_mode == "signrand":
                cur = cur + dphi * (1 if rng.random() < 0.5 else -1)
            else:
                raise ValueError(phase_mode)
            pidx += 1
        phi[n] = cur
    return lam * np.exp(1j * phi)


def fig9(chunk=None):
    """Step physics, reframed after review round 3: TWO damage channels.
    (a) scenario comparison incl. the sign-randomization control: per-step
        junction physics (jump size x density) dominates for staircases;
        net winding adds little at realistic step densities.
    (b) the per-step mechanism: a single valley-orbit phase step is a
        Josephson-like junction binding a subgap state; the Si single-step
        jump 2*k0*(a/4) = 0.85*pi is accidentally near pi (where a Kitaev
        domain-wall zero mode forms).
    Also generates (key numbers): the smooth linear-ramp scan (the second,
    subdominant channel; suppression set by the SOC-boost term closing the
    k=0 branch, E_v-protected — NOT naive Fulde-Ferrell 2*Delta/(hbar v_F))."""
    t0 = time.time()
    B, Delta, al, mu = 1.5, DELTA, ALPHA_DEMO, 35.0
    N, dx = 500, DX            # L = 2.5 um
    nseed = 14
    spacings = np.array([300, 150, 100, 75, 50, 35]) * 1e-9
    scenarios = [
        ("poisson", "random", "C3", "Poisson steps, random phase (rough, zero winding)"),
        ("poisson", "fixed", "C0", "Poisson steps, same-sign Δφ (vicinal miscut)"),
        ("periodic", "fixed", "C1", "periodic steps, same-sign Δφ (ideal staircase)"),
        ("poisson", "signrand", "C2", "CONTROL: ±Δφ random sign (zero net winding)"),
    ]

    def solve(vo):
        H = build_wire_two_valley_iv(N, dx, mu, B, Delta, al, M_SI, G_SI,
                                     vo_profile_ueV=vo)
        E, _ = solve_lowest(H, k=6)
        Ea = np.sort(np.abs(E)) / UEV
        return Ea[0], Ea[2]

    ckpt = os.path.join(DATA, "fig9_scan.npz")
    sig = f"N{N},dx{dx},ns{nseed},sc{len(scenarios)},v2"
    fresh = True
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if str(z["sig"]) == sig:
            E2 = z["E2"]; done = int(z["done"]); fresh = False
    if fresh:
        E2 = np.full((len(scenarios), len(spacings), nseed), np.nan); done = 0
    budget = chunk if chunk else len(scenarios)
    for isc in range(done, min(done + budget, len(scenarios))):
        kind, pmode, _, _ = scenarios[isc]
        for js, s_mean in enumerate(spacings):
            for s in range(nseed):
                rng = np.random.default_rng([71, isc, int(s_mean * 1e9), s])
                vo = _make_vo_profile(N, dx, s_mean, kind, pmode, rng)
                E2[isc, js, s] = solve(vo)[1]
        np.savez(ckpt, E2=E2, done=isc + 1, sig=sig)
        print(f"fig9 scenario {isc+1}/{len(scenarios)}  t={time.time()-t0:.0f}s",
              flush=True)
    if int(np.load(ckpt, allow_pickle=True)["done"]) < len(scenarios):
        print("fig9: partial — rerun to continue"); return

    # (b) single-step junction: one phase step at the wire center
    dphis = np.array([0, 0.1, 0.25, 0.4, 0.5, 0.65, 0.85, 0.95, 1.0]) * np.pi
    step_E2 = []
    half = N // 2
    for dphi in dphis:
        vo = 75.0 * np.exp(1j * np.where(np.arange(N) < half, 0.0, dphi))
        step_E2.append(solve(vo)[1])

    # smooth linear-ramp channel (in-code generator; review round 3)
    x = np.arange(N) * dx
    ramp = {}
    for q in [0, 2e6, 4e6, 8e6, 1.6e7, 5.3e7]:
        e0, e2 = solve(75.0 * np.exp(1j * q * x))
        ramp[f"{q:.1e}"] = (round(e0, 3), round(e2, 2))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a = ax[0]
    for isc, (kind, pmode, c, lab) in enumerate(scenarios):
        m2 = np.median(E2[isc], axis=1)
        q1 = np.percentile(E2[isc], 25, axis=1)
        q3 = np.percentile(E2[isc], 75, axis=1)
        a.semilogy(spacings * 1e9, np.clip(m2, 1e-3, None), "o-", color=c,
                   ms=4, label=lab)
        a.fill_between(spacings * 1e9, np.clip(q1, 1e-3, None),
                       np.clip(q3, 1e-3, None), color=c, alpha=0.15)
    a.set_xlabel("mean step spacing (nm)")
    a.set_ylabel("median E$_2$ (µeV)  [spectral proxy, L-dependent bound]")
    a.set_title("(a) control test: same-sign vs random-sign steps are close —\n"
                "per-step junction physics, not net winding, dominates")
    a.legend(fontsize=7); a.grid(alpha=0.3, which="both")
    a = ax[1]
    a.plot(dphis / np.pi, step_E2, "o-", color="C0")
    a.axvline(0.85, color="r", ls="--", lw=1,
              label="Si single-atomic step: Δφ = 2k₀(a/4) ≈ 0.85π")
    a.set_xlabel("phase jump Δφ / π at a single step")
    a.set_ylabel("lowest excitation above MZMs (µeV)")
    a.set_title("(b) one step = one junction: bound state deepens toward\n"
                "Δφ = π (Kitaev domain wall)")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"Step physics, two channels — wedge point E$_v$=150 µeV, "
                 f"µ={mu:.0f} µeV, α={al} eV·Å, L={N*DX*1e6:.1f} µm, "
                 f"{nseed} seeds", y=1.02)
    fig.savefig(os.path.join(OUT, "fig9_step_statistics.png"), dpi=150,
                bbox_inches="tight")

    j50 = int(np.argmin(np.abs(spacings - 50e-9)))
    med50 = {scenarios[i][3]: round(float(np.median(E2[i, j50])), 2)
             for i in range(len(scenarios))}
    save_numbers("fig9", dict(
        median_E2_at_50nm_by_scenario_ueV=med50,
        single_step_E2_vs_dphi_over_pi={f"{d/np.pi:.2f}": round(float(v), 2)
                                        for d, v in zip(dphis, step_E2)},
        linear_winding_check_E0_E2=ramp,
        note="winding-as-dominant-mechanism claim retired after round-3 "
             "control test; see RESULTS F9",
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 10
def fig10(chunk=None):
    """Luttinger-Kohn justification (and erosion) of the hole parameter box:
    alpha(E), g-tensor(E), m*(E) from the 4-band fin model, and the gap
    accessible along the PHYSICAL (alpha, g, m*) operating curve."""
    import lk_holes
    t0 = time.time()
    Ezs = np.array([2, 5, 10, 15, 20, 30, 40, 50]) * 1e6
    ckpt = os.path.join(DATA, "lk_table.npz")
    sig = "Wy10Wz12_ny11nz13_v1"
    fresh = True
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if str(z["sig"]) == sig:
            tab = z["tab"]; fresh = False
    if fresh:
        rows = []
        for Ez in Ezs:
            m, al, gx, gy, gz, nso = lk_holes.extract(Ez)
            rows.append([Ez, m, al, gx, gy, gz] + list(nso))
            print(f"LK Ez={Ez/1e6:.0f} MV/m done t={time.time()-t0:.0f}s",
                  flush=True)
        tab = np.array(rows)
        np.savez(ckpt, tab=tab, sig=sig)
    Ez, mst, al, gx, gy, gz = (tab[:, i] for i in range(6))

    # accessible gap along the LK operating curve (mu=0-opt inside helper)
    curves = {}
    for label, parent, guse in [
        ("Si:B thick, B∥ẑ (uses g_z)", "SiB_meas", gz),
        ("Si:B Pauli hyp., B∥x̂ (uses g_x)", "SiB_pauli", gx),
        ("Al film, B∥x̂ (uses g_x)", "Al", gx),
    ]:
        gaps = []
        for i in range(len(Ez)):
            gaps.append(_best_gap_hole(max(al[i], 1e-4), abs(guse[i]), parent,
                                       m_rel=mst[i])[0])
        curves[label] = np.array(gaps)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    a = ax[0]
    a.plot(Ez / 1e6, al, "C0-o", ms=4)
    a.set_xlabel("vertical gate field E$_z$ (MV/m)")
    a.set_ylabel("direct-Rashba α (eV·Å)", color="C0")
    a2 = a.twinx()
    for gv, c, lab in ((gx, "C3", "g$_x$ (wire axis)"),
                       (gy, "C2", "g$_y$ (∥ SOC axis — unusable)"),
                       (gz, "C1", "g$_z$ (out-of-plane)")):
        a2.plot(Ez / 1e6, np.abs(gv), c + "--s", ms=3, label=lab)
    a2.set_ylabel("|g| components")
    a2.legend(fontsize=7, loc="center right")
    a.set_title("(a) the α–g covariation: the field that creates α\n"
                "suppresses the wire-axis g")
    a.grid(alpha=0.3)

    a = ax[1]
    lso = HBAR**2 / (mst * ME * np.clip(al, 1e-4, None) * QE * 1e-10) * 1e9
    a.plot(Ez / 1e6, lso, "C0-o", ms=4, label="l$_{so}$ (LK)")
    a.axhspan(20, 60, color="C2", alpha=0.15,
              label="measured FinFET range\n(Camenzind 2022)")
    a.set_xlabel("E$_z$ (MV/m)"); a.set_ylabel("spin-orbit length l$_{so}$ (nm)")
    a.set_ylim(0, 120); a.legend(fontsize=8)
    a.set_title("(b) validation: LK l$_{so}$ lands in the\nmeasured window")
    a.grid(alpha=0.3)

    a = ax[2]
    for (lab, ys), c in zip(curves.items(), ("C2", "C1", "C0")):
        a.plot(Ez / 1e6, ys, c + "-o", ms=4, label=lab)
    a.axhline(20, color="k", ls=":", lw=1)
    a.set_xlabel("E$_z$ (MV/m)")
    a.set_ylabel("max topological gap (µeV)")
    a.set_title("(c) gap along the PHYSICAL operating curve\n"
                "(α, g, m* co-varying), not the free box")
    a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle("Luttinger–Kohn fin model (10×12 nm hard wall, Si γ's, κ=−0.42): "
                 "the parameter box, constrained", y=1.03)
    fig.savefig(os.path.join(OUT, "fig10_lk_constraint.png"), dpi=150,
                bbox_inches="tight")

    best = {lab: (round(float(ys.max()), 1),
                  round(float(Ez[int(np.argmax(ys))] / 1e6), 0))
            for lab, ys in curves.items()}
    i10 = int(np.argmin(np.abs(Ez - 1e7)))
    save_numbers("fig10", dict(
        alpha_range_eVA=[round(float(al.min()), 3), round(float(al.max()), 3)],
        mstar_range=[round(float(mst.min()), 2), round(float(mst.max()), 2)],
        gx_range=[round(float(np.abs(gx).min()), 2),
                  round(float(np.abs(gx).max()), 2)],
        gz_range=[round(float(np.abs(gz).min()), 2),
                  round(float(np.abs(gz).max()), 2)],
        lso_range_nm=[round(float(lso.min()), 0), round(float(lso.max()), 0)],
        nso_at_10MVm=[round(float(v), 2) for v in tab[i10, 6:9]],
        best_gap_by_scenario=best,
        runtime_s=round(time.time() - t0, 1)))


# ----------------------------------------------------------------- Figure 11
def _tilted_gap_ueV(alpha_eVA, m_rel, bperp_ueV, bpar_ueV, Dind_ueV, nk=500):
    """Bulk gap (mu=0) with Zeeman split into components perp/parallel to the
    SOC axis, via batched 4x4 diagonalization (no closed form with bpar)."""
    m = m_rel * ME
    aSI = alpha_eVA * 1e-10 * QE
    bx, bz = bperp_ueV * UEV, bpar_ueV * UEV
    D = Dind_ueV * UEV
    btot = np.hypot(bx, bz)
    # mu=0 topological criterion uses the component PERPENDICULAR to the SOC
    # axis only (a parallel-only Zeeman is the trivial shifted-BCS situation);
    # the parallel component degrades the gap, captured numerically below.
    if abs(bx) <= D:
        return 0.0
    kF = np.sqrt(2 * m * (btot + D)) / HBAR
    kso = m * aSI / HBAR**2
    k = np.linspace(0, 4 * (kF + kso) + 2e7, nk)
    s0_ = np.eye(2); sx_ = np.array([[0, 1], [1, 0]]); sz_ = np.diag([1., -1.])
    sy_ = np.array([[0, -1j], [1j, 0]])
    xi = HBAR**2 * k**2 / (2 * m)
    Hk = np.zeros((nk, 4, 4), dtype=complex)
    hup = (xi[:, None, None] * s0_ + (aSI * k)[:, None, None] * sz_
           + bx * sx_ + bz * sz_)
    hdn = (-xi[:, None, None] * s0_ + (aSI * k)[:, None, None] * sz_
           - bx * sx_ + bz * sz_)
    Hk[:, :2, :2] = hup
    Hk[:, 2:, 2:] = hdn
    Hk[:, :2, 2:] = D * (1j * sy_)
    Hk[:, 2:, :2] = (D * (1j * sy_)).conj().T
    ev = np.linalg.eigvalsh(Hk)
    return float(np.abs(ev).min() / UEV)


def fig11(chunk=None):
    """Field-orientation maps: the binding device-design constraint.
    Gap vs field direction for LK-computed and empirical (Geyer-class)
    g-tensors/SOC axes, for thick measured-Si:B and Al-film parents."""
    t0 = time.time()
    z = np.load(os.path.join(DATA, "lk_table.npz"), allow_pickle=True)
    tab = z["tab"]
    i10 = int(np.argmin(np.abs(tab[:, 0] - 1e7)))
    lk = dict(alpha=float(tab[i10, 2]), m=float(tab[i10, 1]),
              gten=np.abs(tab[i10, 3:6]), nso=np.array([0.0, 1.0, 0.0]))
    emp = dict(alpha=0.06, m=0.25, gten=np.array([2.1, 2.35, 2.7]),
               nso=np.array([0.0, 0.41, 0.91]) / np.linalg.norm([0, .41, .91]))
    parents = [("SiB_meas thick (isotropic B$_{c2}$=0.4 T)", "SiB_meas",
                dict(iso=True)),
               ("Al film (B$_{c∥}$=2 T, B$_{c⊥}$=0.1 T)", "Al",
                dict(iso=False, bcperp=0.1))]
    thetas = np.linspace(0, 90, 8)         # 0 = in-plane, 90 = out-of-plane
    phis = np.linspace(0, 180, 13)         # 0 = along wire (x)
    panels = []
    for pset_name, p in (("LK tensor", lk), ("empirical (Geyer-class)", emp)):
        for plab, parent, pc in parents:
            gapmap = np.zeros((len(thetas), len(phis)))
            for it, th in enumerate(np.deg2rad(thetas)):
                for ip, ph in enumerate(np.deg2rad(phis)):
                    n = np.array([np.cos(th) * np.cos(ph),
                                  np.cos(th) * np.sin(ph), np.sin(th)])
                    best = 0.0
                    Bmax = (0.95 * SIB_BC2_MEAS if parent == "SiB_meas"
                            else 1.96)
                    for B in np.linspace(0.1, Bmax, 6):
                        if parent == "Al" and not pc.get("iso", True):
                            supp = (1 - (B * abs(n[2]) / pc["bcperp"])**2
                                    - (B * np.hypot(n[0], n[1]) / 2.0)**2)
                            Dp = 200.0 * max(supp, 0.0)
                        else:
                            Dp = _parent_gap_ueV(B, parent)
                        if Dp < 8:
                            continue
                        bvec = 0.5 * MU_B_EV * 1e6 * B * p["gten"] * n
                        bpar = float(bvec @ p["nso"])
                        bperp = float(np.sqrt(max((bvec @ bvec)
                                                  - bpar**2, 0.0)))
                        for Dind in np.linspace(8, Dp, 5):
                            gp = _tilted_gap_ueV(p["alpha"], p["m"], bperp,
                                                 bpar, Dind)
                            best = max(best, gp)
                    gapmap[it, ip] = best
            panels.append((f"{pset_name} — {plab}", gapmap))
            print(f"fig11 panel done t={time.time()-t0:.0f}s", flush=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    vmax = max(g.max() for _, g in panels)
    for a, (lab, gm) in zip(ax.flat, panels):
        im = a.pcolormesh(phis, thetas, gm, cmap="viridis", vmin=0, vmax=vmax,
                          shading="auto")
        a.set_title(lab, fontsize=9)
        a.plot(0, 0, "r*", ms=12)          # B along wire, in-plane
        a.plot(90, 90, "w^", ms=9)         # B out-of-plane
    for a in ax[1]:
        a.set_xlabel("φ (°)  [0 = along wire, 90 = in-plane ⊥]")
    for a in ax[:, 0]:
        a.set_ylabel("θ (°)  [0 = in-plane, 90 = out-of-plane]")
    fig.colorbar(im, ax=ax, label="max topological gap (µeV)")
    fig.suptitle("Field-orientation maps (µ=0): star = in-plane along wire; "
                 "triangle = out-of-plane.\nLK says: out-of-plane B (g_z) with "
                 "a thick parent; the in-plane/wire direction is g-starved.",
                 y=1.0)
    fig.savefig(os.path.join(OUT, "fig11_field_orientation.png"), dpi=150,
                bbox_inches="tight")

    kn = {}
    for lab, gm in panels:
        kn[lab] = dict(best=round(float(gm.max()), 1),
                       at_theta_phi=[float(thetas[np.unravel_index(
                           gm.argmax(), gm.shape)[0]]),
                           float(phis[np.unravel_index(
                               gm.argmax(), gm.shape)[1]])],
                       inplane_wire=round(float(gm[0, 0]), 1),
                       outofplane=round(float(gm[-1, 0]), 1))
    kn["runtime_s"] = round(time.time() - t0, 1)
    save_numbers("fig11", kn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all")
    ap.add_argument("--rows", type=int, default=None,
                    help="row budget for fig4 (checkpointed)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="cell budget for fig5 (checkpointed)")
    args = ap.parse_args()
    todo = [args.fig] if args.fig != "all" else [str(i) for i in range(1, 12)]
    dispatch = {"1": fig1, "2": fig2, "3": fig3,
                "4": lambda: fig4(args.rows), "5": lambda: fig5(args.chunk),
                "6": lambda: fig6(args.chunk), "7": lambda: fig7(args.chunk),
                "8": lambda: fig8(args.chunk), "9": lambda: fig9(args.chunk),
                "10": lambda: fig10(args.chunk), "11": lambda: fig11(args.chunk)}
    for f in todo:
        dispatch[f]()
