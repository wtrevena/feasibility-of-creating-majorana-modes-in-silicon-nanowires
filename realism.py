"""
realism.py — review-round-5 controls: parent-superconductor realism and
disorder realism beyond the static-Delta / iid-onsite baseline.

Sections (checkpointed to output/data/realism.json, key numbers merged into
key_numbers.json under tag "realism"):

  SE     Dynamic tunneling self-energy. The wire-parent coupling is treated
         with the full frequency-dependent BCS self-energy
            Sigma(w) = -Gamma (w + Delta_p tau_x) / sqrt(Delta_p^2 - w^2),
         and quasiparticle energies solve det[H(k) + Sigma(w) - w] = 0.
         At w = 0 the induced pairing is exactly Gamma, so the topological
         criterion is E_Z^2 > Gamma^2 + mu^2 (gap closings sit at w = 0 by
         particle-hole symmetry). alpha/g/m renormalization is automatic
         (the w-linear term carries quasiparticle weight Z(w)).
         Outputs: dynamic topological gap vs Gamma at each parent operating
         point, the Gamma-optimal dynamic gap, and the static-model
         comparison (validating the static caricature's direction and size).
  DY     Dynes broadening: subgap DOS fraction gamma_D/Delta_p and its
         effect on the QP-poisoning temperature budget.
  DP     Parent-gap disorder: correlated Delta(x) fluctuations in the
         finite wire (site-resolved builder, validated against build_wire).
  CD     Correlated electrostatic disorder: smooth mu(x) (Gaussian random
         field, correlation length lambda_c) compared with the iid baseline
         at matched RMS; plus alpha(x) and g(x) fluctuations.

Usage: python realism.py [--sec SE,DY,DP,CD|all]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg  # noqa: F401
except Exception:
    sys.path.insert(0, os.path.join(_HERE, "compat"))
    import scipy.sparse as sp  # noqa: F401

from majorana_sim import (HBAR, ME, QE, UEV, KB, s0, sx, sy, sz, EZ_J,
                          bulk_gap_ueV, build_wire, solve_lowest)
from run_analysis import DATA, save_numbers, M_HOLE, SIB_DELTA0, SIB_BC2_MEAS

CKPT = os.path.join(DATA, "realism.json")
AL_H, G_H = 0.06, 2.2          # empirical-center hole point


def _load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def _save(d):
    json.dump(d, open(CKPT, "w"), indent=2)


# ------------------------------------------------------ SE: dynamic Sigma(w)
def _hk4(k, mu_J, EZ, aSI, m_kg):
    """4x4 BdG normal block: H = (xi + a k sigma_z) tau_z + E_Z sigma_x tau_0,
    ordering tau (x) sigma."""
    xi = HBAR**2 * k**2 / (2 * m_kg) - mu_J
    tz = np.array([[1, 0], [0, -1.0]])
    t0 = np.eye(2)
    h = np.kron(tz, xi * s0 + aSI * k * sz) + EZ * np.kron(t0, sx)
    return h


TAUX4 = np.kron(np.array([[0, 1], [1, 0.0]]), s0)


def dynamic_gap_ueV(mu_ueV, B, Dp_ueV, Gamma_ueV, alpha_eVA, m_rel, g,
                    nk=801, nw=240):
    """Topological gap with the full frequency-dependent self-energy.
    Returns 0 if not topological (criterion E_Z^2 > Gamma^2 + mu^2, exact
    at w=0). Solved by scanning w for the first eigenvalue crossing of
    A(w) = H(k) + Sigma(w), batched over k."""
    m = m_rel * ME
    mu = mu_ueV * UEV
    Dp = Dp_ueV * UEV
    G = Gamma_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    if EZ**2 <= G**2 + mu**2 or Dp_ueV <= 0.5:
        return 0.0
    kF = np.sqrt(2 * m * (abs(mu) + EZ + G)) / HBAR
    kso = m * aSI / HBAR**2
    kmax = 4.0 * (kF + kso) + 2e7
    ks = np.linspace(0, kmax, nk)
    Hk = np.stack([_hk4(k, mu, EZ, aSI, m) for k in ks])  # (nk,4,4)
    ws = np.linspace(0.0, 0.995 * Dp, nw)
    # s(k,w) = (smallest |eigenvalue - w|, signed branch min_i (lam_i - w)
    # restricted to lam_i >= 0 at w=0 and tracked by sign change)
    first_cross = np.full(nk, np.inf)
    prev = None
    for iw, w in enumerate(ws):
        den = np.sqrt(Dp**2 - w**2)
        Sig = -(G / den) * (w * np.eye(4) + Dp * TAUX4)
        lam = np.linalg.eigvalsh(Hk + Sig)            # (nk, 4) ascending
        # distance of the positive branch from the w-line:
        smin = np.where(lam >= 0, lam, np.inf).min(axis=1) - w
        if prev is not None:
            hit = (prev > 0) & (smin <= 0) & (first_cross == np.inf)
            if hit.any():
                # linear interpolation in w
                w0 = ws[iw - 1]
                frac = prev[hit] / (prev[hit] - smin[hit])
                first_cross[hit] = w0 + frac * (w - w0)
        prev = smin
    gap = first_cross.min()
    if not np.isfinite(gap):
        gap = ws[-1]                                   # gap pinned at parent edge
    return float(gap / UEV)


def sec_SE(res):
    """Gamma scan of the dynamic gap at the three parent operating points,
    against the static-caricature numbers."""
    out = {}
    pts = dict(
        SiB_meas=dict(B=0.33, Dp=SIB_DELTA0 * (1 - (0.33 / SIB_BC2_MEAS)**2),
                      static_bare=11.0, static_renorm=10.1),
        SiB_pauli=dict(B=1.0, Dp=SIB_DELTA0 - 5.7883818060e1 * 1.0,
                       static_bare=30.6, static_renorm=19.7),
        Al=dict(B=1.7, Dp=200.0 * (1 - (1.7 / 2.0)**2),
                static_bare=50.4, static_renorm=34.4),
    )
    Gammas = [3, 6, 10, 15, 20, 30, 45, 70, 100, 150, 250, 400]
    for tag, p in pts.items():
        row = {}
        best = (0.0, None)
        for G in Gammas:
            gap = dynamic_gap_ueV(0.0, p["B"], p["Dp"], G, AL_H, M_HOLE, G_H)
            row[str(G)] = round(gap, 3)
            if gap > best[0]:
                best = (gap, G)
        # static comparison at the SAME effective coupling: Delta_ind(G*)
        Gs = best[1]
        Dind_eq = Gs * p["Dp"] / (Gs + p["Dp"])
        stat = bulk_gap_ueV(0.0, p["B"], Dind_eq, AL_H, M_HOLE, G_H)
        out[tag] = dict(Dp_ueV=round(p["Dp"], 1), gap_vs_Gamma=row,
                        best_gap=round(best[0], 2), best_Gamma=best[1],
                        static_at_matched_Dind=round(stat, 2),
                        static_bare_quoted=p["static_bare"],
                        static_renorm_quoted=p["static_renorm"])
        print("SE", tag, out[tag], flush=True)
    res["SE_selfenergy"] = out


def sec_DY(res):
    """Dynes subgap DOS and the QP temperature budget."""
    out = {}
    for tag, Dp in (("SiB_meas_op", 29.1), ("SiB_pauli_op", 33.1),
                    ("Al_op", 55.5)):
        row = {}
        for gD_frac in (1e-4, 1e-3, 1e-2):
            n_sub = gD_frac / np.sqrt(gD_frac**2 + 1)   # N(0)/N_n = gD/sqrt(gD^2+Dp^2) with gD in units of Dp
            # equilibrium x_qp floor set by Dynes states: x_qp >= n_sub
            # temperature where thermal x_qp equals the Dynes floor:
            # sqrt(2 pi kT/Dp) exp(-Dp/kT) = n_sub  -> solve for T
            DpJ = Dp * UEV
            from scipy.optimize import brentq
            f = lambda T: (np.sqrt(2 * np.pi * KB * T / DpJ)
                           * np.exp(-DpJ / (KB * T)) - n_sub)
            try:
                Tstar = brentq(f, 1e-3, 5.0)
            except ValueError:
                Tstar = np.nan
            row[f"gammaD/Dp={gD_frac:g}"] = dict(
                subgap_DOS_frac=float(f"{n_sub:.2e}"),
                T_below_which_Dynes_dominates_mK=round(Tstar * 1e3, 1))
        out[tag] = row
        print("DY", tag, row, flush=True)
    res["DY_dynes"] = out


# --------------------------------------- site-resolved builder (DP, CD)
def build_wire_sitewise(N, dx, mu_ueV_arr, B, Delta_ueV_arr, alpha_eVA_arr,
                        m_rel, g_arr):
    """build_wire generalized to site-dependent mu, Delta, alpha, g.
    Validated against build_wire for uniform arrays (test in sec_DP)."""
    m = m_rel * ME
    t = HBAR**2 / (2 * m * dx**2)
    mu = np.asarray(mu_ueV_arr, float) * UEV
    D = np.asarray(Delta_ueV_arr, float) * UEV
    aSI = np.asarray(alpha_eVA_arr, float) * 1e-10 * QE
    EZs = np.array([EZ_J(gv, B) for gv in np.broadcast_to(g_arr, (N,))])
    diag_blocks = [(2 * t - mu[n]) * s0 + EZs[n] * sx for n in range(N)]
    h = sp.block_diag(diag_blocks, format="lil")
    K = sp.diags(np.ones(N - 1), 1, format="csr")
    hop_t = sp.kron(K, -t * s0)
    h = (h + hop_t + hop_t.conj().T).tolil()
    # bond-averaged SOC hopping
    hb = sp.lil_matrix((2 * N, 2 * N), dtype=complex)
    for n in range(N - 1):
        aso = 0.5 * (aSI[n] + aSI[n + 1]) / (2 * dx)
        hb[2*n:2*n+2, 2*(n+1):2*(n+1)+2] = -1j * aso * sz
    hb = hb.tocsr()
    h = (h.tocsr() + hb + hb.conj().T).tocsr()
    Dm = sp.block_diag([D[n] * (1j * sy) for n in range(N)], format="csr")
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


def grf(N, dx, lam_c, rng):
    """Gaussian random field, unit RMS, correlation length lam_c."""
    white = rng.standard_normal(N)
    x = np.arange(N) * dx
    # convolve with Gaussian kernel
    kern = np.exp(-0.5 * ((x - x[N // 2]) / lam_c)**2)
    f = np.real(np.fft.ifft(np.fft.fft(white) * np.fft.fft(np.fft.ifftshift(kern))))
    return f / f.std()


# operating point for the hole wire (fig8 center, empirical tensor)
HOLE_OP = dict(mu=0.0, B=1.0, Dind=33.0, al=AL_H, g=G_H, m=M_HOLE,
               N=1200, dx=2.5e-9)


def _e2(H, N):
    E, _ = solve_lowest(H, k=6)
    return float(np.sort(np.abs(E))[2] / UEV)


def sec_DP(res):
    p = HOLE_OP
    N, dx = p["N"], p["dx"]
    ones = np.ones(N)
    # validation: uniform sitewise == build_wire
    Hu = build_wire_sitewise(N, dx, p["mu"] * ones, p["B"], p["Dind"] * ones,
                             p["al"] * ones, p["m"], p["g"])
    Hr = build_wire(N, dx, p["mu"], p["B"], p["Dind"], p["al"], p["m"], p["g"])
    E1, E2v = _e2(Hu, N), _e2(Hr, N)
    assert abs(E1 - E2v) < 1e-6, (E1, E2v)
    out = dict(validation_uniform=dict(sitewise=round(E1, 4),
                                       build_wire=round(E2v, 4)))
    clean = E1
    for sig_frac in (0.1, 0.25, 0.5):
        meds = []
        for s in range(10):
            rng = np.random.default_rng([91, int(100 * sig_frac), s])
            Dx = p["Dind"] * (1 + sig_frac * grf(N, dx, 50e-9, rng))
            Dx = np.clip(Dx, 0, None)
            H = build_wire_sitewise(N, dx, p["mu"] * ones, p["B"], Dx,
                                    p["al"] * ones, p["m"], p["g"])
            meds.append(_e2(H, N))
        out[f"Delta_disorder_{int(100*sig_frac)}pct_lc50nm"] = dict(
            p5=round(float(np.percentile(meds, 5)), 2),
            p50=round(float(np.median(meds)), 2),
            p95=round(float(np.percentile(meds, 95)), 2))
        print("DP", sig_frac, out[f"Delta_disorder_{int(100*sig_frac)}pct_lc50nm"],
              flush=True)
    out["clean_E2"] = round(clean, 2)
    res["DP_parent_disorder"] = out


def sec_CD(res):
    """Correlated smooth mu(x) disorder vs iid at matched RMS; alpha/g
    fluctuations at the hole operating point."""
    p = HOLE_OP
    N, dx = p["N"], p["dx"]
    ones = np.ones(N)
    out = {}
    for lam_c_nm in (10, 25, 50, 100):
        for W_rms in (25.0, 50.0, 100.0):
            meds = []
            for s in range(10):
                rng = np.random.default_rng([92, lam_c_nm, int(W_rms), s])
                mux = W_rms * grf(N, dx, lam_c_nm * 1e-9, rng)
                H = build_wire_sitewise(N, dx, mux, p["B"], p["Dind"] * ones,
                                        p["al"] * ones, p["m"], p["g"])
                meds.append(_e2(H, N))
            out[f"mu_grf_lc{lam_c_nm}nm_rms{int(W_rms)}"] = dict(
                p5=round(float(np.percentile(meds, 5)), 2),
                p50=round(float(np.median(meds)), 2),
                p95=round(float(np.percentile(meds, 95)), 2))
        print("CD lam_c", lam_c_nm, "done", flush=True)
    # iid baseline at matched RMS (uniform[-W,W] has rms W/sqrt(3))
    for W_rms in (25.0, 50.0, 100.0):
        W = W_rms * np.sqrt(3)
        meds = []
        for s in range(10):
            rng = np.random.default_rng([93, int(W_rms), s])
            H = build_wire(N, dx, p["mu"], p["B"], p["Dind"], p["al"],
                           p["m"], p["g"], disorder_ueV=W, rng=rng)
            meds.append(_e2(H, N))
        out[f"mu_iid_rms{int(W_rms)}"] = dict(
            p50=round(float(np.median(meds)), 2))
    # alpha and g fluctuations (20% rms, lc=50nm)
    for which in ("alpha", "g"):
        meds = []
        for s in range(10):
            rng = np.random.default_rng([94, {"alpha": 1, "g": 2}[which], s])
            f = 1 + 0.2 * grf(N, dx, 50e-9, rng)
            alx = p["al"] * (f if which == "alpha" else ones)
            gx = p["g"] * (f if which == "g" else 1.0)
            H = build_wire_sitewise(N, dx, p["mu"] * ones, p["B"],
                                    p["Dind"] * ones, alx, p["m"],
                                    gx if which == "g" else p["g"])
            meds.append(_e2(H, N))
        out[f"{which}_20pct_lc50nm"] = dict(
            p5=round(float(np.percentile(meds, 5)), 2),
            p50=round(float(np.median(meds)), 2),
            p95=round(float(np.percentile(meds, 95)), 2))
        print("CD", which, out[f"{which}_20pct_lc50nm"], flush=True)
    res["CD_correlated_disorder"] = out


SECS = dict(SE=sec_SE, DY=sec_DY, DP=sec_DP, CD=sec_CD)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all")
    args = ap.parse_args()
    todo = list(SECS) if args.sec == "all" else args.sec.split(",")
    res = _load()
    t0 = time.time()
    for s_ in todo:
        if any(k.startswith(s_ + "_") for k in res):
            print(f"section {s_}: cached", flush=True)
            continue
        print(f"=== {s_} (t={time.time()-t0:.0f}s)", flush=True)
        SECS[s_](res)
        _save(res)
    if all(any(k.startswith(s_ + "_") for k in res) for s_ in SECS):
        save_numbers("realism", res)
