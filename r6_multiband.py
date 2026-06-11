"""
r6_multiband.py - review-round-6 item A4: (M) multi-subband occupancy
robustness for the HOLE route and (G) spatially-dependent proximity
coupling Gamma(x) with the local inverse-proximity (metallization) renorm.

M  (sections M2, M3, M4 - Ny = 2, 3, 4 transverse rows; hard-wall width
   (Ny+1)*dy = 15, 20, 25 nm at dy = 5 nm):
   build_wire_2d (the fig7 multi-subband strip) evaluated at the six-band
   [110] hole point (kp6_110, Ez=30 MV/m: m* = 0.20, alpha = 0.073 eV*A,
   g_x' = 1.657) and the SiB_pauli operating point (B = 1.0 T,
   Delta_ind = 24.7 ueV; platform110 sec A, Ez30_SiB_pauli). For each
   transverse subband n <= 3 the chemical potential is scanned across the
   kx = 0 subband edge - computed INCLUDING the transverse SOC term
   alpha sigma_x k_y, which shifts the lattice edge by up to several tens
   of ueV (comparable to the 41-ueV topological half-width, so centering
   on the bare cosine-band bottoms would miss the domes). E0, E2 and the
   end weight vs mu give the realized topological window per subband.
   CAVEAT (state in the paper): build_wire_2d is the single-band
   effective-mass model fed with hole parameters, NOT the full
   Luttinger-Kohn multiband valence structure. This is an occupancy /
   subband-coupling control, not full valence-band realism; the effective
   treatment at the bottom of the lowest subband is justified by the kp6
   LK subband spacings (sine basis: ~0.7-6 meV for W <= 40 nm), far above
   E_Z ~ 48 ueV and Delta_ind ~ 25 ueV.

G  (sections GU, GF, GD, GE): static-limit site-dependent proximity.
   The full dynamic Sigma(x, omega) is beyond scope (documented); the
   standard first step is the static site-resolved induced gap
       Dind(x) = Gamma(x) Dp / (Gamma(x) + Dp),   Dp = 33.1 ueV
   (SiB_pauli parent at B = 1 T) plus the LOCAL metallization renorm
       Z(x) = 1 - Dind(x)/Dp,   alpha(x) = Z(x) alpha,
       g(x) = Z(x) g + (1 - Z(x)) g_parent,  g_parent = 2
   - the same convention as run_analysis._best_gap_hole(renormalize=True),
   applied site-by-site via realism.build_wire_sitewise at the [110]
   point (m* = 0.204, alpha = 0.073 eV*A, g = 1.657, B = 1 T, mu = 0,
   N = 1200, dx = 2.5 nm, L = 3 um). An uncovered site (Gamma = 0) keeps
   the bare wire parameters and zero pairing - the physically correct
   inverse-proximity limit.
   GU: uniform Gamma0 = 20 ueV (platform110 sec-C dynamic optimum for
       SiB_pauli) + sitewise-vs-build_wire validation.
   GF: Gaussian-random-field Gamma(x) fluctuations, RMS {25, 50}%,
       lc {25, 100} nm, 8 seeds each.
   GD: interior DEAD ZONES Gamma = 0 (parent-film gaps / grain
       boundaries), length {50, 150} nm, one per um (3 per wire), 8 seeds.
   GE: END dead zone - last 100 nm uncovered (deterministic lithography
       case) + shortened-uniform-wire control.

Usage:  python r6_multiband.py --sec M2   (M2,M3,M4,GU,GF,GD,GE or all)
Checkpoints: output/data/r6_multiband.json; ledger tag "r6_multiband"
(written when all seven sections are complete; "scan" curves stripped).
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

from majorana_sim import (HBAR, ME, QE, UEV, s0, sx, EZ_J, bulk_gap_ueV,
                          build_wire, build_wire_2d, subband_bottoms_ueV,
                          solve_lowest, site_density, end_weight)
from run_analysis import DATA, save_numbers
from realism import build_wire_sitewise, grf

CKPT = os.path.join(DATA, "r6_multiband.json")

# ---- M: [110] six-band hole point (kp6_110 Ez=30) at SiB_pauli (platform110)
M_REL, AL110, G110 = 0.20, 0.073, 1.657
B_M, DIND_M = 1.0, 24.7
NX, DXM, DYM = 300, 5e-9, 5e-9
EZ_UEV = EZ_J(G110, B_M) / UEV
WIN = float(np.sqrt(EZ_UEV**2 - DIND_M**2))     # 1D window half-width

# ---- G: sitewise point ([110] Ez=30 values as used by platform110/realism)
N_G, DXG = 1200, 2.5e-9
M_G, AL_G, G_G = 0.204, 0.073, 1.657
B_G, DP_G, GAM0 = 1.0, 33.1, 20.0
G_PARENT = 2.0


def _load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def _save(d):
    json.dump(d, open(CKPT, "w"), indent=2)


# ------------------------------------------------------------------ M tools
def kx0_edges_ueV(Ny, dy, m_rel, alpha_eVA):
    """kx = 0 transverse subband edges (ueV) of build_wire_2d INCLUDING the
    transverse SOC hopping (-ty s0 - i(a/2dy) sx), no Zeeman. The 2Ny
    eigenvalues come in Kramers pairs; the unique values are the edges.
    Zeeman (also sigma_x: it commutes) splits each edge symmetrically, so
    the topological dome of subband n is centered here, not at the bare
    cosine-band bottom (shift -2 cos(theta_j) (sqrt(ty^2+aso^2) - ty))."""
    m = m_rel * ME
    ty = HBAR**2 / (2 * m * dy**2)
    aSI = alpha_eVA * 1e-10 * QE
    hop_y = -ty * s0 - 1j * (aSI / (2 * dy)) * sx
    K = np.diag(np.ones(Ny - 1), 1)
    Ht = (np.kron(np.eye(Ny), 2 * ty * s0) + np.kron(K, hop_y)
          + np.kron(K, hop_y).conj().T)
    vals = np.linalg.eigvalsh(Ht)
    return vals[0::2] / UEV


def x_end_weight(vec, Nx, Ny, frac=0.1):
    """Fraction of |psi|^2 in the outer frac of the wire LENGTH (x)."""
    half = vec.shape[0] // 2
    w = np.abs(vec[:half])**2 + np.abs(vec[half:])**2
    profile = w.reshape(Nx, 2 * Ny).sum(axis=1)
    n = max(1, int(frac * Nx))
    return float((profile[:n].sum() + profile[-n:].sum()) / profile.sum())


def _width(mus, e0s, thr):
    """Width (ueV) of the contiguous E0 < thr region containing the E0
    minimum, edges by linear interpolation. (0, None, None) if none."""
    mus = np.asarray(mus, float)
    e0s = np.asarray(e0s, float)
    i0 = int(np.argmin(e0s))
    if e0s[i0] >= thr:
        return 0.0, None, None
    il = i0
    while il > 0 and e0s[il - 1] < thr:
        il -= 1
    ir = i0
    while ir < len(e0s) - 1 and e0s[ir + 1] < thr:
        ir += 1
    if il == 0:
        lo = mus[0]
    else:
        lo = mus[il] - (mus[il] - mus[il - 1]) * (thr - e0s[il]) / (e0s[il - 1] - e0s[il])
    if ir == len(e0s) - 1:
        hi = mus[-1]
    else:
        hi = mus[ir] + (mus[ir + 1] - mus[ir]) * (thr - e0s[ir]) / (e0s[ir + 1] - e0s[ir])
    return float(hi - lo), float(lo), float(hi)


def sec_M(res, Ny):
    key = f"M{Ny}"
    sub = res.get(key, {})
    bots = subband_bottoms_ueV(Ny, DYM, M_REL)
    edges = kx0_edges_ueV(Ny, DYM, M_REL, AL110)
    nb = min(3, Ny)
    sub["params"] = dict(
        Nx=NX, Ny=Ny, dx_nm=DXM * 1e9, dy_nm=DYM * 1e9, L_um=NX * DXM * 1e6,
        W_hardwall_nm=(Ny + 1) * DYM * 1e9, m_rel=M_REL, alpha_eVA=AL110,
        g=G110, B_T=B_M, Dind_ueV=DIND_M, EZ_ueV=round(EZ_UEV, 2),
        win_pred_halfwidth_ueV=round(WIN, 2),
        gap_1D_same_params_ueV=round(
            bulk_gap_ueV(0.0, B_M, DIND_M, AL110, M_REL, G110), 2),
        bottoms_noSOC_ueV=[round(float(b), 1) for b in bots],
        edges_kx0_SOC_ueV=[round(float(e), 1) for e in edges],
        n_bands_scanned=nb,
        note=("Ny=2 has only 2 transverse subbands" if Ny == 2 else ""))
    for n in range(nb):
        bk = f"band{n + 1}"
        if sub.get(bk, {}).get("done"):
            print(f"{key} {bk}: cached", flush=True)
            continue
        t0 = time.time()
        mus = np.linspace(edges[n] - 100.0, edges[n] + 100.0, 25)
        rows = []
        for mu in mus:
            H = build_wire_2d(NX, Ny, DXM, DYM, float(mu), B_M, DIND_M,
                              AL110, M_REL, G110)
            E, V = solve_lowest(H, k=6)
            Ea = np.sort(np.abs(E)) / UEV
            rows.append([float(mu), float(Ea[0]), float(Ea[2]),
                         x_end_weight(V[:, 0], NX, Ny)])
            print(f"[{key} {bk}] mu={mu:9.1f} E0={Ea[0]:8.4f} "
                  f"E2={Ea[2]:7.3f} ew={rows[-1][3]:.3f}", flush=True)
        arr = np.array(rows)
        met = {}
        for thr in (1.0, 2.0, 5.0):
            w, lo, hi = _width(arr[:, 0], arr[:, 1], thr)
            met[f"width_thr{thr:g}_ueV"] = round(w, 1)
        inw = np.abs(arr[:, 0] - edges[n]) < WIN - 5
        met.update(
            median_E0_inwin_ueV=round(float(np.median(arr[inw, 1])), 4),
            max_E2_inwin_ueV=round(float(arr[inw, 2].max()), 2),
            median_E2_inwin_ueV=round(float(np.median(arr[inw, 2])), 2),
            median_ew_inwin=round(float(np.median(arr[inw, 3])), 3))
        sub[bk] = dict(edge_noSOC_ueV=round(float(bots[n]), 1),
                       edge_soc_ueV=round(float(edges[n]), 1),
                       soc_edge_shift_ueV=round(float(edges[n] - bots[n]), 1),
                       **met, runtime_s=round(time.time() - t0, 1),
                       scan=[[round(v, 4) for v in r] for r in rows],
                       done=True)
        res[key] = sub
        _save(res)
        print(f"{key} {bk} done: " + json.dumps(met), flush=True)
    sub["complete"] = True
    res[key] = sub


# ------------------------------------------------------------------ G tools
def _gamma_maps(Gam):
    """Gamma(x) (ueV) -> (Dind(x), alpha(x), g(x)) under the static-limit
    induced gap + local metallization renorm (see module docstring)."""
    Gam = np.asarray(Gam, float)
    Dind = Gam * DP_G / (Gam + DP_G)
    Z = 1.0 - Dind / DP_G
    return Dind, Z * AL_G, Z * G_G + (1 - Z) * G_PARENT


def _solve_gamma(Gam, k=6):
    Dind, alx, gx = _gamma_maps(Gam)
    H = build_wire_sitewise(N_G, DXG, np.zeros(N_G), B_G, Dind, alx, M_G, gx)
    E, V = solve_lowest(H, k=k)
    return np.sort(np.abs(E)) / UEV, E, V


def sec_GU(res):
    out = {}
    Dind0, al0, g0 = (float(v) for v in _gamma_maps(GAM0))
    EZ0 = EZ_J(g0, B_G) / UEV
    out["point"] = dict(
        Gamma0_ueV=GAM0, Dp_ueV=DP_G, Dind_ueV=round(Dind0, 2),
        Z=round(1 - Dind0 / DP_G, 4), alpha_eff_eVA=round(al0, 4),
        g_eff=round(g0, 4), EZ_eff_ueV=round(EZ0, 2),
        win_pred_halfwidth_ueV=round(float(np.sqrt(EZ0**2 - Dind0**2)), 1),
        N=N_G, dx_nm=DXG * 1e9, L_um=N_G * DXG * 1e6, mu_ueV=0.0, B_T=B_G,
        m_rel=M_G, alpha_bare_eVA=AL_G, g_bare=G_G, g_parent=G_PARENT)
    Ea, _, V = _solve_gamma(GAM0 * np.ones(N_G))
    dens = site_density(V[:, 0], N_G)
    out["uniform"] = dict(E0_ueV=round(float(Ea[0]), 4),
                          E2_ueV=round(float(Ea[2]), 3),
                          end_weight=round(end_weight(dens), 3))
    Hr = build_wire(N_G, DXG, 0.0, B_G, Dind0, al0, M_G, g0)
    Er, _ = solve_lowest(Hr, k=6)
    Ear = np.sort(np.abs(Er)) / UEV
    out["validation_vs_build_wire"] = dict(
        E2_sitewise=round(float(Ea[2]), 6),
        E2_build_wire=round(float(Ear[2]), 6),
        agree=bool(abs(float(Ea[2]) - float(Ear[2])) < 1e-6))
    out["complete"] = True
    res["GU"] = out
    print("GU " + json.dumps(out["uniform"]) + " "
          + json.dumps(out["validation_vs_build_wire"]), flush=True)


def sec_GF(res):
    out = res.get("GF", {})
    for rms in (0.25, 0.50):
        for lc_nm in (25, 100):
            ck = f"rms{int(rms * 100)}_lc{lc_nm}nm"
            if ck in out:
                print(f"GF {ck}: cached", flush=True)
                continue
            rows, dind_rms = [], []
            for s_ in range(8):
                rng = np.random.default_rng([1146, int(rms * 100), lc_nm, s_])
                Gam = np.clip(GAM0 * (1 + rms * grf(N_G, DXG, lc_nm * 1e-9,
                                                    rng)), 0.0, None)
                Dind, _, _ = _gamma_maps(Gam)
                dind_rms.append(float(Dind.std() / Dind.mean()))
                Ea, _, V = _solve_gamma(Gam)
                dens = site_density(V[:, 0], N_G)
                rows.append([float(Ea[0]), float(Ea[2]),
                             end_weight(dens)])
                print(f"[GF {ck}] seed{s_} E0={Ea[0]:.4f} E2={Ea[2]:.3f}",
                      flush=True)
            a = np.array(rows)
            out[ck] = dict(
                E2_p5=round(float(np.percentile(a[:, 1], 5)), 2),
                E2_p50=round(float(np.median(a[:, 1])), 2),
                E2_p95=round(float(np.percentile(a[:, 1], 95)), 2),
                E2_min=round(float(a[:, 1].min()), 2),
                E0_p50=round(float(np.median(a[:, 0])), 4),
                E0_max=round(float(a[:, 0].max()), 4),
                ew_p50=round(float(np.median(a[:, 2])), 3),
                Dind_rms_frac=round(float(np.mean(dind_rms)), 3),
                seeds=[[round(v, 4) for v in r] for r in rows])
            res["GF"] = out
            _save(res)
            print(f"GF {ck} done: E2 p5/p50/p95 = {out[ck]['E2_p5']}/"
                  f"{out[ck]['E2_p50']}/{out[ck]['E2_p95']}", flush=True)
    out["complete"] = True
    res["GF"] = out


def _zone_starts(rng, n_dz, lz, N):
    lo, hi = int(0.08 * N), int(0.92 * N) - lz
    for _ in range(10000):
        st = np.sort(rng.integers(lo, hi, n_dz))
        if n_dz == 1 or np.all(np.diff(st) > lz + 40):
            return st
    raise RuntimeError("zone placement failed")


def sec_GD(res):
    out = res.get("GD", {})
    n_dz = int(round(N_G * DXG / 1e-6))      # one per um -> 3
    for lz_nm in (50, 150):
        ck = f"dz{lz_nm}nm_x{n_dz}"
        if ck in out:
            print(f"GD {ck}: cached", flush=True)
            continue
        lz = int(round(lz_nm * 1e-9 / DXG))
        rows = []
        for s_ in range(8):
            rng = np.random.default_rng([1147, lz_nm, s_])
            Gam = GAM0 * np.ones(N_G)
            for st in _zone_starts(rng, n_dz, lz, N_G):
                Gam[st:st + lz] = 0.0
            Ea, _, V = _solve_gamma(Gam)
            d0 = site_density(V[:, 0], N_G)
            d2 = site_density(V[:, 2], N_G)
            rows.append([float(Ea[0]), float(Ea[2]), end_weight(d0),
                         end_weight(d2), float(np.argmax(d2)) / N_G])
            print(f"[GD {ck}] seed{s_} E0={Ea[0]:.4f} E2={Ea[2]:.3f} "
                  f"ew={rows[-1][2]:.3f} ew(E2)={rows[-1][3]:.3f} "
                  f"E2peak@{rows[-1][4]:.2f}L", flush=True)
        a = np.array(rows)
        out[ck] = dict(
            dead_zone_nm=lz_nm, n_zones=n_dz,
            E2_p5=round(float(np.percentile(a[:, 1], 5)), 2),
            E2_p50=round(float(np.median(a[:, 1])), 2),
            E2_min=round(float(a[:, 1].min()), 2),
            E0_p50=round(float(np.median(a[:, 0])), 4),
            E0_max=round(float(a[:, 0].max()), 4),
            ew_p50=round(float(np.median(a[:, 2])), 3),
            ew_E2state_p50=round(float(np.median(a[:, 3])), 3),
            E2state_peak_pos_fracL=[round(float(v), 2) for v in a[:, 4]],
            seeds=[[round(v, 4) for v in r] for r in rows])
        res["GD"] = out
        _save(res)
        print(f"GD {ck} done: E2 p5/p50 = {out[ck]['E2_p5']}/"
              f"{out[ck]['E2_p50']}", flush=True)
    out["complete"] = True
    res["GD"] = out


def sec_GE(res):
    out = {}
    lz = int(round(100e-9 / DXG))            # 40 sites
    Gam = GAM0 * np.ones(N_G)
    Gam[-lz:] = 0.0
    Ea, _, V = _solve_gamma(Gam)
    d0 = site_density(V[:, 0], N_G)
    right = d0.copy()
    right[:N_G // 2] = 0.0
    out["end_dead_100nm"] = dict(
        E0_ueV=round(float(Ea[0]), 4), E2_ueV=round(float(Ea[2]), 3),
        end_weight=round(end_weight(d0), 3),
        weight_in_dead_zone=round(float(d0[-lz:].sum() / d0.sum()), 3),
        right_peak_pos_fracL=round(float(np.argmax(right)) / N_G, 3),
        covered_boundary_fracL=round(1 - lz / N_G, 3), dead_sites=lz)
    # control: uniform wire shortened by the dead length
    Ns = N_G - lz
    ones = np.ones(Ns)
    Dind0, al0, g0 = (float(v) for v in _gamma_maps(GAM0))
    Hs = build_wire_sitewise(Ns, DXG, 0.0 * ones, B_G, Dind0 * ones,
                             al0 * ones, M_G, g0 * ones)
    Es, _ = solve_lowest(Hs, k=6)
    Eas = np.sort(np.abs(Es)) / UEV
    out["shortened_uniform_control"] = dict(
        N=Ns, L_um=Ns * DXG * 1e6, E0_ueV=round(float(Eas[0]), 4),
        E2_ueV=round(float(Eas[2]), 3))
    out["complete"] = True
    res["GE"] = out
    print("GE " + json.dumps(out["end_dead_100nm"]) + " control "
          + json.dumps(out["shortened_uniform_control"]), flush=True)


# ------------------------------------------------------------------- runner
SECS = ("M2", "M3", "M4", "GU", "GF", "GD", "GE")
RUN = {"M2": lambda r: sec_M(r, 2), "M3": lambda r: sec_M(r, 3),
       "M4": lambda r: sec_M(r, 4), "GU": sec_GU, "GF": sec_GF,
       "GD": sec_GD, "GE": sec_GE}


def _slim(res):
    def rec(o):
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items() if k != "scan"}
        return o
    return rec(json.loads(json.dumps(res)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all")
    args = ap.parse_args()
    todo = list(SECS) if args.sec == "all" else args.sec.split(",")
    res = _load()
    t0 = time.time()
    for s_ in todo:
        if res.get(s_, {}).get("complete"):
            print(f"section {s_}: cached", flush=True)
            continue
        print(f"=== {s_} (t={time.time() - t0:.0f}s)", flush=True)
        RUN[s_](res)
        _save(res)
    if all(res.get(s_, {}).get("complete") for s_ in SECS):
        save_numbers("r6_multiband", _slim(res))
        print("ledger tag r6_multiband saved", flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)
