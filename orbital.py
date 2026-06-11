"""
orbital.py — review-round-5 item R5-2: representative ORBITAL magnetic-field
control for the hole-platform field-orientation results (figs 8/10/11).

A referee noted that every wire calculation in the study omits orbital
coupling of the magnetic field, and that the preferred tilted /
out-of-plane-field operating points of the hole platform could be
invalidated by it. This script builds a 2D strip (x along the wire, y
transverse, hard walls) with 2D Rashba SOC, a Zeeman term and s-wave
pairing — the same lattice model as majorana_sim.build_wire_2d — and adds
the orbital coupling of the out-of-plane field component B_perp via Peierls
phases on the x-hoppings (Landau gauge A = (-B_perp*y, 0, 0), with y
measured from the strip centre):

    t_x(n,m -> n+1,m)  ->  t_x * exp(+i (e/hbar) B_perp y_m dx)

In BdG the hole block is built as -h.conj(), so it automatically carries
the conjugate (opposite) phase and exact particle-hole symmetry is
preserved by construction. The Zeeman magnitude is taken from the FULL |B|
with the empirical hole g = 2.2 and kept along sigma_x for every tilt
angle, both WITH and WITHOUT the Peierls phases: any with/without
difference at fixed |B| is therefore purely orbital (the g-tensor
anisotropy of the real device is treated separately in fig 11; the goal
here is to isolate the one effect the model previously omitted).

Sanity scale: Peierls phase per dx*dy = 5x5 nm plaquette at B_perp = 1 T is
e*B*dx*dy/hbar = 0.0380 rad.

Sections (each checkpointed to output/data/orbital.json; reruns resume):
  A  validation: (i) at B_perp = 0 the builder reproduces
     majorana_sim.build_wire_2d (matrix identity + lowest-|E| spectrum to
     < 1e-8 ueV); (ii) Hermiticity and exact particle-hole symmetry of the
     spectrum with the Peierls phases ON.
  B  gap vs tilt angle theta (theta = 0: B in plane along the wire;
     B_perp = |B| sin theta), |B| = 1 T, Delta_ind = 33 ueV (fig-8 centre
     operating scale), widths W = 10/20/40 nm, with vs without Peierls.
  C  the paper's preferred tilted SiB_meas design point (B = 0.333 T,
     theta = 38.6 deg — the fig-11 LK-tensor optimum — Delta_ind =
     11.3 ueV): orbital gap suppression vs strip width, and an estimate of
     the width below which the orbital correction stays < 10%.

mu convention: for each geometry mu is set to the kx = 0 energy of the
lowest transverse subband of the normal-state strip (computed from the
2Ny x 2Ny transverse Bloch Hamiltonian including the SOC-y hopping), so the
effective chemical potential of the lowest subband is 0 — the centre of the
topological lobe, as everywhere else in the study.

Usage:  python orbital.py --sec A|B|C|report|all  [--budget 150]
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from majorana_sim import (HBAR, ME, QE, UEV, s0, sx, sy,
                          EZ_J, solve_lowest, build_wire_2d,
                          subband_bottoms_ueV)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output", "data")
os.makedirs(DATA, exist_ok=True)
CKPT = os.path.join(DATA, "orbital.json")

# ---- hole-platform parameters (run_analysis fig8/10/11 conventions) -------
M_REL = 0.25            # hole effective mass
ALPHA = 0.06            # eV*A
G_EMP = 2.2             # empirical g magnitude, applied to the full |B|
NX, DX, DY = 300, 5e-9, 5e-9

# section B: strong-field tilt scan at the fig-8 centre pairing scale
B_FULL = 1.0            # T
DELTA_B = 33.0          # ueV
THETAS = [0, 15, 30, 45, 60, 75, 90]
NY_LIST = [2, 4, 8]     # W = Ny*dy = 10, 20, 40 nm at dy = 5 nm

# section C: preferred tilted SiB_meas operating point
#   key_numbers fig8 center_SiB_measured_args = (B = 0.33 T, Dind = 11.3 ueV)
#   key_numbers fig11 LK-tensor optimum at theta = 38.57 deg
B_C = 0.333             # T
THETA_C = 38.57         # deg
DELTA_C = 11.3          # ueV
# (dy_nm, Ny) -> W = Ny*dy: 10..40 nm on the 5 nm grid, 7.5..15 nm on 2.5 nm
WIDTHS_C = [(5.0, 2), (5.0, 3), (5.0, 4), (5.0, 5), (5.0, 6), (5.0, 8),
            (2.5, 3), (2.5, 4), (2.5, 5), (2.5, 6)]


# ------------------------------------------------------------ checkpointing
def _load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def _save(d):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, CKPT)


# ----------------------------------------------------------------- builder
def build_strip_orbital(Nx, Ny, dx, dy, mu_ueV, EZ_ueV, Delta_ueV,
                        alpha_eVA, m_rel, B_perp=0.0):
    """Clone of majorana_sim.build_wire_2d (that module is untouched) with
    (i) the Zeeman magnitude passed directly in ueV (term EZ*sigma_x, same
    form as build_wire_2d) and (ii) Peierls phases of the out-of-plane field
    component B_perp (T) on the x-hoppings, Landau gauge A = (-B_perp*y, 0):
    hop (n,m)->(n+1,m) multiplied by exp(i e B_perp y_m dx / hbar), y_m from
    the strip centre. Returns the sparse 4*Nx*Ny BdG Hamiltonian; the hole
    block -h.conj() carries the conjugate phase automatically (exact PHS)."""
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_ueV * UEV
    aSI = alpha_eVA * 1e-10 * QE
    tx = HBAR**2 / (2 * m * dx**2)
    ty = HBAR**2 / (2 * m * dy**2)
    Ns = Nx * Ny
    onsite = (2 * tx + 2 * ty - mu) * s0 + EZ * sx
    hop_x = -tx * s0 + 1j * (aSI / (2 * dx)) * sy   # from -alpha sigma_y k_x
    hop_y = -ty * s0 - 1j * (aSI / (2 * dy)) * sx   # from +alpha sigma_x k_y
    y = (np.arange(Ny) - 0.5 * (Ny - 1)) * dy       # centred transverse coord
    phase = np.exp(1j * (QE / HBAR) * B_perp * y * dx)
    Px = sp.kron(sp.diags(np.ones(Nx - 1), 1), sp.diags(phase), format="csr")
    Ky = sp.kron(sp.eye(Nx), sp.diags(np.ones(Ny - 1), 1), format="csr")
    hx = sp.kron(Px, hop_x).tocsr()
    hy = sp.kron(Ky, hop_y).tocsr()
    h = (sp.kron(sp.eye(Ns), onsite) + hx + hx.conj().T
         + hy + hy.conj().T).tocsr()
    Dm = sp.kron(sp.eye(Ns), D * (1j * sy)).tocsr()
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


def lowest_edge_ueV(Ny, dy, alpha_eVA=ALPHA, m_rel=M_REL):
    """kx = 0 energy (ueV) of the lowest transverse subband of the
    normal-state strip, no Zeeman: at kx = 0 the x-kinetic term contributes
    2tx - 2tx = 0 and the SOC-x hopping cancels, leaving the dense
    2Ny x 2Ny transverse problem with the SOC-y hopping. Setting mu to this
    value puts the effective chemical potential of band 1 at 0."""
    m = m_rel * ME
    ty = HBAR**2 / (2 * m * dy**2)
    aSI = alpha_eVA * 1e-10 * QE
    hop_y = -ty * s0 - 1j * (aSI / (2 * dy)) * sx
    K = np.diag(np.ones(Ny - 1), 1)
    Hp = (np.kron(np.eye(Ny), 2 * ty * s0) + np.kron(K, hop_y)
          + np.kron(K, hop_y).conj().T)
    return float(np.linalg.eigvalsh(Hp)[0] / UEV)


# ------------------------------------------------------------------ solving
def _x_end_weight(vec, Nx, Ny, frac=0.1):
    """Fraction of |psi|^2 in the outer frac of the wire LENGTH (x)."""
    half = vec.shape[0] // 2
    w = (np.abs(vec[:half])**2 + np.abs(vec[half:])**2)
    profile = w.reshape(Nx, 2 * Ny).sum(axis=1)
    n = max(1, int(frac * Nx))
    return float((profile[:n].sum() + profile[-n:].sum()) / profile.sum())


def solve_point(Ny, dy, EZ_ueV, Delta_ueV, B_perp, Nx=NX, dx=DX, k=8):
    """One strip diagonalization. Returns E0 (Majorana splitting, smallest
    |E|), gap (3rd-smallest |E| = excitation gap when a single near-zero
    pair exists), end weight of the lowest state, and the mu used."""
    mu = lowest_edge_ueV(Ny, dy)
    H = build_strip_orbital(Nx, Ny, dx, dy, mu, EZ_ueV, Delta_ueV,
                            ALPHA, M_REL, B_perp)
    E, V = solve_lowest(H, k=k)
    Eabs = np.abs(E) / UEV
    return dict(E0_ueV=round(float(Eabs[0]), 4),
                gap_ueV=round(float(Eabs[2]), 4),
                end_weight=round(_x_end_weight(V[:, 0], Nx, Ny), 3),
                mu_edge_ueV=round(mu, 2),
                B_perp_T=round(float(B_perp), 4))


# ---------------------------------------------------------------- section A
def sec_A(data, t0, budget):
    if data.get("A", {}).get("complete"):
        return True
    Ny, dy = 4, DY
    EZ = EZ_J(G_EMP, B_FULL) / UEV
    mu = lowest_edge_ueV(Ny, dy)
    out = dict(Nx=NX, Ny=Ny, dx_nm=DX * 1e9, dy_nm=dy * 1e9,
               mu_edge_ueV=round(mu, 4),
               mu_edge_noSOC_ueV=round(
                   float(subband_bottoms_ueV(Ny, dy, M_REL)[0]), 4),
               EZ_ueV=round(EZ, 4))
    # (a) B_perp = 0 vs build_wire_2d (identical parameters)
    H0 = build_strip_orbital(NX, Ny, DX, dy, mu, EZ, DELTA_B, ALPHA, M_REL,
                             B_perp=0.0)
    Hr = build_wire_2d(NX, Ny, DX, dy, mu, B_FULL, DELTA_B, ALPHA, M_REL,
                       G_EMP)
    dH = (H0 - Hr).tocsr()
    out["max_matrix_diff_ueV"] = float(
        np.abs(dH.data).max() / UEV) if dH.nnz else 0.0
    E0, _ = solve_lowest(H0, k=8)
    Er, _ = solve_lowest(Hr, k=8)
    out["max_spectrum_diff_ueV"] = float(
        np.max(np.abs(np.sort(E0) - np.sort(Er))) / UEV)
    out["spectrum_match_1e-8"] = bool(out["max_spectrum_diff_ueV"] < 1e-8)
    # (b) Hermiticity + exact PHS with Peierls phases ON (B_perp = 0.5 T)
    Hb = build_strip_orbital(NX, Ny, DX, dy, mu, EZ, DELTA_B, ALPHA, M_REL,
                             B_perp=0.5)
    dAH = (Hb - Hb.conj().T).tocsr()
    out["max_nonhermiticity_ueV"] = float(
        np.abs(dAH.data).max() / UEV) if dAH.nnz else 0.0
    Eb, _ = solve_lowest(Hb, k=8)
    Es = np.sort(Eb)
    out["max_PHS_asymmetry_ueV"] = float(
        np.max(np.abs(Es + Es[::-1])) / UEV)
    # (c) phase scale sanity: 5x5 nm plaquette at 1 T
    out["plaquette_phase_rad_at_1T_5x5nm"] = float(QE * 1.0 * DX * DY / HBAR)
    out["complete"] = True
    data["A"] = out
    _save(data)
    print("[A]", json.dumps(out, indent=1))
    return True


# ---------------------------------------------------------------- section B
def sec_B(data, t0, budget):
    B = data.setdefault("B", {})
    EZ = EZ_J(G_EMP, B_FULL) / UEV
    B.setdefault("params", dict(B_T=B_FULL, EZ_ueV=round(EZ, 3),
                                Delta_ind_ueV=DELTA_B, alpha_eVA=ALPHA,
                                m_rel=M_REL, g=G_EMP, Nx=NX, dx_nm=5.0,
                                dy_nm=5.0, note=(
        "E_Z fixed from |B| (g=2.2, along sigma_x) for ALL theta in both "
        "columns; theta enters only through B_perp = |B| sin(theta) in the "
        "Peierls phases, so with-vs-without differences are purely orbital. "
        "The no-orbital value is theta-independent by construction "
        "(identical Hamiltonian).")))
    for Ny in NY_LIST:
        key = "W%dnm" % (Ny * 5)
        rec = B.setdefault(key, {})
        if "no_orbital" not in rec:
            rec["no_orbital"] = solve_point(Ny, DY, EZ, DELTA_B, 0.0)
            _save(data)
            print("[B]", key, "no_orbital", rec["no_orbital"])
            if time.time() - t0 > budget:
                return False
        for th in THETAS:
            tkey = "theta%d" % th
            if tkey in rec:
                continue
            Bp = B_FULL * np.sin(np.radians(th))
            rec[tkey] = solve_point(Ny, DY, EZ, DELTA_B, Bp)
            _save(data)
            print("[B]", key, tkey, rec[tkey])
            if time.time() - t0 > budget:
                return False
    B["complete"] = True
    _save(data)
    return True


# ---------------------------------------------------------------- section C
def sec_C(data, t0, budget):
    C = data.setdefault("C", {})
    EZ = EZ_J(G_EMP, B_C) / UEV
    Bp = B_C * np.sin(np.radians(THETA_C))
    C.setdefault("params", dict(B_T=B_C, theta_deg=THETA_C,
                                B_perp_T=round(float(Bp), 4),
                                EZ_ueV=round(EZ, 3), Delta_ind_ueV=DELTA_C,
                                alpha_eVA=ALPHA, m_rel=M_REL, g=G_EMP,
                                Nx=NX, dx_nm=5.0, note=(
        "Preferred tilted SiB_meas operating point: B=0.333 T, "
        "Delta_ind=11.3 ueV (key_numbers fig8 center_SiB_measured_args), "
        "tilt theta=38.57 deg (fig11 LK-tensor optimum). Suppression = "
        "1 - gap(orbital)/gap(no orbital) at fixed E_Z from |B|.")))
    for dynm, Ny in WIDTHS_C:
        key = "W%gnm_dy%gnm" % (dynm * Ny, dynm)
        rec = C.setdefault(key, {})
        for lbl, bp in (("no_orbital", 0.0), ("orbital", float(Bp))):
            if lbl in rec:
                continue
            rec[lbl] = solve_point(Ny, dynm * 1e-9, EZ, DELTA_C, bp)
            _save(data)
            print("[C]", key, lbl, rec[lbl])
            if time.time() - t0 > budget:
                return False
        g0 = rec["no_orbital"]["gap_ueV"]
        g1 = rec["orbital"]["gap_ueV"]
        rec["W_nm"] = dynm * Ny
        rec["suppression_pct"] = round(100.0 * (1.0 - g1 / g0), 3)
        _save(data)
    # width where the orbital suppression reaches 10% (log-log interp + fit)
    pts = {}
    for dynm, Ny in WIDTHS_C:
        key = "W%gnm_dy%gnm" % (dynm * Ny, dynm)
        W = dynm * Ny
        s = C[key]["suppression_pct"]
        if W not in pts or dynm < pts[W][0]:    # prefer the finer grid
            pts[W] = (dynm, s)
    Ws = np.array(sorted(pts))
    Ss = np.array([pts[w][1] for w in Ws])
    thr = None
    for i in range(1, len(Ws)):
        if Ss[i - 1] < 10.0 <= Ss[i] and Ss[i - 1] > 0:
            lw = (np.log(10.0 / Ss[i - 1]) / np.log(Ss[i] / Ss[i - 1])
                  * np.log(Ws[i] / Ws[i - 1]) + np.log(Ws[i - 1]))
            thr = float(np.exp(lw))
            break
    sel = Ss > 0.0
    fit = None
    if sel.sum() >= 2:
        p, c = np.polyfit(np.log(Ws[sel]), np.log(Ss[sel]), 1)
        fit = dict(power=round(float(p), 2),
                   W10pct_nm=round(float(np.exp((np.log(10.0) - c) / p)), 1))
        if thr is None:
            fit["note"] = ("no 10% crossing inside the sampled 7.5-40 nm "
                           "range; W10pct_nm is a power-law extrapolation "
                           "far outside it (single-subband model)")
    C["suppression_vs_W"] = {("%gnm" % w): pts[w][1] for w in Ws}
    C["W_10pct_interp_nm"] = round(thr, 2) if thr else None
    C["powerlaw_fit"] = fit
    C["complete"] = True
    _save(data)
    print("[C] suppression_vs_W", C["suppression_vs_W"],
          "W_10pct", C["W_10pct_interp_nm"], "fit", fit)
    return True


# ------------------------------------------------------------------- report
def report(data):
    if not all(data.get(s, {}).get("complete") for s in "ABC"):
        print("sections incomplete; run --sec all first")
        return False
    A, B, C = data["A"], data["B"], data["C"]
    btab = {}
    for Ny in NY_LIST:
        key = "W%dnm" % (Ny * 5)
        rec = B[key]
        btab[key] = dict(
            no_orbital_gap_ueV=rec["no_orbital"]["gap_ueV"],
            with_orbital_gap_ueV={("theta%d" % th):
                                  rec["theta%d" % th]["gap_ueV"]
                                  for th in THETAS},
            suppression_pct_at_theta90=round(
                100 * (1 - rec["theta90"]["gap_ueV"]
                       / rec["no_orbital"]["gap_ueV"]), 1))
    summary = dict(
        model=("2D strip + Peierls phases of B_perp on x-hoppings (Landau "
               "gauge, y from centre); E_Z from full |B|, g=2.2; orbital "
               "effect isolated as with-vs-without Peierls at fixed E_Z"),
        validation=dict(
            spectrum_match_B0_ueV=A["max_spectrum_diff_ueV"],
            nonhermiticity_ueV=A["max_nonhermiticity_ueV"],
            PHS_asymmetry_ueV=A["max_PHS_asymmetry_ueV"],
            plaquette_phase_rad_1T=round(
                A["plaquette_phase_rad_at_1T_5x5nm"], 5)),
        tilt_scan_B1T_D33=btab,
        design_point=dict(C["params"],
                          suppression_pct_vs_W=C["suppression_vs_W"]),
        W_orbital_10pct_nm=C["W_10pct_interp_nm"],
        powerlaw=C["powerlaw_fit"],
    )
    import run_analysis
    run_analysis.save_numbers("orbital", summary)
    return True


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all")
    ap.add_argument("--budget", type=float, default=150.0,
                    help="wall-clock budget (s); checkpoints + resumes")
    args = ap.parse_args()
    t0 = time.time()
    data = _load()
    todo = ["A", "B", "C", "report"] if args.sec == "all" else [args.sec]
    done = True
    for s in todo:
        if s == "A":
            done = sec_A(data, t0, args.budget)
        elif s == "B":
            done = sec_B(data, t0, args.budget)
        elif s == "C":
            done = sec_C(data, t0, args.budget)
        elif s == "report":
            done = report(data)
        else:
            raise SystemExit("unknown --sec %r" % s)
        if not done:
            break
    print("STATUS:", "COMPLETE" if done else
          "INCOMPLETE (budget hit) — rerun the same command to resume",
          "| elapsed %.1f s" % (time.time() - t0))
