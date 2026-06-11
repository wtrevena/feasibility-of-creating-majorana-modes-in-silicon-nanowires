"""
r6_transport.py
===============
Review-round-6 item A2: transport realism for the HOLE platform.

Reuses the validated RGF engine of transport.py UNCHANGED (Ando wave-matched
normal leads, Caroli total transmission, gauge-fixed sign-det-r invariant):
the hole wire is the SAME 4-orbital single-channel Lutchyn-Oreg chain, only
with hole-band parameters (m*, alpha, g) and the in-plane field, so every
function imports directly.  New here: an arbitrary-onsite-potential closed
wire builder (reusing majorana_sim._normal_h, verified element-by-element
against the transport cell chain), finite-T/Lorentzian smearing of T(E),
and the trivial-ABS failure-mode library.

Sections (each checkpoints output/data/r6_transport.json and the ledger tag
"r6_transport" in key_numbers.json; run:  python r6_transport.py --sec H|F|X|all)

H  Hole-scenario transport at three operating points (L=2 um, dx=2.5 nm):
     a  [110] six-band SiB_pauli (Ez=30 MV/m): m*=0.204, alpha=0.073 eV*A,
        g_x'=1.657, B=1.0 T, Dind=24.7 ueV, mu=0  (platform110 Ez30_SiB_pauli,
        static spectral gap 23.25 ueV)
     b  [110] six-band Al (Ez=10 MV/m): m*=0.194, alpha=0.052 eV*A,
        g_x'=1.549, B=1.753 T, Dind=42 ueV        (platform110 Ez10_Al, 36.57)
     c  old empirical hole tensor: m*=0.25, alpha=0.06 eV*A, g=2.2,
        B=1.0 T, Dind=33 ueV                      (fig8 center_a006_g22, 30.6)
   For each: k_so*dx check, clean transport gap E_T (T > 0.01) vs analytic
   bulk gap vs closed-wire (E0, E2, end weight), T(E=0), and the invariant
   Q = sign(det r * det r_ref) at 0.6/0.9/1.1 x B* and at B_op (lead held at
   B_op, reference region B=0, exactly as in transport.py).

F  Finite-T / finite-broadening conductance at point (a): clean and mildly
   disordered (W=200 ueV, 3 seeds) T(E) convolved with the thermal kernel
   (-df/dE) at T in {20,50,100} mK (x) Lorentzian of HWHM gamma in
   {0.5,2} ueV.  Total T(E) is even in E by particle-hole symmetry of the
   BdG S-matrix, so T(|E|) is mirrored before convolving.  Reported per
   combo: G(0), G(0)/plateau, apparent gap at 10% and 50% of the above-gap
   plateau (plateau = mean G on 40-60 ueV), vs the true transport gap.

X  Failure-mode library on the point-(a) wire, trivial side (B = 0.6 B*,
   B* = Dind/(g muB / 2) at mu=0) AND topological side (B = B_op = 1.0 T):
     dot_V-150 / dot_V-250 : smooth end quantum dot, half-cosine well,
                             depth V0 ueV over 60 nm at the left end
     ramp_300ueV_150nm     : smooth confinement, +300 ueV tanh barrier
                             (w = 40 nm) over the first/last 150 nm
     disorder_W600         : W = 600 ueV; the worst (lowest closed-wire E0)
                             of 10 seeds, default_rng([96, 6, 1, r])
   For each: closed-wire E0, E2, end weight (majorana_metrics), T(E=0), and
   sign det r -- which diagnostics flag the mimic as trivial.

Caveats: two-terminal NS(N) transport only; NONLOCAL (left lead -> right
lead) conductance and full G(V,B) maps are NOT implemented here -- the
paper must scope transport claims to local two-terminal spectroscopy.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from majorana_sim import (HBAR, ME, QE, UEV, MU_B_EV, sy, EZ_J, _normal_h,
                          build_wire, bulk_gap_ueV, majorana_metrics)
from transport import (cell_blocks, lead_data, lead_sigmas,
                       total_transmission, transport_gap,
                       reflection_invariant, self_tests,
                       MU_LEAD, T_THRESHOLD, TAUZ4)
from run_analysis import save_numbers

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output", "data")
CKPT = os.path.join(DATA, "r6_transport.json")

DX = 2.5e-9                       # m  (k_so*dx < 0.1 verified per point)
L = 2e-6                          # m
N = int(round(L / DX))            # 800 cells
KB_UEV_PER_MK = 8.617333262e-5 * 1e3   # kB in ueV/mK  (= 0.08617)

POINTS = {
    "a_110_SiB_pauli": dict(m=0.204, al=0.073, g=1.657, B=1.0, Dind=24.7,
                            mu=0.0, src="platform110 Ez30_SiB_pauli, "
                                        "static gap 23.25 ueV"),
    "b_110_Al":        dict(m=0.194, al=0.052, g=1.549, B=1.753, Dind=42.0,
                            mu=0.0, src="platform110 Ez10_Al, "
                                        "static gap 36.57 ueV"),
    "c_old_empirical": dict(m=0.25, al=0.06, g=2.2, B=1.0, Dind=33.0,
                            mu=0.0, src="fig8 center_a006_g22, "
                                        "static gap 30.6 ueV"),
}


def _load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def _save(ck):
    os.makedirs(DATA, exist_ok=True)
    with open(CKPT, "w") as f:
        json.dump(ck, f, indent=2)


def Bstar_T(g, Dind, mu=0.0):
    """Clean phase boundary: EZ(B*) = sqrt(Dind^2 + mu^2)."""
    return float(np.sqrt(Dind**2 + mu**2) / (0.5 * g * MU_B_EV * 1e6))


def build_wire_profile(Nc, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
                       V_ueV):
    """majorana_sim.build_wire with an ARBITRARY onsite potential profile
    (ueV per site); +V tau_z convention, identical to build_wire's disorder
    and transport.py's Vdis.  Reuses _normal_h, so the basis is the standard
    [u (site x spin), v (site x spin)] expected by majorana_metrics."""
    h = _normal_h(Nc, dx, mu_ueV * UEV, EZ_J(g, B), alpha_eVA * 1e-10 * QE,
                  m_rel * ME, np.asarray(V_ueV, dtype=float) * UEV)
    Dm = sp.kron(sp.eye(Nc), Delta_ueV * UEV * (1j * sy)).tocsr()
    return sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")


def verify_profile_assembly(p, Nv=40):
    """build_wire_profile vs the transport cell chain, element by element
    (after the site-major permutation), for a smooth+noisy profile.
    Mirrors transport.verify_cell_blocks.  Returns max |dH| in ueV."""
    rng = np.random.default_rng(5)
    V = 100.0 * np.sin(np.linspace(0, 3, Nv)) + rng.uniform(-50, 50, Nv)
    U, Vh = cell_blocks(DX, p["mu"], p["B"], p["Dind"], p["al"],
                        p["m"], p["g"])
    Hp = build_wire_profile(Nv, DX, p["mu"], p["B"], p["Dind"], p["al"],
                            p["m"], p["g"], V).toarray() / UEV
    perm = np.empty(4 * Nv, dtype=int)
    for n in range(Nv):
        perm[4 * n + 0] = 2 * n
        perm[4 * n + 1] = 2 * n + 1
        perm[4 * n + 2] = 2 * Nv + 2 * n
        perm[4 * n + 3] = 2 * Nv + 2 * n + 1
    Hc = np.zeros((4 * Nv, 4 * Nv), dtype=complex)
    for n in range(Nv):
        Hc[4*n:4*n+4, 4*n:4*n+4] = U + V[n] * TAUZ4
        if n < Nv - 1:
            Hc[4*n:4*n+4, 4*n+4:4*n+8] = Vh
            Hc[4*n+4:4*n+8, 4*n:4*n+4] = Vh.conj().T
    return float(np.abs(Hc - Hp[np.ix_(perm, perm)]).max())


# -------------------------------------------------------------- section H
def run_point(tag, p):
    t0 = time.time()
    m, al, g, B, Dind, mu = (p["m"], p["al"], p["g"], p["B"], p["Dind"],
                             p["mu"])
    kso_dx = float((m * ME) * (al * 1e-10 * QE) / HBAR**2 * DX)
    U_wire, Vh = cell_blocks(DX, mu, B, Dind, al, m, g)
    U_lead, Vh_l = cell_blocks(DX, MU_LEAD, B, 0.0, al, m, g)
    assert np.abs(Vh - Vh_l).max() == 0.0
    Eg = np.arange(0.0, 60.0 + 1e-9, 0.5)
    SigL, SigR, GamL, GamR = lead_sigmas(Eg, U_lead, Vh)
    T = total_transmission(Eg, np.zeros(N), U_wire, Vh,
                           SigL, SigR, GamL, GamR)
    ET, cens = transport_gap(Eg, T)
    bulk = bulk_gap_ueV(mu, B, Dind, al, m, g)
    E0, E2, ew = majorana_metrics(build_wire(N, DX, mu, B, Dind, al, m, g), N)
    # invariant: lead fixed at B_op, reference region at B=0 (trivial)
    ld0 = lead_data(0.0, U_lead, Vh)
    U_ref, _ = cell_blocks(DX, mu, 0.0, Dind, al, m, g)
    ref = reflection_invariant(np.zeros(N), U_ref, Vh, ld0)
    sgn = float(np.sign(ref["det_r"]))
    Bs = Bstar_T(g, Dind, mu)
    inv = {}
    for lab, Bx in (("0.6Bstar", 0.6 * Bs), ("0.9Bstar", 0.9 * Bs),
                    ("1.1Bstar", 1.1 * Bs), ("B_op", B)):
        Ux, _ = cell_blocks(DX, mu, Bx, Dind, al, m, g)
        q = reflection_invariant(np.zeros(N), Ux, Vh, ld0, ref_sign=sgn)
        inv[lab] = dict(B_T=round(Bx, 4), Q=q["Q"],
                        det_r=round(q["det_r"], 4))
    out = dict(params=dict(m=m, alpha_eVA=al, g=g, B_T=B, Dind_ueV=Dind,
                           mu_ueV=mu, src=p["src"]),
               kso_dx=round(kso_dx, 4),
               B_star_T=round(Bs, 4),
               transport_gap_ueV=round(ET, 2),
               transport_gap_censored=bool(cens),
               bulk_gap_ueV=round(bulk, 2),
               closed_wire=dict(E0_ueV=round(E0, 4), E2_ueV=round(E2, 2),
                                end_weight=round(ew, 3)),
               T_at_E0=float(f"{T[0]:.4e}"),
               invariant=inv,
               runtime_s=round(time.time() - t0, 1))
    print(f"[H:{tag}] kso*dx={kso_dx:.3f}  E_T={ET:.2f}  bulk={bulk:.2f}  "
          f"E2={E2:.2f}  T(0)={T[0]:.3f}  "
          f"Q: " + ", ".join(f"{k}={v['Q']:+d}" for k, v in inv.items()),
          flush=True)
    return out


def sec_H():
    t0 = time.time()
    pa = POINTS["a_110_SiB_pauli"]
    U_lead, Vh = cell_blocks(DX, MU_LEAD, pa["B"], 0.0, pa["al"], pa["m"],
                             pa["g"])
    tests = self_tests(U_lead, Vh)               # hole-parameter lead checks
    vprof = verify_profile_assembly(pa)
    print(f"[H] hole-lead self-tests: {tests}", flush=True)
    print(f"[H] profile-assembly check (max |dH|): {vprof:.2e} ueV",
          flush=True)
    out = {"verification": {**{k: float(f"{v:.3e}") for k, v in tests.items()},
                            "profile_vs_cell_chain_maxdH_ueV":
                                float(f"{vprof:.3e}")}}
    for tag, p in POINTS.items():
        out[tag] = run_point(tag, p)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# -------------------------------------------------------------- section F
def sec_F():
    t0 = time.time()
    p = POINTS["a_110_SiB_pauli"]
    m, al, g, B, Dind, mu = (p["m"], p["al"], p["g"], p["B"], p["Dind"],
                             p["mu"])
    dE = 0.25
    Eg = np.arange(0.0, 120.0 + 1e-9, dE)
    U_wire, Vh = cell_blocks(DX, mu, B, Dind, al, m, g)
    U_lead, _ = cell_blocks(DX, MU_LEAD, B, 0.0, al, m, g)
    SigL, SigR, GamL, GamR = lead_sigmas(Eg, U_lead, Vh)
    curves = {"clean": np.zeros(N)}
    for r in range(3):
        rng = np.random.default_rng([96, 6, 0, r])
        curves[f"W200_seed{r}"] = rng.uniform(-200.0, 200.0, N)
    Ek = np.arange(-60.0, 60.0 + 1e-9, dE)
    ipl = (Eg >= 40.0) & (Eg <= 60.0)
    out = {"note": "G(V) = T(E) (*) [-df/dE](T) (*) Lorentzian(gamma); "
                   "T(E) mirrored to E<0 by PHS; plateau = mean G on "
                   "40-60 ueV; apparent gaps = first V with G > 0.1 / 0.5 "
                   "of the plateau (transport_gap interpolation)",
           "dE_ueV": dE, "curves": {}}
    for name, V in curves.items():
        T = total_transmission(Eg, V, U_wire, Vh, SigL, SigR, GamL, GamR)
        ET, cens = transport_gap(Eg, T)
        T_sym = np.concatenate([T[:0:-1], T])     # even in E (PHS)
        entry = dict(true_transport_gap_ueV=round(ET, 2),
                     censored=bool(cens),
                     T_at_E0=float(f"{T[0]:.3e}"),
                     plateau_raw=round(float(np.mean(T[ipl])), 4),
                     smeared={})
        for Tmk in (20, 50, 100):
            kT = KB_UEV_PER_MK * Tmk
            kth = 0.25 / kT / np.cosh(Ek / (2.0 * kT))**2
            for gam in (0.5, 2.0):
                klo = (gam / np.pi) / (Ek**2 + gam**2)
                ker = np.convolve(kth, klo, mode="same") * dE
                ker = ker / (ker.sum() * dE)
                G = np.convolve(T_sym, ker, mode="same") * dE
                Gp = G[len(Eg) - 1:]              # V >= 0 half
                plat = float(np.mean(Gp[ipl]))
                g10, c10 = transport_gap(Eg, Gp, thr=0.1 * plat)
                g50, c50 = transport_gap(Eg, Gp, thr=0.5 * plat)
                entry["smeared"][f"{Tmk}mK_g{gam}"] = dict(
                    kT_ueV=round(kT, 3),
                    G0=float(f"{Gp[0]:.4e}"),
                    G0_over_plateau=float(f"{Gp[0] / plat:.4e}"),
                    plateau=round(plat, 4),
                    app_gap_10pct_ueV=(round(g10, 2) if not c10 else None),
                    app_gap_half_ueV=(round(g50, 2) if not c50 else None))
        out["curves"][name] = entry
        s50 = entry["smeared"]["50mK_g0.5"]
        print(f"[F:{name}] E_T={ET:.2f}  50mK/0.5ueV: G0/plat="
              f"{s50['G0_over_plateau']:.3f}  app10%={s50['app_gap_10pct_ueV']}"
              f"  app50%={s50['app_gap_half_ueV']}", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# -------------------------------------------------------------- section X
def sec_X():
    t0 = time.time()
    p = POINTS["a_110_SiB_pauli"]
    m, al, g, Dind, mu = p["m"], p["al"], p["g"], p["Dind"], p["mu"]
    Bs = Bstar_T(g, Dind, mu)
    x = (np.arange(N) + 0.5) * DX
    profiles = {}
    Ldot = 60e-9
    for V0 in (-150.0, -250.0):
        profiles[f"dot_V{int(V0)}"] = np.where(
            x < Ldot, V0 * 0.5 * (1.0 + np.cos(np.pi * x / Ldot)), 0.0)
    Vc, x0, w = 300.0, 150e-9, 40e-9
    profiles["ramp_300ueV_150nm"] = (
        Vc * 0.5 * (1.0 - np.tanh((x - x0) / w))
        + Vc * 0.5 * (1.0 + np.tanh((x - (L - x0)) / w)))
    out = {"B_star_T": round(Bs, 4),
           "note": "point-(a) wire; trivial side B=0.6*B_star (prescribed) plus "
                   "a near-threshold 0.9*B_star supplement (classic ABS-mimic "
                   "regime), topological side B=1.0 T (operating point); "
                   "disorder seed = lowest "
                   "closed-wire E0 of 10 (default_rng([96,6,1,r]), W=600); "
                   "E2 in the trivial phase is the second excitation, not a "
                   "Majorana splitting",
           "phases": {}}
    for phase, Bx in (("trivial_0.6Bstar", 0.6 * Bs),
                      ("trivial_0.9Bstar", 0.9 * Bs),
                      ("topological_B1.0T", 1.0)):
        U_wire, Vh = cell_blocks(DX, mu, Bx, Dind, al, m, g)
        U_lead, _ = cell_blocks(DX, MU_LEAD, Bx, 0.0, al, m, g)
        ld0 = lead_data(0.0, U_lead, Vh)
        SigL, SigR, GamL, GamR = lead_sigmas(np.array([0.0]), U_lead, Vh)
        U_ref, _ = cell_blocks(DX, mu, 0.0, Dind, al, m, g)
        sgn = float(np.sign(
            reflection_invariant(np.zeros(N), U_ref, Vh, ld0)["det_r"]))
        E0s, Vds = [], []
        for r in range(10):
            rng = np.random.default_rng([96, 6, 1, r])
            Vd = rng.uniform(-600.0, 600.0, N)
            Vds.append(Vd)
            H = build_wire_profile(N, DX, mu, Bx, Dind, al, m, g, Vd)
            E0s.append(majorana_metrics(H, N)[0])
        rbest = int(np.argmin(E0s))
        cases = {"clean": np.zeros(N), **profiles,
                 "disorder_W600": Vds[rbest]}
        res = {}
        for name, V in cases.items():
            H = build_wire_profile(N, DX, mu, Bx, Dind, al, m, g, V)
            E0, E2, ew = majorana_metrics(H, N)
            T0 = float(total_transmission(np.array([0.0]), V, U_wire, Vh,
                                          SigL, SigR, GamL, GamR)[0])
            q = reflection_invariant(V, U_wire, Vh, ld0, ref_sign=sgn)
            res[name] = dict(E0_ueV=round(E0, 3), E2_ueV=round(E2, 2),
                             end_weight=round(ew, 3),
                             T_at_E0=float(f"{T0:.3e}"),
                             Q=q["Q"], det_r=round(q["det_r"], 4))
            print(f"[X:{phase}:{name}] E0={E0:.3f}  E2={E2:.2f}  "
                  f"ew={ew:.3f}  T(0)={T0:.3e}  Q={q['Q']:+d} "
                  f"(det r={q['det_r']:+.3f})", flush=True)
        out["phases"][phase] = dict(
            B_T=round(Bx, 4),
            bulk_gap_clean_ueV=round(bulk_gap_ueV(mu, Bx, Dind, al, m, g), 2),
            disorder_E0_all_seeds_ueV=[round(e, 3) for e in E0s],
            disorder_worst_seed=rbest,
            cases=res)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all", choices=["H", "F", "X", "all"])
    args = ap.parse_args()
    ck = _load()
    if "meta" not in ck:
        ck["meta"] = dict(
            module="r6_transport.py (review round 6, item A2)",
            method="RGF (Caroli) + Ando wave-matched normal leads + "
                   "gauge-fixed sign-det-r invariant, all imported from the "
                   "validated transport.py; closed-wire metrics via "
                   "majorana_sim (profile builder verified against the "
                   "transport cell chain)",
            geometry=f"NSN, L={L*1e6:.0f} um, dx={DX*1e9:.1f} nm, N={N}, "
                     f"mu_lead={MU_LEAD:.0f} ueV, threshold T>{T_THRESHOLD}",
            caveat="two-terminal local NS(N) transport only; nonlocal "
                   "(left->right) conductance and G(V,B) maps NOT "
                   "implemented -- scope paper claims accordingly")
        _save(ck)
    if args.sec in ("H", "all"):
        ck["H"] = sec_H()
        _save(ck)
        save_numbers("r6_transport", {"H": {
            tag: dict(kso_dx=ck["H"][tag]["kso_dx"],
                      E_T_ueV=ck["H"][tag]["transport_gap_ueV"],
                      bulk_ueV=ck["H"][tag]["bulk_gap_ueV"],
                      E2_ueV=ck["H"][tag]["closed_wire"]["E2_ueV"],
                      Q_op=ck["H"][tag]["invariant"]["B_op"]["Q"],
                      Q_06Bstar=ck["H"][tag]["invariant"]["0.6Bstar"]["Q"])
            for tag in POINTS}})
    if args.sec in ("F", "all"):
        ck["F"] = sec_F()
        _save(ck)
        cl = ck["F"]["curves"]["clean"]
        save_numbers("r6_transport", {"F_clean": dict(
            true_gap_ueV=cl["true_transport_gap_ueV"],
            **{k: dict(G0_over_plateau=v["G0_over_plateau"],
                       app10=v["app_gap_10pct_ueV"],
                       app50=v["app_gap_half_ueV"])
               for k, v in cl["smeared"].items()})})
    if args.sec in ("X", "all"):
        ck["X"] = sec_X()
        _save(ck)
        save_numbers("r6_transport", {"X": {
            ph: {nm: dict(E0=c["E0_ueV"], E2=c["E2_ueV"], Q=c["Q"],
                          T0=c["T_at_E0"])
                 for nm, c in d["cases"].items()}
            for ph, d in ck["X"]["phases"].items()}})
    print("r6_transport done:", args.sec, flush=True)


if __name__ == "__main__":
    main()
# end of r6_transport.py
