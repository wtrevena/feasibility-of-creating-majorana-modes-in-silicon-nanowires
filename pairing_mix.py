"""
pairing_mix.py — review round 5, item R5-3 (referee-requested control).

Continuous interpolation between pure INTER-valley pairing (the physical
channel for a uniform s-wave parent: valleys at +-k0 are time-reversed
partners, so zero-momentum pairing is inter-valley) and INTRA-valley
pairing (induced only by interface momentum scattering, since an
intra-valley pair carries momentum 2*k0 that a uniform parent cannot
supply):

    D(eta) = Delta * [ sqrt(1-eta^2) * (nu_x  (x) i sigma_y)
                       +        eta  * (nu_0  (x) i sigma_y) ],   eta in [0,1].

Both terms are antisymmetric (D^T = -D), so particle-hole symmetry is exact
for every eta, and tr D^dag D is eta-independent (total pairing weight
preserved). eta is the intra-valley pairing FRACTION (amplitude).

Algebra worth knowing before reading the numbers: for a UNIFORM valley-orbit
phase phi=0 (lam*nu_x onsite), rotating to the valley-split band basis maps
nu_x -> nu_z, nu_0 -> nu_0, so the two bands at mu_eff = mu -+ |lam| see
effective pairings Delta*(sqrt(1-eta^2) + eta) [topological band, mu_eff =
mu - lam] and Delta*(eta - sqrt(1-eta^2)) [trivial band, mu_eff = mu + lam]:
the trivial band's gap closes at eta = 1/sqrt(2). Crucially, the intra-valley
(nu_0) term is invariant under valley rotations and therefore does NOT see
the valley-orbit phase phi(x); only the inter-valley component experiences
the step-junction phase jump.

Sections (each checkpoints output/data/pairing_mix.json + key_numbers.json):
  A  validation: eta=0 reproduces build_wire_two_valley_iv exactly
     (matrix identity + lowest-6 |E| to 1e-9 ueV, clean AND single-step).
  B  clean wire at the wedge point: E0, E2, end weight vs eta.
  C  single-step wire (dphi = 0.85*pi at center): same table; threshold eta
     at which the 4.03 ueV step bound state weakens >2x or changes character.
  D  50-nm Poisson same-sign staircase ensemble (8 seeds, fig9 seed family
     [71, 1, 50, s]) at one eta beyond the threshold: median E2.

Usage:  python pairing_mix.py --sec A|B|C|D [--eta 0.5]
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from majorana_sim import (
    HBAR, ME, QE, UEV, EZ_J, s0, sx, sy, sz,
    build_wire_two_valley_iv, solve_lowest, site_density, end_weight,
    majorana_metrics,
)
from run_analysis import (
    save_numbers, DATA, _make_vo_profile, M_SI, G_SI, DELTA, DX, ALPHA_DEMO,
)

CKPT = os.path.join(DATA, "pairing_mix.json")

# fig9 wedge-point baseline (run_analysis.fig9)
MU, B_T, LAM, N = 35.0, 1.5, 75.0, 500          # ueV, T, ueV, sites (dx = DX)
ETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DPHI_STEP = 0.85 * np.pi


# ------------------------------------------------------------------ builder
def build_wire_two_valley_mix(N, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel,
                              g, eta, vo_profile_ueV=None,
                              valley_pol_ueV=0.0,
                              disorder_ueV=0.0, rng=None):
    """Mirror of majorana_sim.build_wire_two_valley_iv (soc_mode='rashba'
    only), with the mixed pairing D(eta) defined in the module docstring.
    eta=0 reproduces build_wire_two_valley_iv bit-for-bit (validated, sec A).
    """
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    if vo_profile_ueV is None:
        lam = np.zeros(N, dtype=complex)
    else:
        lam = np.asarray(vo_profile_ueV, dtype=complex) * UEV
    Epol = valley_pol_ueV * UEV
    t = HBAR**2 / (2 * m * dx**2)
    aso = aSI / (2 * dx)
    if disorder_ueV:
        if rng is None:
            raise ValueError("disorder_ueV > 0 requires an explicit rng")
        V = rng.uniform(-disorder_ueV, disorder_ueV, N) * UEV
    else:
        V = np.zeros(N)
    v0 = np.eye(2)
    vx = np.array([[0, 1], [1, 0]], dtype=complex)
    vy = np.array([[0, -1j], [1j, 0]])
    vz = np.diag([1.0, -1.0]).astype(complex)
    onsite_spin = (2 * t - mu) * s0 + EZ * sx
    diag_blocks = [np.kron(v0, onsite_spin) + V[n] * np.kron(v0, s0)
                   + np.kron(lam[n].real * vx + lam[n].imag * vy, s0)
                   + 0.5 * Epol * np.kron(vz, s0)
                   for n in range(N)]
    h = sp.block_diag(diag_blocks, format="lil")
    K = sp.diags(np.ones(N - 1), 1, format="csr")
    hop_kin = np.kron(v0, -t * s0)
    h = (h + sp.kron(K, hop_kin) + sp.kron(K.T, hop_kin.conj().T)).tolil()
    hop_soc = np.kron(v0, -1j * aso * sz)
    h = (h.tocsr() + sp.kron(K, hop_soc)
         + sp.kron(K.T, hop_soc.conj().T)).tocsr()
    # mixed pairing block: D^T = -D termwise; tr D^dag D independent of eta
    Dpair = D * (np.sqrt(1.0 - eta**2) * np.kron(vx, 1j * sy)
                 + eta * np.kron(v0, 1j * sy))
    Dm = sp.kron(sp.eye(N), Dpair).tocsr()
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


# ------------------------------------------------------------------ helpers
def vo_clean():
    return LAM * np.ones(N, dtype=complex)


def vo_step(dphi=DPHI_STEP):
    half = N // 2
    return LAM * np.exp(1j * np.where(np.arange(N) < half, 0.0, dphi))


def H_mix(eta, vo):
    return build_wire_two_valley_mix(N, DX, MU, B_T, DELTA, ALPHA_DEMO,
                                     M_SI, G_SI, eta, vo_profile_ueV=vo)


def metrics(eta, vo):
    """(E0_ueV, E2_ueV, end_weight) via majorana_sim.majorana_metrics."""
    return majorana_metrics(H_mix(eta, vo), N, k=6)


def ckpt_load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def ckpt_save(sec, d):
    alld = ckpt_load()
    alld[sec] = d
    with open(CKPT, "w") as f:
        json.dump(alld, f, indent=2)
    print(f"[pairing_mix:{sec}] checkpointed -> {CKPT}")


def lowest6_ueV(H, seed=0):
    np.random.seed(seed)              # deterministic eigsh start vector
    E, _ = solve_lowest(H, k=6)
    return np.sort(np.abs(E)) / UEV


# ----------------------------------------------------------------- sections
def sec_A():
    """eta=0 must reproduce build_wire_two_valley_iv exactly."""
    t0 = time.time()
    out = {}
    for tag, vo in [("clean", vo_clean()), ("step085pi", vo_step())]:
        Href = build_wire_two_valley_iv(N, DX, MU, B_T, DELTA, ALPHA_DEMO,
                                        M_SI, G_SI, vo_profile_ueV=vo)
        Hnew = H_mix(0.0, vo)
        dH = (Href - Hnew)
        max_dH = float(abs(dH).max()) if dH.nnz else 0.0
        Eref = lowest6_ueV(Href)
        Enew = lowest6_ueV(Hnew)
        max_dE = float(np.max(np.abs(Eref - Enew)))
        out[tag] = dict(
            max_matrix_diff_J=max_dH,
            lowest6_ref_ueV=[round(float(e), 6) for e in Eref],
            lowest6_mix_eta0_ueV=[round(float(e), 6) for e in Enew],
            max_abs_dE_ueV=max_dE,
            passes_tol_1em9_ueV=bool(max_dE < 1e-9),
        )
        print(f"A[{tag}]: max|dH|={max_dH:.3e} J, max|dE|={max_dE:.3e} ueV, "
              f"pass={max_dE < 1e-9}")
    out["runtime_s"] = round(time.time() - t0, 1)
    ckpt_save("A_validation", out)
    save_numbers("pairing_mix", dict(A_validation=out))


def sec_table(vo, tag):
    t0 = time.time()
    rows = {}
    for eta in ETAS:
        e0, e2, w = metrics(eta, vo)
        rows[f"{eta:.2f}"] = dict(E0_ueV=round(e0, 4), E2_ueV=round(e2, 3),
                                  end_weight=round(w, 3))
        print(f"{tag} eta={eta:.2f}: E0={e0:.4f} E2={e2:.3f} endw={w:.3f}",
              flush=True)
    out = dict(rows=rows, runtime_s=round(time.time() - t0, 1),
               params=dict(mu_ueV=MU, B_T=B_T, Delta_ueV=DELTA,
                           alpha_eVA=ALPHA_DEMO, lam_ueV=LAM, N=N, dx_m=DX))
    ckpt_save(tag, out)
    save_numbers("pairing_mix", {tag: out})


def sec_D(eta):
    """50-nm Poisson same-sign staircase, fig9 seed family, at given eta."""
    t0 = time.time()
    E2s, E0s = [], []
    for s in range(8):
        rng = np.random.default_rng([71, 1, 50, s])
        vo = _make_vo_profile(N, DX, 50e-9, "poisson", "fixed", rng)
        e0, e2, _ = metrics(eta, vo)
        E2s.append(round(e2, 3)); E0s.append(round(e0, 4))
        print(f"D seed {s}: E0={e0:.4f} E2={e2:.3f}", flush=True)
    out = dict(eta=eta, seeds=list(range(8)), E0_ueV=E0s, E2_ueV=E2s,
               median_E2_ueV=round(float(np.median(E2s)), 3),
               runtime_s=round(time.time() - t0, 1))
    key = f"D_ensemble_eta{eta:g}"
    ckpt_save(key, out)
    save_numbers("pairing_mix", {key: out})


def sec_E():
    """Threshold refinement + apples-to-apples eta=0 ensemble baseline.
    (i) finer eta grid for the single step around the 2x-deepening point;
    (ii) the SAME 8 seeds as section D at eta=0 (fig9 quotes 14 seeds)."""
    t0 = time.time()
    fine = {}
    for eta in [0.15, 0.22, 0.25, 0.28, 0.35, 0.40]:
        e0, e2, w = metrics(eta, vo_step())
        fine[f"{eta:.2f}"] = dict(E0_ueV=round(e0, 4), E2_ueV=round(e2, 3),
                                  end_weight=round(w, 3))
        print(f"E step eta={eta:.2f}: E0={e0:.4f} E2={e2:.3f}", flush=True)
    E2s = []
    for s in range(8):
        rng = np.random.default_rng([71, 1, 50, s])
        vo = _make_vo_profile(N, DX, 50e-9, "poisson", "fixed", rng)
        E2s.append(round(metrics(0.0, vo)[1], 3))
    # eta where the single-step E2 crosses half its eta=0 value
    # (4.034/2 = 2.017 ueV): linear interpolation between 0.22 and 0.25
    e_lo, e_hi = fine["0.22"]["E2_ueV"], fine["0.25"]["E2_ueV"]
    eta_star = 0.22 + 0.03 * (e_lo - 4.034 / 2) / (e_lo - e_hi)
    out = dict(step_fine_grid=fine,
               ensemble_eta0_same8seeds_E2_ueV=E2s,
               ensemble_eta0_median_E2_ueV=round(float(np.median(E2s)), 3),
               eta_star_2x_deepening=round(float(eta_star), 3),
               runtime_s=round(time.time() - t0, 1))
    ckpt_save("E_threshold_refine", out)
    save_numbers("pairing_mix", dict(E_threshold_refine=out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--eta", type=float, default=0.5,
                    help="intra-valley fraction for section D")
    args = ap.parse_args()
    if args.sec == "A":
        sec_A()
    elif args.sec == "B":
        sec_table(vo_clean(), "B_clean_vs_eta")
    elif args.sec == "C":
        sec_table(vo_step(), "C_step085pi_vs_eta")
    elif args.sec == "D":
        sec_D(args.eta)
    else:
        sec_E()
