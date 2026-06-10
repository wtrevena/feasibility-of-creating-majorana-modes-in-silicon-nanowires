"""
transport.py
============
Transport-gap calculation for the proximitized-nanowire (Lutchyn-Oreg) model
of majorana_sim.py.  It replaces the spectral proxy E2 (3rd-smallest |E| of
the CLOSED finite wire) used in fig5: under strong disorder E2 counts
localized bulk states that carry no current, so it stops measuring the
protection that matters for transport.

Geometry / method (no kwant available in this environment: pip wheel build
fails without root; everything below is a self-contained recursive-Green's-
function (RGF) implementation, validated against majorana_sim.build_wire).

  N(lead) -- S(disordered wire) -- N(lead), standard NS(N) setup:

  * Scattering region: EXACTLY the same 1D BdG chain as build_wire, but
    assembled site-major with 4 orbitals per cell (u_up, u_dn, v_up, v_dn):
        onsite   U_n   = [[(2t - mu + V_n) s0 + EZ sx ,   Delta (i sy)     ],
                          [        -Delta (i sy)      , -((2t-mu+V_n) s0 + EZ sx)]]
        hopping  V_hop = diag(-t s0 - i a_so sz ,  +t s0 - i a_so sz)   (n -> n+1)
    with t = hbar^2/(2 m* dx^2), a_so = alpha/(2 dx).  verify_cell_blocks()
    checks this block structure against build_wire ELEMENT BY ELEMENT (after
    the [u-block, v-block] -> site-major permutation) to machine precision,
    clean and disordered, plus an eigenvalue comparison.
  * Leads: semi-infinite normal metal, same kinetic terms (t, a_so, EZ),
    Delta = 0, no disorder, mu_lead = +2000 ueV, so 4 quasiparticle modes
    (2 spin-split electron + 2 hole) propagate at every |E| <= 60 ueV.
  * Lead surface Green's functions: exact wave-function matching (Ando):
    8x8 generalized Bloch eigenproblem  lambda*B*x = A*x,  modes split into
    incoming/outgoing by velocity (propagating) or |lambda| (evanescent).
    Cross-checked against Sancho-Rubio decimation.
  * Transmission: RGF forward sweep, Caroli formula
        T(E) = Tr[ Gamma_R  G_{N-1,0}  Gamma_L  G_{N-1,0}^dag ] ,
    the TOTAL quasiparticle transmission (electron + hole outgoing channels
    summed; Gammas are block-diagonal in e/h since the leads are normal).
  * Transport gap E_T: smallest E in [0, 60] ueV with T(E) > 0.01
    (1 ueV grid for the ensemble, 0.5 ueV for the clean wire; the crossing
    is refined by log-linear interpolation).  If T never exceeds 0.01 up to
    60 ueV the value is censored at 60 ueV and flagged in the JSON.
  * Topological invariant Q = sign det r at E = 0:  r is the full 4x4
    flux-normalized LEFT reflection block of the two-terminal S-matrix
    (both leads attached).  The second lead is essential: it broadens the
    far-end Majorana (width >> the residual splitting eps ~ 3e-4 ueV), so
    det r -> -+1 with the infinite-wire sign up to O(T(0)) corrections.
    With a hard-wall far end det r = +1 IDENTICALLY for any finite wire
    (at E = 0 exactly the split pair at +-eps is off resonance - the E->0
    limit of a closed finite wire is singular), verified numerically.
    To make det r a clean +-1 we rotate the incoming and
    outgoing propagating modes to a particle-hole-canonical (Majorana) basis:
    with P = tau_x K, M = X^{-1} (tau_x X*) is unitary symmetric; an
    Autonne-Takagi factor A (unitary, M = A A^T) is built by jointly
    diagonalizing the commuting real/imaginary parts of M with a real
    orthogonal Q (scipy.linalg.sqrtm is NOT used: M generically has the
    degenerate eigenvalue -1 sitting on its branch cut).  Then
    r_can = A_out^dag r A_in is REAL orthogonal and det r_can = +-1.
    The ABSOLUTE sign of det r_can still depends on the residual O(4) gauge
    of the Majorana mode bases (incoming and outgoing canonicalized
    independently), so the invariant is gauge-fixed by a KNOWN-TRIVIAL
    reference: the same lead, same lead-mode canonicalization, with the
    scattering region set to B = 0 (EZ = 0 < Delta, trivial).  Then
    Q = sign(det r_wire * det r_reference) = -1 (topological) / +1 (trivial).

Conventions
-----------
All transport matrices are in ueV.  Cell basis (u_up, u_dn, v_up, v_dn);
V_hop is the block H_{n,n+1}.  Retarded GFs via outgoing-wave lead boundary
conditions plus +i*eta, eta = 1e-8 ueV (transmission), 1e-12 ueV (invariant).
Disorder identical to run_analysis.py fig5 case 1 ("engineered Si"):
V_n ~ uniform(-W, W), rng = np.random.default_rng([91, 1, iw, r]) with iw the
index of W in fig5's Ws list [0,20,50,100,200,400,800,1600] and r the seed.

Running `python3 transport.py` regenerates
    output/fig12_transport_gap.png
    output/data/transport_numbers.json
(creates nothing else, modifies no existing file).
"""

import json
import os
import time

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from majorana_sim import HBAR, ME, QE, UEV, EZ_J, s0, sx, sy, sz, \
    build_wire, bulk_gap_ueV

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
DATA = os.path.join(OUT, "data")

# ---- fig5 case-1 parameters ("engineered Si"), mirrored from run_analysis.py
M_SI, G_SI = 0.19, 2.0
DELTA = 50.0          # ueV
ALPHA = 0.05          # eV*Angstrom
DX = 2.5e-9           # m
L = 2e-6              # m
B_FIELD = 1.5         # T
MU = 0.0              # ueV
MU_LEAD = 2000.0      # ueV  (normal leads, several propagating modes)
ETA_T = 1e-8          # ueV, transmission sweeps
ETA_Q = 1e-12         # ueV, E=0 invariant sweep
T_THRESHOLD = 0.01
E_MAX = 60.0          # ueV
W_LIST = [100.0, 200.0, 400.0, 800.0]
IW_OF_W = {100.0: 3, 200.0: 4, 400.0: 5, 800.0: 6}   # index in fig5 Ws list
N_SEEDS = 10

I4 = np.eye(4, dtype=complex)
ISY = 1j * sy                                  # [[0,1],[-1,0]]
TAUZ4 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
TAUX4 = np.kron(np.array([[0, 1], [1, 0]], dtype=complex), np.eye(2))


# --------------------------------------------------------------- cell blocks
def cell_blocks(dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g):
    """(U, V_hop) 4x4 blocks in ueV for one BdG cell (u_up,u_dn,v_up,v_dn)."""
    t = HBAR**2 / (2 * m_rel * ME * dx**2) / UEV
    aso = (alpha_eVA * 1e-10 * QE) / (2 * dx) / UEV
    EZ = EZ_J(g, B) / UEV
    h_on = (2 * t - mu_ueV) * s0 + EZ * sx          # real 2x2
    U = np.zeros((4, 4), dtype=complex)
    U[:2, :2] = h_on
    U[2:, 2:] = -h_on.conj()
    U[:2, 2:] = Delta_ueV * ISY
    U[2:, :2] = (Delta_ueV * ISY).conj().T          # = -Delta*ISY for real Delta
    V = np.zeros((4, 4), dtype=complex)
    V[:2, :2] = -t * s0 - 1j * aso * sz
    V[2:, 2:] = (t * s0 - 1j * aso * sz)            # = -(-t s0 - i aso sz)^*
    return U, V


def verify_cell_blocks(N=40, W=300.0):
    """Element-by-element check of the cell-based chain vs build_wire (after
    permuting build_wire's [u-block, v-block] ordering to site-major cells),
    clean and disordered, plus an eigenvalue comparison.  Returns dict."""
    U, V = cell_blocks(DX, MU, B_FIELD, DELTA, ALPHA, M_SI, G_SI)
    perm = np.empty(4 * N, dtype=int)
    for n in range(N):
        perm[4 * n + 0] = 2 * n
        perm[4 * n + 1] = 2 * n + 1
        perm[4 * n + 2] = 2 * N + 2 * n
        perm[4 * n + 3] = 2 * N + 2 * n + 1
    out = {}
    for tag, Wdis in (("clean", 0.0), ("disordered", W)):
        if Wdis:
            rng = np.random.default_rng([91, 1, 3, 0])
            Vd = rng.uniform(-Wdis, Wdis, N)
            rng2 = np.random.default_rng([91, 1, 3, 0])   # same stream
            Hw = build_wire(N, DX, MU, B_FIELD, DELTA, ALPHA, M_SI, G_SI,
                            disorder_ueV=Wdis, rng=rng2).toarray() / UEV
        else:
            Vd = np.zeros(N)
            Hw = build_wire(N, DX, MU, B_FIELD, DELTA, ALPHA, M_SI,
                            G_SI).toarray() / UEV
        Hc = np.zeros((4 * N, 4 * N), dtype=complex)
        for n in range(N):
            Hc[4*n:4*n+4, 4*n:4*n+4] = U + Vd[n] * TAUZ4
            if n < N - 1:
                Hc[4*n:4*n+4, 4*n+4:4*n+8] = V
                Hc[4*n+4:4*n+8, 4*n:4*n+4] = V.conj().T
        Hwp = Hw[np.ix_(perm, perm)]
        out[f"max_abs_H_diff_{tag}_ueV"] = float(np.abs(Hc - Hwp).max())
        ev_c = np.linalg.eigvalsh(Hc)
        ev_w = np.linalg.eigvalsh(Hw)
        out[f"max_eigval_diff_{tag}_ueV"] = float(np.abs(ev_c - ev_w).max())
    return out


# ----------------------------------------------------------------- lead modes
def lead_data(E_ueV, H00, Vh, prop_tol=1e-7):
    """Exact lead boundary data at real energy E (ueV) by wave matching.

    Solves the Bloch problem (E - H00 - Vh*lam - Vh^dag/lam) phi = 0 as an
    8x8 generalized eigenproblem.  Returns retarded surface GFs / self-
    energies for a LEFT lead (cells n<=-1) and a RIGHT lead (cells n>=N),
    plus the incoming/outgoing propagating modes of the left lead (for the
    reflection matrix).  Velocities are in arbitrary common units (only
    signs and ratios are ever used)."""
    n = 4
    A = np.zeros((2 * n, 2 * n), dtype=complex)
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Vh.conj().T
    A[n:, n:] = E_ueV * np.eye(n) - H00
    Bm = np.eye(2 * n, dtype=complex)
    Bm[n:, n:] = Vh
    lam, X = sla.eig(A, b=Bm)
    if not np.all(np.isfinite(lam)):
        raise RuntimeError("lead eigenproblem produced non-finite lambda")
    phi = X[:n, :]
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    absl = np.abs(lam)
    prop = np.abs(absl - 1.0) < prop_tol
    # group velocity ~ -2 Im(lam phi^dag Vh phi)  (positive = right-moving)
    v = np.array([-2.0 * np.imag(lam[i] * np.vdot(phi[:, i], Vh @ phi[:, i]))
                  for i in range(2 * n)])
    out_L = (prop & (v < 0)) | (~prop & (absl > 1.0))   # left lead: outgoing
    out_R = (prop & (v > 0)) | (~prop & (absl < 1.0))   # right lead: outgoing
    in_L = prop & (v > 0)                                # left lead: incoming
    if out_L.sum() != n or out_R.sum() != n:
        raise RuntimeError(f"mode count failure at E={E_ueV}: "
                           f"{out_L.sum()} / {out_R.sum()}")
    PhiL, lamL = phi[:, out_L], lam[out_L]
    PhiR, lamR = phi[:, out_R], lam[out_R]
    LamL = PhiL @ np.diag(1.0 / lamL) @ np.linalg.inv(PhiL)   # psi_-2 = LamL psi_-1
    LamR = PhiR @ np.diag(lamR) @ np.linalg.inv(PhiR)         # psi_N+1 = LamR psi_N
    EI = E_ueV * np.eye(n)
    gL = np.linalg.inv(EI - H00 - Vh.conj().T @ LamL)
    gR = np.linalg.inv(EI - H00 - Vh @ LamR)
    SigL = Vh.conj().T @ gL @ Vh        # self-energy on cell 0
    SigR = Vh @ gR @ Vh.conj().T        # self-energy on cell N-1
    return dict(gL=gL, gR=gR, SigL=SigL, SigR=SigR, LamL=LamL,
                Phi_in=phi[:, in_L], lam_in=lam[in_L], v_in=v[in_L],
                Phi_outL=PhiL, lam_outL=lamL, v_outL=v[out_L],
                prop_outL=prop[out_L])


def sancho_rubio(E_ueV, H00, H01, eta=1e-4, maxiter=200, tol=1e-14):
    """Decimation surface GF (cross-check only). H01 = hopping from the
    surface cell toward the lead interior."""
    Ec = (E_ueV + 1j * eta) * np.eye(H00.shape[0])
    eps_s = H00.copy(); eps = H00.copy()
    a = H01.copy(); b = H01.conj().T.copy()
    for _ in range(maxiter):
        g = np.linalg.inv(Ec - eps)
        agb = a @ g @ b
        eps_s = eps_s + agb
        eps = eps + agb + b @ g @ a
        a = a @ g @ a
        b = b @ g @ b
        if np.abs(a).max() + np.abs(b).max() < tol:
            break
    return np.linalg.inv(Ec - eps_s)


def lead_sigmas(E_grid, H00_lead, Vh):
    """Stacked (nE,4,4) retarded lead self-energies and broadenings."""
    nE = len(E_grid)
    SigL = np.empty((nE, 4, 4), complex)
    SigR = np.empty((nE, 4, 4), complex)
    for i, E in enumerate(E_grid):
        ld = lead_data(E, H00_lead, Vh)
        SigL[i] = ld["SigL"]
        SigR[i] = ld["SigR"]
    GamL = 1j * (SigL - SigL.conj().transpose(0, 2, 1))
    GamR = 1j * (SigR - SigR.conj().transpose(0, 2, 1))
    return SigL, SigR, GamL, GamR


# ------------------------------------------------------------ RGF transmission
def total_transmission(E_grid, Vdis_ueV, U, Vh, SigL, SigR, GamL, GamR,
                       eta=ETA_T):
    """Total quasiparticle transmission T(E) through the NSN device,
    batched over the energy grid.  Caroli: T = Tr[GamR G_{N-1,0} GamL G^dag].
    Vdis_ueV: onsite disorder per site (enters as +V tau_z, like build_wire)."""
    N = len(Vdis_ueV)
    A = (np.asarray(E_grid) + 1j * eta)[:, None, None] * I4
    Vd = Vh.conj().T
    g = np.linalg.inv(A - (U + Vdis_ueV[0] * TAUZ4) - SigL)
    F = g.copy()
    for nn in range(1, N):
        Un = U + Vdis_ueV[nn] * TAUZ4
        M = A - Un - Vd @ g @ Vh
        if nn == N - 1:
            M = M - SigR
        g = np.linalg.inv(M)
        F = g @ (Vd @ F)
    Fd = F.conj().transpose(0, 2, 1)
    T = np.einsum("eab,ebc,ecd,eda->e", GamR, F, GamL, Fd).real
    return T


def transport_gap(E_grid, T, thr=T_THRESHOLD):
    """Smallest E with T > thr (log-linear interpolated). Returns
    (E_T_ueV, censored)."""
    idx = np.where(T > thr)[0]
    if len(idx) == 0:
        return float(E_grid[-1]), True
    i = int(idx[0])
    if i == 0:
        return float(E_grid[0]), False
    T0, T1 = T[i - 1], T[i]
    if T0 <= 0:
        return float(E_grid[i]), False
    f = (np.log(thr) - np.log(T0)) / (np.log(T1) - np.log(T0))
    return float(E_grid[i - 1] + f * (E_grid[i] - E_grid[i - 1])), False


# --------------------------------------------------- E=0 reflection invariant
def _takagi_unitary(M, tries=8):
    """Autonne-Takagi factor of a UNITARY SYMMETRIC M: unitary A with
    M = A A^T.  Since M M* = 1, Re M and Im M are commuting real symmetric
    matrices; a real orthogonal Q diagonalizes both (joint diagonalization,
    certified below), M = Q diag(e^{i th}) Q^T, A = Q diag(e^{i th/2}) Q^T.
    (scipy.linalg.sqrtm fails here: M generically has a degenerate
    eigenvalue -1 on the principal branch cut.)"""
    Mr, Mi = M.real, M.imag
    rng = np.random.default_rng(7)
    for k in range(tries):
        c = 0.61803398875 if k == 0 else rng.standard_normal()
        _, Q = np.linalg.eigh(Mr + c * Mi)
        Dr = Q.T @ Mr @ Q
        Di = Q.T @ Mi @ Q
        offd = max(np.abs(Dr - np.diag(np.diag(Dr))).max(),
                   np.abs(Di - np.diag(np.diag(Di))).max())
        if offd < 1e-9:
            th = np.angle(np.diag(Dr) + 1j * np.diag(Di))
            return (Q * np.exp(0.5j * th)[None, :]) @ Q.T
    raise RuntimeError("Takagi joint diagonalization failed")


def _ph_canonical(Phi, vabs):
    """Takagi rotation A (unitary) such that the flux-normalized modes
    X' = X A satisfy P X' = X' with P = tau_x K.  Returns (A, checks)."""
    X = Phi / np.sqrt(vabs)[None, :]
    M = np.linalg.solve(X, TAUX4 @ X.conj())
    sym_err = float(np.abs(M - M.T).max())
    uni_err = float(np.abs(M @ M.conj().T - np.eye(4)).max())
    A = _takagi_unitary(M)
    fac_err = float(np.abs(A @ A.T - M).max())
    return A, max(sym_err, uni_err, fac_err)


def reflection_invariant(Vdis_ueV, U, Vh, ld, eta=ETA_Q, ref_sign=1.0,
                         attach_right=True):
    """det r at E=0: left reflection block of the two-terminal S-matrix
    (attach_right=True; the right lead broadens the far-end Majorana, see
    module docstring -- with attach_right=False, hard wall, any FINITE wire
    gives det r = +1 identically).  ld = lead_data(0, ...).  Returns dict
    with det_r (real, +-1 up to O(T(0)), in the fixed Majorana gauge of THIS
    lead) and Q = sign(det_r * ref_sign), where ref_sign is the det_r of a
    known-trivial reference region with the same lead (gauge fixing; see
    module docstring).  Q = 0 flags failed numerical checks (e.g. wire
    effectively gapless at E=0, |det r| not close to 1)."""
    N = len(Vdis_ueV)
    A0 = (0.0 + 1j * eta) * I4
    Vd = Vh.conj().T
    # backward sweep: surface GF of cells [n..N-1], right end open or walled
    g = np.linalg.inv(A0 - (U + Vdis_ueV[N - 1] * TAUZ4)
                      - (ld["SigR"] if attach_right else 0.0))
    for nn in range(N - 2, 0, -1):
        g = np.linalg.inv(A0 - (U + Vdis_ueV[nn] * TAUZ4) - Vh @ g @ Vd)
    G00 = np.linalg.inv(A0 - (U + Vdis_ueV[0] * TAUZ4) - ld["SigL"]
                        - Vh @ g @ Vd)
    Phi_in, lam_in, v_in = ld["Phi_in"], ld["lam_in"], ld["v_in"]
    Phi_out, v_out, prop_out = ld["Phi_outL"], ld["v_outL"], ld["prop_outL"]
    if Phi_in.shape[1] != 4 or not prop_out.all():
        raise RuntimeError("expected 4 propagating modes per direction at E=0")
    # source: q = V^dag gL V^dag (lam^-1 - LamL) phi  for each incoming mode
    K = Vd @ (Phi_in @ np.diag(1.0 / lam_in) - ld["LamL"] @ Phi_in)
    Psi0 = G00 @ (Vd @ ld["gL"] @ K)
    Psim1 = ld["gL"] @ (K + Vh @ Psi0)
    C = np.linalg.solve(Phi_out, Psim1 - Phi_in)
    r = np.diag(np.sqrt(np.abs(v_out))) @ C @ np.diag(1.0 / np.sqrt(v_in))
    uni_err = float(np.abs(r.conj().T @ r - np.eye(4)).max())
    Ain, chk_in = _ph_canonical(Phi_in, np.abs(v_in))
    Aout, chk_out = _ph_canonical(Phi_out, np.abs(v_out))
    r_can = Aout.conj().T @ r @ Ain
    real_err = float(np.abs(r_can.imag).max())
    detr = complex(np.linalg.det(r_can))
    # with both leads r is sub-unitary by the (tiny, e^{-L/xi}) transmission;
    # |det r| << 1 means the wire is effectively gapless at E=0: Q undefined
    ok = real_err < 1e-3 and max(chk_in, chk_out) < 1e-7 \
        and abs(detr.real) > 0.5 and uni_err < 0.5
    Q = int(np.sign(detr.real * ref_sign)) if ok else 0
    return dict(Q=Q, det_r=float(detr.real), unitarity_err=uni_err,
                reality_err=real_err, basis_err=max(chk_in, chk_out))


# ----------------------------------------------------------------- self-tests
def self_tests(U_lead, Vh):
    """Lead self-energy and full-machinery sanity checks. Returns dict."""
    res = {}
    # (1) wave-matching vs Sancho-Rubio decimation at a propagating energy
    E = 30.0
    ld = lead_data(E, U_lead, Vh)
    gL_sr = sancho_rubio(E, U_lead, Vh.conj().T, eta=1e-4)
    gR_sr = sancho_rubio(E, U_lead, Vh, eta=1e-4)
    res["gL_modes_vs_sancho"] = float(np.abs(ld["gL"] - gL_sr).max()
                                      / np.abs(ld["gL"]).max())
    res["gR_modes_vs_sancho"] = float(np.abs(ld["gR"] - gR_sr).max()
                                      / np.abs(ld["gR"]).max())
    # (2) ideal normal region (same as lead) must give T = 4 exactly
    Eg = np.array([10.0, 30.0, 55.0])
    SigL, SigR, GamL, GamR = lead_sigmas(Eg, U_lead, Vh)
    T = total_transmission(Eg, np.zeros(50), U_lead, Vh,
                           SigL, SigR, GamL, GamR)
    res["ballistic_T_minus_4"] = float(np.abs(T - 4.0).max())
    return res


# ----------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    N = int(round(L / DX))                                     # 800 cells
    U_wire, Vh = cell_blocks(DX, MU, B_FIELD, DELTA, ALPHA, M_SI, G_SI)
    U_lead, Vh_lead = cell_blocks(DX, MU_LEAD, B_FIELD, 0.0, ALPHA, M_SI, G_SI)
    assert np.abs(Vh - Vh_lead).max() == 0.0    # same kinetic hopping

    print("verifying cell blocks against build_wire ...", flush=True)
    verify = verify_cell_blocks()
    print("  ", verify, flush=True)

    print("lead / machinery self-tests ...", flush=True)
    tests = self_tests(U_lead, Vh)
    print("  ", tests, flush=True)

    # ---------------------------------------------------- clean wire (task 4a)
    Eg_clean = np.arange(0.0, E_MAX + 1e-9, 0.5)
    SigLc, SigRc, GamLc, GamRc = lead_sigmas(Eg_clean, U_lead, Vh)
    T_clean = total_transmission(Eg_clean, np.zeros(N), U_wire, Vh,
                                 SigLc, SigRc, GamLc, GamRc)
    ET_clean, cens_clean = transport_gap(Eg_clean, T_clean)
    bulk = bulk_gap_ueV(MU, B_FIELD, DELTA, ALPHA, M_SI, G_SI)
    print(f"clean transport gap = {ET_clean:.2f} ueV "
          f"(bulk gap {bulk:.2f} ueV)", flush=True)

    # E=0 invariant. Gauge-fix det r with a known-trivial reference region
    # (wire at B=0: EZ=0 < Delta) attached to the SAME lead, then
    # Q = sign(det r * det r_ref): -1 topological / +1 trivial.
    ld0 = lead_data(0.0, U_lead, Vh)
    U_ref, _ = cell_blocks(DX, MU, 0.0, DELTA, ALPHA, M_SI, G_SI)
    inv_ref = reflection_invariant(np.zeros(N), U_ref, Vh, ld0)
    sgn_ref = float(np.sign(inv_ref["det_r"]))
    inv_topo = reflection_invariant(np.zeros(N), U_wire, Vh, ld0,
                                    ref_sign=sgn_ref)
    # cross-validation at B=0.3 T (trivial: EZ=17.4 < Delta=50), own lead+ref
    U_triv, _ = cell_blocks(DX, MU, 0.3, DELTA, ALPHA, M_SI, G_SI)
    U_lead_triv, _ = cell_blocks(DX, MU_LEAD, 0.3, 0.0, ALPHA, M_SI, G_SI)
    ld0_triv = lead_data(0.0, U_lead_triv, Vh)
    inv_ref_triv = reflection_invariant(np.zeros(N), U_ref, Vh, ld0_triv)
    inv_triv = reflection_invariant(
        np.zeros(N), U_triv, Vh, ld0_triv,
        ref_sign=float(np.sign(inv_ref_triv["det_r"])))
    print(f"invariant: Q = {inv_topo['Q']:+d} (B=1.5 T, expect -1), "
          f"Q = {inv_triv['Q']:+d} (B=0.3 T, expect +1)", flush=True)
    # B-sweep of the REGION at fixed (B=1.5 T) lead: Q must flip at the
    # known clean phase boundary B* = Delta/(g muB / 2) = 0.864 T
    inv_Bsweep = {}
    for Bx in (0.3, 0.6, 0.8, 0.95, 1.2, 1.5):
        Ux, _ = cell_blocks(DX, MU, Bx, DELTA, ALPHA, M_SI, G_SI)
        qx = reflection_invariant(np.zeros(N), Ux, Vh, ld0, ref_sign=sgn_ref)
        inv_Bsweep[f"B_{Bx}"] = qx["Q"]
    print("invariant region-B sweep (lead at 1.5 T, B*=0.864 T):",
          inv_Bsweep, flush=True)

    # ------------------------------------------- disordered ensemble (task 4b)
    Eg = np.arange(0.0, E_MAX + 1e-9, 1.0)
    SigL, SigR, GamL, GamR = lead_sigmas(Eg, U_lead, Vh)
    ensemble = {}
    example_curves = {}
    for W in W_LIST:
        iw = IW_OF_W[W]
        ETs, cens, Qs, detrs = [], [], [], []
        for r in range(N_SEEDS):
            rng = np.random.default_rng([91, 1, iw, r])
            Vdis = rng.uniform(-W, W, N)
            T = total_transmission(Eg, Vdis, U_wire, Vh,
                                   SigL, SigR, GamL, GamR)
            ET, c = transport_gap(Eg, T)
            ETs.append(ET); cens.append(c)
            qi = reflection_invariant(Vdis, U_wire, Vh, ld0,
                                      ref_sign=sgn_ref)
            Qs.append(qi["Q"]); detrs.append(qi["det_r"])
            if r == 0 and W in (200.0, 800.0):
                example_curves[W] = T.copy()
        ensemble[W] = dict(E_T_seeds_ueV=[round(x, 3) for x in ETs],
                           n_censored=int(sum(cens)),
                           median_ueV=float(np.median(ETs)),
                           q1_ueV=float(np.percentile(ETs, 25)),
                           q3_ueV=float(np.percentile(ETs, 75)),
                           Q_seeds=Qs,
                           frac_topological=float(np.mean(
                               [q == -1 for q in Qs])),
                           det_r_seeds=[round(d, 4) for d in detrs])
        print(f"W={W:5.0f}: median E_T = {ensemble[W]['median_ueV']:6.2f} ueV"
              f"  (censored {ensemble[W]['n_censored']}/10,"
              f"  frac Q=-1: {ensemble[W]['frac_topological']:.1f})",
              flush=True)

    # ------------------------------------------- spectral proxy for comparison
    spectral = dict(note="E2 = 3rd-smallest |E| of the closed wire, "
                         "fig5 case1 of run_analysis.py")
    spec_med16, spec_med10 = {}, {}
    try:
        z = np.load(os.path.join(DATA, "fig5_scan_v2.npz"), allow_pickle=True)
        E2c1 = z["E2"][1]
        spectral["clean_E2_ueV"] = round(float(E2c1[0, 0]), 2)
        for W in W_LIST:
            iw = IW_OF_W[W]
            spec_med16[W] = float(np.median(E2c1[iw, :]))
            spec_med10[W] = float(np.median(E2c1[iw, :N_SEEDS]))
        spectral["median16_ueV"] = {f"{W:.0f}": round(v, 2)
                                    for W, v in spec_med16.items()}
        spectral["median_same10seeds_ueV"] = {f"{W:.0f}": round(v, 2)
                                              for W, v in spec_med10.items()}
    except Exception as exc:                                 # pragma: no cover
        spectral["error"] = f"could not load fig5_scan_v2.npz: {exc}"

    # ------------------------------------------------------- W_half estimates
    def first_crossing_declining(xs, ys, level):
        for i in range(len(ys)):
            if ys[i] <= level:
                if i == 0:
                    return float(xs[0])
                y0, y1 = ys[i - 1], ys[i]
                if y1 == y0:
                    return float(xs[i])
                return float(xs[i - 1] + (level - y0) / (y1 - y0)
                             * (xs[i] - xs[i - 1]))
        return None

    Ws_all = [0.0] + W_LIST
    med_T = [ET_clean] + [ensemble[W]["median_ueV"] for W in W_LIST]
    Wh_T = first_crossing_declining(Ws_all, med_T, ET_clean / 2)
    Wh_T_out = (round(Wh_T, 1) if Wh_T is not None
                else f">{W_LIST[-1]:.0f} (not reached)")
    if spec_med16:
        med_S = [spectral["clean_E2_ueV"]] + [spec_med16[W] for W in W_LIST]
        Wh_S = first_crossing_declining(Ws_all, med_S,
                                        spectral["clean_E2_ueV"] / 2)
    else:
        Wh_S = None

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ax[0].semilogy(Eg_clean, np.clip(T_clean, 1e-16, None), "-", color="C0",
                   label="clean")
    for W, c in ((200.0, "C1"), (800.0, "C3")):
        ax[0].semilogy(Eg, np.clip(example_curves[W], 1e-16, None), "-",
                       color=c, label=f"W = {W:.0f} µeV (seed 0)")
    ax[0].axhline(T_THRESHOLD, color="gray", ls=":", lw=1,
                  label=f"threshold T = {T_THRESHOLD}")
    ax[0].axvline(bulk, color="k", ls="--", lw=1,
                  label=f"clean bulk gap {bulk:.1f} µeV")
    ax[0].set_xlabel("E (µeV)")
    ax[0].set_ylabel("total transmission T(E)")
    ax[0].set_ylim(1e-14, 8)
    ax[0].set_title("(a) NSN transmission vs energy")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3, which="both")

    Wp = np.clip(Ws_all, 50, None)        # W=0 plotted at 50 (log axis)
    q1 = [ET_clean] + [ensemble[W]["q1_ueV"] for W in W_LIST]
    q3 = [ET_clean] + [ensemble[W]["q3_ueV"] for W in W_LIST]
    ax[1].semilogx(Wp, med_T, "o-", color="C0",
                   label=f"transport gap E$_T$ (T>{T_THRESHOLD}), median of "
                         f"{N_SEEDS}")
    ax[1].fill_between(Wp, q1, q3, color="C0", alpha=0.2, label="IQR")
    if spec_med16:
        ax[1].semilogx(Wp, med_S, "s--", color="C2",
                       label="spectral proxy E$_2$ (fig5, median of 16)")
    ax[1].axhline(ET_clean / 2, color="C0", ls=":", lw=1)
    ax[1].axhline(bulk, color="k", ls="--", lw=0.8, alpha=0.5)
    txt = f"W$_{{1/2}}$(transport) = {Wh_T_out}"
    if Wh_S is not None:
        txt += f"\nW$_{{1/2}}$(spectral)  = {Wh_S:.0f} µeV"
    txt += "\n(censored at 60 µeV where T<0.01 everywhere)"
    ax[1].text(0.03, 0.05, txt, transform=ax[1].transAxes, fontsize=8,
               bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    ax[1].set_xlabel("onsite disorder amplitude W (µeV)  [W=0 plotted at 50]")
    ax[1].set_ylabel("gap (µeV)")
    ax[1].set_title("(b) transport gap vs spectral proxy")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Fig 12 — transport gap of the engineered-Si wire "
                 f"(α={ALPHA} eV·Å, Δ={DELTA:.0f} µeV, B={B_FIELD} T, µ=0, "
                 f"L={L*1e6:.0f} µm, dx={DX*1e9:.1f} nm; normal leads "
                 f"µ_lead={MU_LEAD:.0f} µeV)", y=1.0)
    fig.savefig(os.path.join(OUT, "fig12_transport_gap.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------- JSON
    numbers = {
        "method": {
            "kwant_available": False,
            "approach": "recursive Green's function (Caroli) + Ando wave "
                        "matching for lead self-energies; see module "
                        "docstring",
            "geometry": "normal lead | BdG wire | normal lead (NSN); "
                        "invariant from single-lead NS reflection at E=0",
            "mu_lead_ueV": MU_LEAD, "eta_ueV": ETA_T,
            "threshold_T": T_THRESHOLD,
            "E_grid_ueV": f"0..{E_MAX:.0f} step 1.0 (ensemble) / 0.5 (clean)",
            "seeds": "np.random.default_rng([91, 1, iw, r]), identical "
                     "disorder realizations to run_analysis fig5 case1",
        },
        "verification": {**verify, **tests},
        "clean": {
            "transport_gap_ueV": round(ET_clean, 2),
            "bulk_gap_ueV": round(bulk, 2),
            "spectral_E2_ueV": spectral.get("clean_E2_ueV"),
            "T_at_E0": float(T_clean[0]),
            "invariant_topological_B1p5T": inv_topo,
            "invariant_trivial_B0p3T": inv_triv,
            "invariant_reference": {
                "region": "B=0 (EZ=0 < Delta: trivial), same lead",
                **inv_ref},
            "invariant_region_B_sweep_Q": inv_Bsweep,
        },
        "disorder": {f"W_{W:.0f}": ensemble[W] for W in W_LIST},
        "spectral_proxy": spectral,
        "comparison": {
            "W_half_transport_ueV": Wh_T_out,
            "W_half_spectral_fig5_ueV": (round(Wh_S, 1) if Wh_S is not None
                                         else None),
            "W_half_spectral_key_numbers_ueV": 410.6,
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    # human-readable verdict
    medians_T = {f"{W:.0f}": round(ensemble[W]["median_ueV"], 1)
                 for W in W_LIST}
    numbers["comparison"]["median_transport_gap_ueV"] = medians_T
    numbers["comparison"]["median_spectral_E2_ueV"] = spectral.get(
        "median16_ueV")
    with open(os.path.join(DATA, "transport_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    print(json.dumps(numbers["comparison"], indent=2))
    print(f"done in {numbers['runtime_s']} s")
    return numbers


if __name__ == "__main__":
    main()
# end of transport.py
