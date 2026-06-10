"""
transport_valley.py — RGF transport gap for the TWO-VALLEY wire (pre-submission
item 2): extends transport.py's method to 8-orbital cells with per-site onsite
blocks (valley-orbit profiles), answering whether the step-disorder damage seen
in the spectral proxy (figs 6/9) is a real transport-gap collapse and how it
scales with L.

Cell basis per site (site-major): [u(valley x spin) (4), v(valley x spin) (4)].
Scattering region: rashba-mode build_wire_two_valley_iv physics, validated
against the closed-chain spectrum to machine precision. Leads: normal
(Delta = 0, no valley-orbit), mu_lead = +2000 ueV, same t/a_so/EZ -> 8
propagating modes for |E| <= 60 ueV. Method identical to transport.py
(Ando wave matching + Caroli RGF), dimension-generic.

Run: python3 transport_valley.py   (writes valley section into
output/data/transport_numbers.json; ~1 min)
"""
import json
import time

import numpy as np
import scipy.linalg as sla

from majorana_sim import (HBAR, ME, QE, UEV, EZ_J, build_wire_two_valley_iv,
                          solve_lowest)

S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1., -1.]).astype(complex)
V0 = np.eye(2, dtype=complex)
VX = np.array([[0, 1], [1, 0]], dtype=complex)
VY = np.array([[0, -1j], [1j, 0]])


def valley_cells(N, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
                 vo_profile_ueV=None, mu_lead_ueV=None):
    """(onsite_list[N] (8x8), hop (8x8)) in ueV units, site-major."""
    m = m_rel * ME
    t = HBAR**2 / (2 * m * dx**2) / UEV
    aso = (alpha_eVA * 1e-10 * QE) / (2 * dx) / UEV
    EZ = EZ_J(g, B) / UEV
    mu = mu_lead_ueV if mu_lead_ueV is not None else mu_ueV
    D = Delta_ueV
    lam = (np.zeros(N, complex) if vo_profile_ueV is None
           else np.asarray(vo_profile_ueV, complex))
    hopA = np.kron(V0, -t * S0 - 1j * aso * SZ)
    hop = np.zeros((8, 8), complex)
    hop[:4, :4] = hopA
    hop[4:, 4:] = -hopA.conj()
    Dm = D * np.kron(VX, 1j * SY)
    cells = []
    for n in range(N):
        A = (np.kron(V0, (2 * t - mu) * S0 + EZ * SX)
             + np.kron(lam[n].real * VX + lam[n].imag * VY, S0))
        U = np.zeros((8, 8), complex)
        U[:4, :4] = A
        U[4:, 4:] = -A.conj()
        U[:4, 4:] = Dm
        U[4:, :4] = Dm.conj().T
        cells.append(U)
    return cells, hop


def verify_against_msim(N=60, seed=3):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, N)
    vo = 75 * np.exp(1j * phi)
    H = build_wire_two_valley_iv(N, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                                 vo_profile_ueV=vo).toarray() / UEV
    cells, hop = valley_cells(N, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0, vo)
    Hc = np.zeros((8 * N, 8 * N), complex)
    for n in range(N):
        Hc[8*n:8*n+8, 8*n:8*n+8] = cells[n]
        if n < N - 1:
            Hc[8*n:8*n+8, 8*(n+1):8*(n+1)+8] = hop
            Hc[8*(n+1):8*(n+1)+8, 8*n:8*n+8] = hop.conj().T
    # permutation: msim [u(site x 4), v(site x 4)] -> site-major [u4, v4]
    perm = np.empty(8 * N, int)
    for n in range(N):
        perm[8*n:8*n+4] = 4*n + np.arange(4)
        perm[8*n+4:8*n+8] = 4*N + 4*n + np.arange(4)
    Hp = H[np.ix_(perm, perm)]
    err = np.abs(Hp - Hc).max()
    assert err < 1e-9, err
    return err


def lead_data_n(E, H00, Vh, prop_tol=1e-7):
    n = H00.shape[0]
    A = np.zeros((2 * n, 2 * n), complex)
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Vh.conj().T
    A[n:, n:] = E * np.eye(n) - H00
    Bm = np.eye(2 * n, dtype=complex)
    Bm[n:, n:] = Vh
    lam, X = sla.eig(A, b=Bm)
    phi = X[:n, :]
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    absl = np.abs(lam)
    prop = np.abs(absl - 1.0) < prop_tol
    v = np.array([-2.0 * np.imag(lam[i] * np.vdot(phi[:, i], Vh @ phi[:, i]))
                  for i in range(2 * n)])
    out_L = (prop & (v < 0)) | (~prop & (absl > 1.0))
    out_R = (prop & (v > 0)) | (~prop & (absl < 1.0))
    if out_L.sum() != n or out_R.sum() != n:
        raise RuntimeError(f"mode count {out_L.sum()}/{out_R.sum()} at E={E}")
    PhiL, lamL = phi[:, out_L], lam[out_L]
    PhiR, lamR = phi[:, out_R], lam[out_R]
    LamL = PhiL @ np.diag(1.0 / lamL) @ np.linalg.inv(PhiL)
    LamR = PhiR @ np.diag(lamR) @ np.linalg.inv(PhiR)
    EI = E * np.eye(n)
    gL = np.linalg.inv(EI - H00 - Vh.conj().T @ LamL)
    gR = np.linalg.inv(EI - H00 - Vh @ LamR)
    return (Vh.conj().T @ gL @ Vh, Vh @ gR @ Vh.conj().T)


def transmission(E_grid, cells, hop, H00_lead, eta=1e-6):
    N = len(cells)
    n = 8
    T = np.zeros(len(E_grid))
    Vd = hop.conj().T
    for i, E in enumerate(E_grid):
        SigL, SigR = lead_data_n(E, H00_lead, hop)
        GamL = 1j * (SigL - SigL.conj().T)
        GamR = 1j * (SigR - SigR.conj().T)
        A = (E + 1j * eta) * np.eye(n)
        g = np.linalg.inv(A - cells[0] - SigL)
        F = g.copy()
        for nn in range(1, N):
            M = A - cells[nn] - Vd @ g @ hop
            if nn == N - 1:
                M = M - SigR
            g = np.linalg.inv(M)
            F = g @ (Vd @ F)
        T[i] = np.real(np.trace(GamR @ F @ GamL @ F.conj().T))
    return T


def transport_gap(E_grid, T, thr=0.01):
    idx = np.where(T > thr)[0]
    if len(idx) == 0:
        return float(E_grid[-1]), True
    i = int(idx[0])
    if i == 0:
        return float(E_grid[0]), False
    T0, T1 = max(T[i - 1], 1e-30), T[i]
    f = (np.log(thr) - np.log(T0)) / (np.log(T1) - np.log(T0))
    return float(E_grid[i-1] + f * (E_grid[i] - E_grid[i-1])), False


def vicinal_profile(N, dx, s_mean, rng, dphi=0.85 * np.pi, lam=75.0):
    phi = np.zeros(N)
    pos, x = [], 0.0
    while True:
        x += rng.exponential(s_mean)
        if x >= N * dx:
            break
        pos.append(x)
    cur, p = 0.0, 0
    for nn in range(N):
        while p < len(pos) and pos[p] <= nn * dx:
            cur += dphi
            p += 1
        phi[nn] = cur
    return lam * np.exp(1j * phi)


def main():
    t0 = time.time()
    out = {"method": "dimension-generic RGF, 8-orbital valley cells, "
                     "normal leads mu=2000ueV, thr=0.01"}
    err = verify_against_msim()
    out["verify_vs_msim_maxerr_ueV"] = float(f"{err:.2e}")
    print(f"cell verification vs build_wire_two_valley_iv: {err:.2e} ueV")

    pars = dict(mu_ueV=35, B=1.5, Delta_ueV=50, alpha_eVA=0.05,
                m_rel=0.19, g=2.0)
    dx = 5e-9
    E_grid = np.arange(0.5, 30.1, 0.75)

    def run(vo, N):
        cells, hop = valley_cells(N, dx, pars["mu_ueV"], pars["B"],
                                  pars["Delta_ueV"], pars["alpha_eVA"],
                                  pars["m_rel"], pars["g"], vo)
        lead_cells, lead_hop = valley_cells(
            1, dx, 0, pars["B"], 0.0, pars["alpha_eVA"], pars["m_rel"],
            pars["g"], None, mu_lead_ueV=2000.0)
        T = transmission(E_grid, cells, hop, lead_cells[0])
        return transport_gap(E_grid, T)

    # clean wedge validation (uniform VO): spectral gap is ~22 ueV
    N0 = 300
    Et, cen = run(75 * np.ones(N0, complex), N0)
    out["clean_wedge"] = dict(transport_gap_ueV=round(Et, 1), censored=cen,
                              spectral_E2_ueV=22.0)
    print(f"clean wedge: E_T = {Et:.1f} ueV (spectral 22.0)")

    # single step (the 4-ueV junction bound state): resonance vs gap
    vo = 75 * np.exp(1j * np.where(np.arange(N0) < N0 // 2, 0, 0.85 * np.pi))
    Et, cen = run(vo, N0)
    out["single_step"] = dict(transport_gap_ueV=round(Et, 1), censored=cen,
                              spectral_E2_ueV=4.0)
    print(f"single step: E_T = {Et:.1f} ueV (spectral bound state 4.0)")

    # vicinal 50 nm: transport gap vs L (the L-scaling question)
    Ls = {}
    for N in (300, 600, 1200):
        vals = []
        for s in range(6):
            rng = np.random.default_rng([83, N, s])
            Et, cen = run(vicinal_profile(N, dx, 50e-9, rng), N)
            vals.append(Et)
        Ls[f"L{N*dx*1e6:.1f}um"] = dict(
            median=round(float(np.median(vals)), 2),
            q1=round(float(np.percentile(vals, 25)), 2),
            q3=round(float(np.percentile(vals, 75)), 2))
        print(f"vicinal 50nm L={N*dx*1e6:.1f}um: {Ls[f'L{N*dx*1e6:.1f}um']}",
              flush=True)
    out["vicinal_50nm_vs_L"] = Ls
    out["runtime_s"] = round(time.time() - t0, 1)
    tn = json.load(open('output/data/transport_numbers.json'))
    tn["valley"] = out
    json.dump(tn, open('output/data/transport_numbers.json', 'w'), indent=2)
    print("written transport_numbers.json [valley]")


if __name__ == "__main__":
    main()
