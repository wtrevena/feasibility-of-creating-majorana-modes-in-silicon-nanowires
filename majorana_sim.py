"""
majorana_sim.py
================
Corrected spinful Bogoliubov-de Gennes (BdG) simulation of a proximitized
semiconductor nanowire (Lutchyn-Oreg model), built to assess the feasibility
of Majorana zero modes (MZMs) in silicon.

Model (single band, continuum -> finite differences):

    h(k)  = (hbar^2 k^2 / 2m* - mu + V(x)) sigma_0 + alpha k sigma_z + E_Z sigma_x
    H_BdG = [[ h,            Delta (i sigma_y) ],
             [ (Delta i sigma_y)^dag,  -h*     ]]

Basis per site: Psi = (u_up, u_dn, v_up, v_dn).
The [[h, D], [D^dag, -h*]] construction with D^T = -D guarantees exact
particle-hole symmetry: if (u, v) is an eigenvector with energy E, then
(v*, u*) is an eigenvector with energy -E.

Topological criterion (single band, clean): E_Z^2 > Delta^2 + mu^2.

Units: SI Joules internally. Public API takes mu, Delta, E_v, disorder in ueV;
alpha in eV*Angstrom; B in Tesla; lengths in meters.

Two-valley extension (toy model for Si conduction band): two copies of the
wire with a coherent valley splitting E_v (nu_z) and residual random-phase
inter-valley scattering delta_iv; intra-valley singlet pairing.

Caveats (documented in RESULTS.md): single transverse subband; no orbital
magnetic-field effects; no self-energy treatment of the superconductor;
valley model is a minimal toy; disorder is onsite iid.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------- constants
HBAR = 1.054571817e-34       # J s
ME   = 9.1093837015e-31      # kg
QE   = 1.602176634e-19       # C
MU_B_EV = 5.7883818060e-5    # eV / T
UEV  = 1e-6 * QE             # 1 ueV in J
KB   = 1.380649e-23          # J / K

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# representative material parameter sets (single-band caricatures)
MATERIALS = {
    "Si":      dict(m=0.19, g=2.0),    # conduction band, transverse mass
    "InAs":    dict(m=0.026, g=15.0),
    "InSb":    dict(m=0.014, g=50.0),
    "GeSi_hh": dict(m=0.09,  g=5.0),   # Ge/Si core-shell holes (rough)
}


def EZ_J(g, B):
    """Zeeman energy (J) for g-factor g at field B (T)."""
    return 0.5 * abs(g) * MU_B_EV * B * QE


def is_topological(mu_ueV, B, Delta_ueV, g):
    """Clean single-band criterion E_Z^2 > Delta^2 + mu^2."""
    EZ = EZ_J(g, B)
    return EZ**2 > (Delta_ueV * UEV)**2 + (mu_ueV * UEV)**2


# ------------------------------------------------------------- bulk (k-space)
def bulk_E2_lower(k, mu_J, EZ, Delta_J, alpha_SI, m_kg):
    """Lower BdG branch squared, E_-^2(k), from the analytic dispersion
    E^2 = xi^2 + a^2 + EZ^2 + D^2 +- 2 sqrt(xi^2 a^2 + EZ^2 (xi^2 + D^2)),
    a = alpha k, xi = hbar^2 k^2/2m - mu."""
    xi = HBAR**2 * k**2 / (2 * m_kg) - mu_J
    a = alpha_SI * k
    root = np.sqrt(xi**2 * a**2 + EZ**2 * (xi**2 + Delta_J**2))
    e2 = xi**2 + a**2 + EZ**2 + Delta_J**2 - 2 * root
    return np.maximum(e2, 0.0)


def bulk_gap_ueV(mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g, nk=6001):
    """Bulk excitation gap (ueV) of the infinite clean wire: min_k E_-(k)."""
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    kF = np.sqrt(2 * m * (abs(mu) + EZ + D)) / HBAR
    kso = m * aSI / HBAR**2
    kmax = 4.0 * (kF + kso) + 2e7
    k = np.linspace(0, kmax, nk)
    e2 = bulk_E2_lower(k, mu, EZ, D, aSI, m)
    return float(np.sqrt(e2.min()) / UEV)


def topological_gap_ueV(mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g, nk=6001):
    """Bulk gap if topological, else 0 (protection only counts in the phase)."""
    if not is_topological(mu_ueV, B, Delta_ueV, g):
        return 0.0
    return bulk_gap_ueV(mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g, nk)


# --------------------------------------------------------------- finite wire
def _normal_h(N, dx, mu_J, EZ, alpha_SI, m_kg, V_J):
    """Sparse 2N x 2N normal-state Hamiltonian h (site (x) spin)."""
    t = HBAR**2 / (2 * m_kg * dx**2)
    aso = alpha_SI / (2 * dx)
    onsite = (2 * t - mu_J) * s0 + EZ * sx
    hop = -t * s0 - 1j * aso * sz          # block from site n to n+1
    diag_blocks = [onsite + V_J[n] * s0 for n in range(N)]
    h = sp.block_diag(diag_blocks, format="lil")
    K = sp.diags(np.ones(N - 1), 1, format="csr")
    h = (h + sp.kron(K, hop) + sp.kron(K.T, hop.conj().T)).tocsr()
    return h


def build_wire(N, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
               disorder_ueV=0.0, rng=None):
    """Sparse 4N x 4N BdG Hamiltonian. Basis: [u (site x spin), v (site x spin)].
    disorder_ueV: onsite potential drawn uniform in [-W, W]."""
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    if disorder_ueV:
        if rng is None:
            raise ValueError("disorder_ueV > 0 requires an explicit rng")
        V = rng.uniform(-disorder_ueV, disorder_ueV, N) * UEV
    else:
        V = np.zeros(N)
    h = _normal_h(N, dx, mu, EZ, aSI, m, V)
    Dm = sp.kron(sp.eye(N), D * (1j * sy)).tocsr()      # D^T = -D  (singlet)
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


def build_wire_two_valley(N, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
                          Ev_ueV=0.0, delta_iv_ueV=0.0,
                          disorder_ueV=0.0, rng=None):
    """Two-valley toy model: 8N x 8N. Internal order per site: valley (x) spin.

    Ev (nu_z): coherent valley splitting (basis chosen so the smooth
    valley-orbit coupling is diagonal).
    delta_iv: residual inter-valley scattering with a random phase per site,
    delta_iv (cos th_n nu_x + sin th_n nu_y). The 2k0 inter-valley momentum
    transfer makes the phase effectively random on a 5 nm lattice; a uniform
    phase would be a mere valley rotation and could not hybridize the
    Majorana pairs. Pairing is intra-valley singlet (toy; see RESULTS.md).
    Requires rng if delta_iv > 0."""
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    Ev = Ev_ueV * UEV
    div = delta_iv_ueV * UEV
    t = HBAR**2 / (2 * m * dx**2)
    aso = aSI / (2 * dx)
    if (disorder_ueV or delta_iv_ueV) and rng is None:
        raise ValueError("disorder_ueV or delta_iv_ueV > 0 requires an explicit rng")
    if disorder_ueV:
        V = rng.uniform(-disorder_ueV, disorder_ueV, N) * UEV
    else:
        V = np.zeros(N)
    v0 = np.eye(2)
    vx = np.array([[0, 1], [1, 0]], dtype=complex)
    vy = np.array([[0, -1j], [1j, 0]])
    vz = np.diag([1.0, -1.0]).astype(complex)
    onsite_spin = (2 * t - mu) * s0 + EZ * sx
    base = np.kron(v0, onsite_spin) + 0.5 * Ev * np.kron(vz, s0)
    th = rng.uniform(0, 2 * np.pi, N) if div else np.zeros(N)
    hop = np.kron(v0, -t * s0 - 1j * aso * sz)
    diag_blocks = [base + V[n] * np.kron(v0, s0)
                   + div * np.kron(np.cos(th[n]) * vx + np.sin(th[n]) * vy, s0)
                   for n in range(N)]
    h = sp.block_diag(diag_blocks, format="lil")
    K = sp.diags(np.ones(N - 1), 1, format="csr")
    h = (h + sp.kron(K, hop) + sp.kron(K.T, hop.conj().T)).tocsr()
    Dm = sp.kron(sp.eye(N), D * np.kron(v0, 1j * sy)).tocsr()
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


# ------------------------------------------------------------------ solvers
def solve_lowest(H, k=8, sigma=1e-30):
    """k eigenpairs closest to zero energy, sorted by |E|. Returns (E_J, V)."""
    vals, vecs = eigsh(H, k=k, sigma=sigma, which="LM")
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:, order]


def site_density(vec, N):
    """Total |psi|^2 per site, summed over spin (x valley) and e/h sectors."""
    half = vec.shape[0] // 2
    ni = half // N
    u = np.abs(vec[:half].reshape(N, ni))**2
    v = np.abs(vec[half:].reshape(N, ni))**2
    return u.sum(axis=1) + v.sum(axis=1)


def end_weight(density, frac=0.1):
    """Fraction of probability in the outer `frac` of the wire on each side."""
    N = len(density)
    n = max(1, int(frac * N))
    return float((density[:n].sum() + density[-n:].sum()) / density.sum())


def majorana_metrics(H, N, k=6):
    """Convenience: (E0_ueV, Egap_ueV, end_weight) of the state closest to 0.
    Egap is the 3rd-smallest |E|: it equals the excitation gap only when exactly
    one near-zero pair exists (topological phase, single channel). In the trivial
    phase it is the second excitation; with two near-zero pairs it is the second
    splitting. Interpret accordingly."""
    E, V = solve_lowest(H, k=k)
    Eabs = np.abs(E) / UEV
    dens = site_density(V[:, 0], N)
    return float(Eabs[0]), float(Eabs[2]), end_weight(dens)


# ---------------------------------------- physical inter-valley pairing model
def step_phase_profile(N, dx, L_step, rng):
    """Piecewise-constant valley-orbit phase phi(x): interface atomic steps at
    Poisson-distributed positions (mean spacing L_step) randomize the phase.
    Use Ev/2 * exp(1j*phi) as vo_profile_ueV for build_wire_two_valley_iv."""
    phi = np.zeros(N)
    cur = rng.uniform(0, 2 * np.pi)
    for n in range(N):
        if rng.random() < dx / L_step:
            cur = rng.uniform(0, 2 * np.pi)
        phi[n] = cur
    return phi


def build_wire_two_valley_iv(N, dx, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
                             Ev_ueV=0.0, vo_profile_ueV=None,
                             valley_pol_ueV=0.0,
                             soc_mode="rashba",
                             disorder_ueV=0.0, rng=None):
    """Two-valley wire with the *physical* pairing channel.

    Valleys at +-k0 are time-reversed partners, so a uniform s-wave parent
    induces zero-momentum INTER-VALLEY singlet pairing D = Delta (nu_x x i sigma_y)
    (antisymmetric: D^T = -D). Valley splitting enters as valley-orbit coupling
    (Ev/2) [cos phi(x) nu_x + sin phi(x) nu_y]; a perfect interface has uniform
    phi, while atomic steps randomize phi(x) (use step_phase_profile).
    Key algebraic fact (verified numerically): for ANY uniform in-plane
    valley-orbit phase the model is unitarily equivalent to two decoupled
    bands at mu -+ |lambda| with pairing +-Delta, so a smooth interface is
    benign regardless of phase. The physical threats are (i) phase GRADIENTS
    (atomic steps -> vo_profile_ueV varying along x), and (ii) valley
    POLARIZATION valley_pol_ueV (nu_z, TRS-breaking in valley space), which is
    a genuine pair-breaker for inter-valley pairing.
    soc_mode: "rashba" -> valley-scalar SOC alpha k nu_0 sigma_z.
              "dresselhaus" -> the physical interface (Dresselhaus-like) SOC,
              which is PHASE-LOCKED to the valley-orbit phase:
              alpha k (cos phi nu_x + sin phi nu_y) sigma_z, with phi(x) taken
              from vo_profile_ueV (bond-averaged). This is time-reversal even
              (TR operator: nu_x i sigma_y K) and gives opposite SOC signs in
              the two valley-split EIGENSTATES, as measured at Si interfaces.
              For a uniform phase it preserves the exact two-band equivalence
              with full SOC magnitude in each band.
              (The previous nu_z sigma_z k option was TR-odd / unphysical and
              has been removed; see RESULTS.md corrections log.)
    vo_profile_ueV: complex per-site lambda(x); splitting = 2|lambda|.
    """
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    if vo_profile_ueV is None:
        lam = np.full(N, 0.5 * Ev_ueV) * UEV          # uniform, splitting = Ev
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
    if soc_mode == "rashba":
        hop_soc = np.kron(v0, -1j * aso * sz)
        h = (h.tocsr() + sp.kron(K, hop_soc)
             + sp.kron(K.T, hop_soc.conj().T)).tocsr()
    elif soc_mode == "dresselhaus":
        # bond phase = CIRCULAR mean of neighboring valley-orbit phases
        # (arithmetic mean of np.angle outputs is not gauge covariant across
        # the +-pi branch cut; fixed after review round 3). Sites with lam=0
        # get phase 0 by convention and full SOC magnitude (documented).
        ph_site = np.exp(1j * np.angle(lam + (np.abs(lam) < 1e-300) * 1.0))
        hb = sp.lil_matrix((4 * N, 4 * N), dtype=complex)
        for n in range(N - 1):
            zb = ph_site[n] + ph_site[n + 1]
            ph = np.angle(zb) if abs(zb) > 1e-12 else 0.0
            Mv = np.cos(ph) * vx + np.sin(ph) * vy
            blk = np.kron(Mv, -1j * aso * sz)
            hb[4*n:4*n+4, 4*(n+1):4*(n+1)+4] = blk
        hb = hb.tocsr()
        h = (h.tocsr() + hb + hb.conj().T).tocsr()
    else:
        raise ValueError(f"unknown soc_mode {soc_mode!r}")
    h = h.tocsr()
    Dm = sp.kron(sp.eye(N), D * np.kron(vx, 1j * sy)).tocsr()   # inter-valley
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


# --------------------------------------------------- multi-subband 2D strip
def build_wire_2d(Nx, Ny, dx, dy, mu_ueV, B, Delta_ueV, alpha_eVA, m_rel, g,
                  disorder_ueV=0.0, rng=None):
    """2D strip (Nx x Ny sites) with 2D Rashba SOC, B along the wire (x, in
    plane: no flux threads the lattice plaquettes, so there is no Peierls
    phase in this geometry), and s-wave pairing. Internal order: site
    (ix*Ny+iy) x spin; BdG doubles to 4*Nx*Ny.

    H_R = alpha (sigma_x k_y - sigma_y k_x), E_Z along sigma_x (parallel to
    the k_y SOC component, perpendicular to the k_x component, as physical).
    """
    m = m_rel * ME
    mu = mu_ueV * UEV
    D = Delta_ueV * UEV
    EZ = EZ_J(g, B)
    aSI = alpha_eVA * 1e-10 * QE
    tx = HBAR**2 / (2 * m * dx**2)
    ty = HBAR**2 / (2 * m * dy**2)
    Ns = Nx * Ny
    if disorder_ueV:
        if rng is None:
            raise ValueError("disorder_ueV > 0 requires an explicit rng")
        V = rng.uniform(-disorder_ueV, disorder_ueV, Ns) * UEV
    else:
        V = np.zeros(Ns)
    onsite = (2 * tx + 2 * ty - mu) * s0 + EZ * sx
    hop_x = -tx * s0 + 1j * (aSI / (2 * dx)) * sy    # from -alpha sigma_y k_x
    hop_y = -ty * s0 - 1j * (aSI / (2 * dy)) * sx    # from +alpha sigma_x k_y
    Kx = sp.kron(sp.diags(np.ones(Nx - 1), 1), sp.eye(Ny), format="csr")
    Ky = sp.kron(sp.eye(Nx), sp.diags(np.ones(Ny - 1), 1), format="csr")
    h = (sp.kron(sp.diags(V), s0)
         + sp.kron(sp.eye(Ns), onsite)
         + sp.kron(Kx, hop_x) + sp.kron(Kx.T, hop_x.conj().T)
         + sp.kron(Ky, hop_y) + sp.kron(Ky.T, hop_y.conj().T)).tocsr()
    Dm = sp.kron(sp.eye(Ns), D * (1j * sy)).tocsr()
    H = sp.bmat([[h, Dm], [Dm.conj().T, -h.conj()]], format="csc")
    return H


def subband_bottoms_ueV(Ny, dy, m_rel):
    """Hard-wall transverse subband energies (ueV) above the 2D band bottom
    used in build_wire_2d (lattice dispersion, measured from the mu origin)."""
    m = m_rel * ME
    ty = HBAR**2 / (2 * m * dy**2)
    j = np.arange(1, Ny + 1)
    return (2 * ty * (1 - np.cos(j * np.pi / (Ny + 1)))) / UEV
