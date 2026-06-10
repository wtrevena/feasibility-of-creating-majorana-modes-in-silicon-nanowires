"""
lk_holes.py — 4-band Luttinger-Kohn model of a Si FinFET hole channel.

Purpose: justify (or erode) the single-band hole parameter box used in fig8
by computing, from k.p theory, the gate-field dependence of the quantities
the Majorana optimizer consumes: effective mass m*, direct-Rashba SOC alpha,
and the g-tensor (gx along the wire, gy in-plane transverse, gz out-of-plane)
INCLUDING their covariation under the same vertical gate field E_z.

Model: Gamma_8 4-band LK Hamiltonian (hole picture, energies positive into
the valence band), hard-wall rectangular fin cross-section (y = fin width,
z = vertical), wire along x. Vertical field enters as U = e E_z z; magnetic
field enters via BOTH the bare Zeeman term 2 kappa mu_B B.J and orbital
substitution k -> k + eA/hbar with A = (B_y z - B_z y, 0, B_x y).

Si Luttinger parameters: gamma1 = 4.285, gamma2 = 0.339, gamma3 = 1.446
(Lawaetz 1971); kappa = -0.42. Caveats: no strain, no cubic q J^3 Zeeman
term, hard-wall caricature of the real tri-gate electrostatics, no
split-off band (Si Delta_SO = 44 meV is small — 6-band corrections are the
first upgrade).

Validation (run `python3 lk_holes.py`):
  V1. bulk masses along [100]: m_HH = 1/(g1-2g2) = 0.277, m_LH = 0.201;
  V2. Hermiticity of the assembled Hamiltonian;
  V3. alpha(E_z = 0) = 0 by inversion symmetry;
  V4. l_so = hbar^2/(m* alpha) lands in the measured FinFET range 20-100 nm
      at realistic fields.
"""

import numpy as np

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
QE = 1.602176634e-19
MU_B_J = 9.2740100783e-24      # J/T

G1, G2, G3 = 4.285, 0.339, 1.446
KAPPA = -0.42

s3 = np.sqrt(3.0)
JX = 0.5 * np.array([[0, s3, 0, 0], [s3, 0, 2, 0],
                     [0, 2, 0, s3], [0, 0, s3, 0]], dtype=complex)
JY = 0.5j * np.array([[0, -s3, 0, 0], [s3, 0, -2, 0],
                      [0, 2, 0, -s3], [0, 0, s3, 0]], dtype=complex)
JZ = np.diag([1.5, 0.5, -0.5, -1.5]).astype(complex)
I4 = np.eye(4, dtype=complex)


def _sym(A, B):
    return 0.5 * (A @ B + B @ A)


def _d1(n, d):
    m = np.zeros((n, n))
    for i in range(n - 1):
        m[i, i + 1] = 1.0 / (2 * d)
        m[i + 1, i] = -1.0 / (2 * d)
    return m


def _d2(n, d):
    m = -2.0 * np.eye(n)
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m / d**2


def build_fin_H(kx, Ez_Vm=0.0, B=(0.0, 0.0, 0.0),
                Wy=10e-9, Wz=12e-9, ny=11, nz=13):
    """Dense 4*ny*nz Hamiltonian of the fin cross-section at wire momentum kx.
    Returns (H, dims). Energies in Joules (hole picture, positive)."""
    dy, dz = Wy / (ny + 1), Wz / (nz + 1)
    y = (np.arange(ny) + 1) * dy - Wy / 2          # centered
    z = (np.arange(nz) + 1) * dz - Wz / 2
    Iy, Iz = np.eye(ny), np.eye(nz)
    Yd = np.kron(np.diag(y), Iz)
    Zd = np.kron(Iy, np.diag(z))
    Ig = np.eye(ny * nz)
    Bx, By, Bz = B
    c = QE / HBAR
    # momentum operators on the grid (units 1/m), with orbital substitution
    KX = kx * Ig + c * (By * Zd - Bz * Yd)
    KY = -1j * np.kron(_d1(ny, dy), Iz)
    KZ = -1j * np.kron(Iy, _d1(nz, dz)) + c * Bx * Yd
    KX2 = KX @ KX
    KY2 = -np.kron(_d2(ny, dy), Iz) + 0j
    KZ2raw = -np.kron(Iy, _d2(nz, dz))
    KZ2 = KZ2raw + c * Bx * (KZsym := _sym(-1j * np.kron(Iy, _d1(nz, dz)),
                                           Yd)) * 2 + (c * Bx * Yd) @ (c * Bx * Yd) \
        if Bx else KZ2raw + 0j
    K2 = KX2 + KY2 + KZ2
    pref = HBAR**2 / (2 * M0)
    H = pref * ((G1 + 2.5 * G2) * np.kron(K2, I4)
                - 2 * G2 * (np.kron(KX2, JX @ JX) + np.kron(KY2, JY @ JY)
                            + np.kron(KZ2, JZ @ JZ))
                - 4 * G3 * (np.kron(_sym(KX, KY), _sym(JX, JY))
                            + np.kron(_sym(KY, KZ), _sym(JY, JZ))
                            + np.kron(_sym(KZ, KX), _sym(JZ, JX))))
    H = H + np.kron(QE * Ez_Vm * Zd, I4)
    H = H + 2 * KAPPA * MU_B_J * np.kron(Ig, Bx * JX + By * JY + Bz * JZ)
    return H


def extract(Ez_Vm, Wy=10e-9, Wz=12e-9, ny=11, nz=13,
            k1=2e7, Btest=0.05):
    """(m*/m0, alpha_eVA, gx, gy, gz, nso) at gate field Ez."""
    e0 = np.linalg.eigvalsh(build_fin_H(0.0, Ez_Vm, (0, 0, 0), Wy, Wz, ny, nz))[:2]
    ek = np.linalg.eigvalsh(build_fin_H(k1, Ez_Vm, (0, 0, 0), Wy, Wz, ny, nz))[:2]
    alpha_Jm = (ek[1] - ek[0]) / (2 * k1)
    dE = ek.mean() - e0.mean()
    mstar = HBAR**2 * k1**2 / (2 * dE) / M0 if dE > 0 else np.nan
    gs = []
    for ax in range(3):
        Bv = [0.0, 0.0, 0.0]; Bv[ax] = Btest
        eb = np.linalg.eigvalsh(build_fin_H(0.0, Ez_Vm, tuple(Bv),
                                            Wy, Wz, ny, nz))[:2]
        gs.append((eb[1] - eb[0]) / (MU_B_J * Btest))
    # SOC axis: <J> difference of the kx-split doublet (pseudospin direction)
    w, v = np.linalg.eigh(build_fin_H(k1, Ez_Vm, (0, 0, 0), Wy, Wz, ny, nz))
    nso = []
    Ng = (v.shape[0] // 4)
    for Jm in (JX, JY, JZ):
        Op = np.kron(np.eye(Ng), Jm)
        nso.append(float(np.real(v[:, 0].conj() @ Op @ v[:, 0]
                                 - v[:, 1].conj() @ Op @ v[:, 1])))
    nso = np.array(nso)
    nrm = np.linalg.norm(nso)
    nso = nso / nrm if nrm > 1e-9 else nso
    return (float(mstar), float(alpha_Jm / QE / 1e-10), gs[0], gs[1], gs[2],
            nso.tolist())


if __name__ == "__main__":
    # V1: bulk masses along [100]
    for k in (2e8,):
        Hb = (HBAR**2 / (2 * M0)) * ((G1 + 2.5 * G2) * k**2 * I4
                                     - 2 * G2 * k**2 * (JX @ JX))
        ev = np.linalg.eigvalsh(Hb)
        mh = HBAR**2 * k**2 / (2 * ev[0]) / M0
        ml = HBAR**2 * k**2 / (2 * ev[-1]) / M0
        print(f"V1 bulk [100]: m_HH = {mh:.3f} (expect 0.277), "
              f"m_LH = {ml:.3f} (expect 0.201)")
    # V2: Hermiticity incl. field+orbital terms
    H = build_fin_H(2e7, 3e7, (0.05, 0.05, 0.05))
    print(f"V2 hermiticity: max|H-H^dag| = {np.abs(H - H.conj().T).max():.2e} J")
    # V3/V4: alpha(E)
    for Ez in (0.0, 1e7, 3e7):
        m, al, gx, gy, gz, nso = extract(Ez)
        lso = HBAR**2 / (m * M0 * al * QE * 1e-10) * 1e9 if al > 1e-6 else np.inf
        print(f"Ez={Ez/1e6:5.0f} MV/m: m*={m:.3f}  alpha={al:.4f} eV.A  "
              f"l_so={lso:7.1f} nm  g=({gx:.2f},{gy:.2f},{gz:.2f})  "
              f"nso=({nso[0]:.2f},{nso[1]:.2f},{nso[2]:.2f})")
