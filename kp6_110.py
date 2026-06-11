"""
kp6_110.py — [110]-channel rotation of the validated 6-band (Gamma8+Gamma7)
LKBP fin model of kp6_holes.py: the decisive orientation test for the
hole-platform g-factor analysis.

Question answered: every experimental benchmark device (Geyer 2024 and
Camenzind 2022 FinFETs, Voisin 2016 and Crippa 2018 SOI nanowires; see
drafts/kp6_benchmarks.md) has its channel along [110] on a (001) surface,
while kp6_holes.py and lk_holes.py put the wire along [100]. The published
comparison "model g_x = 0.4-1.1 vs measured 1.9-2.3" therefore compares
different crystallographic axes. This module rotates the CRYSTAL relative to
the device — device frame x' = [110]/sqrt2 (wire), y' = [-110]/sqrt2
(in-plane transverse), z' = [001] (vertical, gate field) — and recomputes
the exact counterparts of the kp6 tables, isolating the orientation effect
with everything else (hard wall, no strain, no inhomogeneous electrostatics)
held fixed.

IMPLEMENTATION (rotate the L,M,N kinetic tensor; keep the basis pipeline)
-------------------------------------------------------------------------
The p-multiplet orbital kinetic Hamiltonian of kp6_holes is
H_ab(k) = sum_cd T[a,b,c,d] k_c k_d with the L,M,N tensor in crystal axes
(a,b orbital, c,d momentum slots). The device-frame Hamiltonian follows by
substituting k = R^T k' (R = rotation crystal->device, +45 deg about z),
i.e. contracting the momentum slots with R^T while KEEPING the crystal
orbital basis (X,Y,Z) — and hence the identical U6 |J,mJ> projection,
CAB orbital projectors and Zeeman machinery of kp6_holes:

    C[a,b,i,j] = sum_cd T[a,b,c,d] R^T[c,i] R^T[d,j]
    H'_ab(k')  = sum_ij C[a,b,i,j] {k'_i k'_j}_sym .

Envelope operators (hard-wall sine/Galerkin basis — the kp6 production
mode; the FD branch is not duplicated here), the gate field U = +e E_z z
and the Peierls substitution A = (B_y' z - B_z' y, 0, B_x' y) all live in
the DEVICE frame (they are geometric); the Zeeman term is the crystal-frame
H_Z = mu_B (g_L^h B_c.L + g_s^h B_c.S) with B_c = R^T B_device (B.L and B.S
are rotation scalars, so the crystal-axis Lj, Sj matrices of kp6_holes are
reused unchanged). R = identity reproduces kp6_holes.build_fin_H6/extract6
to machine precision (validated below). Reported g components and the SOC
axis n_so are DEVICE-frame: (g_x', g_y', g_z) = (along [110] wire,
along [-110], along [001]); n_so_dev = R n_so_crystal.

VALIDATIONS (section A; gate B-D, mirroring kp6_holes)
------------------------------------------------------
 a   bulk k = 0 unmoved by rotation: Gamma8 quadruplet at 0, Gamma7 at
     +44 meV (machine precision);
 b   bulk dispersion of the ROTATED model along device x', y', z' and 5
     random directions equals the UNROTATED bulk_H6 at the mapped crystal
     momentum k = R^T k' (machine precision); the [110] branches are
     themselves pinned by the closed-form Gamma8 4-band [110] masses
     m_HH/LH = 1/(g1 -/+ sqrt(g2^2 + 3 g3^2)) = 0.569 / 0.147 m0;
 c   isotropic limit: with g2 = g3 set artificially equal the rotated and
     unrotated fin extractions must coincide (the spherical Hamiltonian
     makes the orbital-slot rotation a pure internal unitary) — gate
     < 0.5 %, expected ~ machine;
 c2  R = identity reproduces kp6_holes.extract6 (production dst basis) to
     < 1e-6 % — pins every convention against the [100] code;
 d   Kramers degeneracy at B = 0 (< 1e-10 split/spacing) and hermiticity
     with kx, Ez and all three B components on (< 1e-25 J);
 e   basis-size convergence at the production point (11x13 -> 15x17 modes,
     < 3 % like kp6_holes).

Sections (python kp6_110.py --sec A|B|C|D|all), checkpointed in
output/data/kp6_110.json, ledger tag "kp6_110":
  A  validation gauntlet above;
  B  extract6_rot vs Ez in {2,5,10,15,20,30,40,50} MV/m at the 10x12 nm
     production fin — the direct [110] counterpart of kp6 section B;
  C  seven-geometry bracket (same GEOMS as kp6 section C, cross-checked
     against output/data/kp6.json) — the [110] g_x' range vs [100] g_x;
  D  benchmark comparison: closest rows to the Geyer lab-frame tensors
     (2.31, 2.00, 1.50) / (1.86, 2.76, 1.46), Camenzind g* = 1.94-2.35
     (B in-plane perp to fin = device y'), Venitucci g_z ~ 4.66-5 anchor,
     and the gap-closure fraction for the wire-axis g deficit.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:  # sandbox without scipy/matplotlib: use the repo's compat shim
    import scipy.sparse.linalg  # noqa: F401
    import matplotlib.pyplot    # noqa: F401
except Exception:
    sys.path.insert(0, os.path.join(_HERE, "compat"))

import kp6_holes
from lk_holes import HBAR, M0, QE, MU_B_J, G1, G2, G3, _sym
from kp6_holes import (U6, I2, I6, P7, CAB, J6, _zeeman6, _dst_ops,
                       bulk_H6, _lk_bulk4, GEOMS, MEV_J, DSO_MEV,
                       _row, NAMES, _pct, save_numbers)

DATA = os.path.join(_HERE, "output", "data")
os.makedirs(DATA, exist_ok=True)
CKPT = os.path.join(DATA, "kp6_110.json")
KP6_CKPT = os.path.join(DATA, "kp6.json")

PROD = dict(Wy=10e-9, Wz=12e-9, ny=11, nz=13, Ez=1e7)

SQ2 = np.sqrt(2.0)
# rows = device axes in crystal coordinates (rotation crystal -> device):
# x' = [110]/sq2 (wire), y' = [-110]/sq2, z' = [001]
R110 = np.array([[1.0, 1.0, 0.0],
                 [-1.0, 1.0, 0.0],
                 [0.0, 0.0, SQ2]]) / SQ2
RID = np.eye(3)

# benchmark anchors (drafts/kp6_benchmarks.md)
GEYER = {"Q1": (2.31, 2.00, 1.50), "Q2": (1.86, 2.76, 1.46)}
GEYER_GXX = [1.86, 2.31]            # lab-frame xx = along-fin [110]
CAMENZIND_GY = [1.94, 2.35]         # B in-plane perp to fin = device y'
VENITUCCI_GZ = [4.66, 5.0]          # 6|kappa| + 2.14 envelope correction


# ------------------------------------------------------ rotated kinetic part
def _Ctensor(R, g1=G1, g2=G2, g3=G3):
    """C[a,b,i,j]: L,M,N kinetic tensor with crystal orbital slots (a,b)
    and DEVICE momentum slots (i,j), via k_crystal = R^T k_device."""
    pref = HBAR ** 2 / (2 * M0)
    L, M, N = pref * (g1 + 4 * g2), pref * (g1 - 2 * g2), pref * 6 * g3
    T = np.zeros((3, 3, 3, 3))
    for a in range(3):
        T[a, a, :, :] += M * np.eye(3)
        T[a, a, a, a] += L - M
        for b in range(3):
            if a != b:
                T[a, b, a, b] += N / 2.0
                T[a, b, b, a] += N / 2.0
    Rt = R.T
    return np.einsum("abcd,ci,dj->abij", T, Rt, Rt)


def bulk_H6_rot(kdev, Dso_meV=DSO_MEV, R=R110, g=(G1, G2, G3)):
    """Bulk 6x6 of the rotated model at numeric DEVICE-frame k' (Joules),
    built from the same C tensor as the fin Hamiltonian."""
    C = _Ctensor(R, *g)
    k = np.asarray(kdev, float)
    Hor = np.einsum("abij,i,j->ab", C, k, k).astype(complex)
    return U6.conj().T @ np.kron(I2, Hor) @ U6 + Dso_meV * MEV_J * P7


# ------------------------------------------------------------ rotated fin
def build_fin_H6_rot(kx, Ez_Vm=0.0, B=(0.0, 0.0, 0.0),
                     Wy=10e-9, Wz=12e-9, ny=11, nz=13, Dso_meV=DSO_MEV,
                     R=R110, g=(G1, G2, G3)):
    """Dense 6*ny*nz fin Hamiltonian at DEVICE-frame wire momentum kx,
    sine/Galerkin (dst) basis — kp6_holes production mode. kx, B, geometry
    in device coordinates; R = identity reproduces kp6_holes.build_fin_H6
    (basis='dst') exactly. Energies in Joules (hole picture, positive)."""
    C = _Ctensor(R, *g)
    Iy, Iz = np.eye(ny), np.eye(nz)
    Ig = np.eye(ny * nz)
    Bx, By, Bz = B
    c = QE / HBAR
    K2y, D1y, Xy = _dst_ops(ny, Wy)
    K2z, D1z, Xz = _dst_ops(nz, Wz)
    Yd = np.kron(Xy, Iz)
    Zd = np.kron(Iy, Xz)
    KYb = -1j * np.kron(D1y, Iz)
    KZb = -1j * np.kron(Iy, D1z)
    KY2 = np.kron(K2y, Iz) + 0j
    KZ2b = np.kron(Iy, K2z) + 0j
    KX = kx * Ig + c * (By * Zd - Bz * Yd)
    KY = KYb
    KZ = KZb + c * Bx * Yd
    # exact Galerkin second derivatives on the diagonal momentum slots,
    # Peierls-corrected exactly as kp6_holes.build_fin_H6
    K2 = [KX @ KX,
          KY2,
          KZ2b + (2 * c * Bx * _sym(KZb, Yd) + (c * Bx) ** 2 * (Yd @ Yd)
                  if Bx else 0)]
    Kop = [KX, KY, KZ]
    n6 = 6 * ny * nz
    H = np.zeros((n6, n6), complex)
    for a in range(3):
        for b in range(3):
            Op = sum(C[a, b, i, i] * K2[i] for i in range(3))
            for i in range(3):
                for j in range(i + 1, 3):
                    if C[a, b, i, j]:
                        Op = Op + 2.0 * C[a, b, i, j] * _sym(Kop[i], Kop[j])
            H += np.kron(Op, CAB[a][b])
    H += np.kron(QE * Ez_Vm * Zd, I6)
    H += Dso_meV * MEV_J * np.kron(Ig, P7)
    if any(B):
        Bc = tuple(R.T @ np.asarray(B, float))     # crystal-frame B
        H += np.kron(Ig, _zeeman6(Bc))
    return H


def extract6_rot(Ez_Vm, Wy=10e-9, Wz=12e-9, ny=11, nz=13,
                 k1=2e7, Btest=0.05, Dso_meV=DSO_MEV,
                 R=R110, g=(G1, G2, G3)):
    """(m*/m0, alpha_eVA, gx', gy', gz, nso_device) for the lowest Kramers
    doublet — identical extraction recipes to kp6_holes.extract6, axes in
    the device frame (x' = wire = [110])."""
    kw = dict(Wy=Wy, Wz=Wz, ny=ny, nz=nz, Dso_meV=Dso_meV, R=R, g=g)
    e0 = np.linalg.eigvalsh(build_fin_H6_rot(0.0, Ez_Vm, (0, 0, 0), **kw))[:2]
    ek = np.linalg.eigvalsh(build_fin_H6_rot(k1, Ez_Vm, (0, 0, 0), **kw))[:2]
    alpha_Jm = (ek[1] - ek[0]) / (2 * k1)
    dE = ek.mean() - e0.mean()
    mstar = HBAR**2 * k1**2 / (2 * dE) / M0 if dE > 0 else np.nan
    gs = []
    for ax in range(3):
        Bv = [0.0, 0.0, 0.0]
        Bv[ax] = Btest
        eb = np.linalg.eigvalsh(
            build_fin_H6_rot(0.0, Ez_Vm, tuple(Bv), **kw))[:2]
        gs.append((eb[1] - eb[0]) / (MU_B_J * Btest))
    w, v = np.linalg.eigh(build_fin_H6_rot(k1, Ez_Vm, (0, 0, 0), **kw))
    Ng = v.shape[0] // 6
    nso_c = []
    for Jm in J6:
        Op = np.kron(np.eye(Ng), Jm)
        nso_c.append(float(np.real(v[:, 0].conj() @ Op @ v[:, 0]
                                   - v[:, 1].conj() @ Op @ v[:, 1])))
    nso = np.asarray(R, float) @ np.array(nso_c)   # device frame
    nrm = np.linalg.norm(nso)
    nso = nso / nrm if nrm > 1e-9 else nso
    return (float(mstar), float(alpha_Jm / QE / 1e-10),
            float(gs[0]), float(gs[1]), float(gs[2]), nso.tolist())


# ------------------------------------------------------------- checkpointing
def _load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def _save(d):
    with open(CKPT, "w") as f:
        json.dump(d, f, indent=2)


def _kp6():
    if os.path.exists(KP6_CKPT):
        with open(KP6_CKPT) as f:
            return json.load(f)
    return {}


# ------------------------------------------------------------------ sections
def sec_A():
    """Validation gauntlet a-e. All must pass before physics sections."""
    t0 = time.time()
    out, ok = {}, True

    # (a) bulk k = 0 unmoved by the rotation
    ev = np.linalg.eigvalsh(bulk_H6_rot((0.0, 0.0, 0.0)))
    g8 = float(np.abs(ev[:4]).max() / MEV_J)
    g7 = ev[4:] / MEV_J
    pa = g8 < 1e-12 and np.abs(g7 - DSO_MEV).max() < 1e-10
    out["a_bulk_k0"] = dict(gamma8_max_abs_meV=g8,
                            gamma7_meV=[float(x) for x in g7],
                            passed=bool(pa))
    ok &= pa

    # (b) rotated bulk along device axes + random k' == unrotated bulk at
    # the mapped crystal momentum k = R^T k' (machine precision)
    kmag = 2.3e8
    dev = 0.0
    dirs = [np.array(d) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    rng = np.random.default_rng(11)
    dirs += [rng.uniform(-1, 1, 3) for _ in range(5)]
    for n in dirs:
        n = n / np.linalg.norm(n)
        kdev = n * kmag
        e_rot = np.linalg.eigvalsh(bulk_H6_rot(tuple(kdev)))
        e_ref = np.linalg.eigvalsh(bulk_H6(tuple(R110.T @ kdev)))
        dev = max(dev, float(np.abs(e_rot - e_ref).max()
                             / np.abs(e_ref).max()))
    # closed-form Gamma8 [110] masses pin the underlying [110] branches
    s = np.sqrt(G2**2 + 3 * G3**2)
    kt = HBAR**2 * kmag**2 / (2 * M0)
    kc = np.array([1.0, 1.0, 0.0]) / SQ2 * kmag
    e4 = np.linalg.eigvalsh(_lk_bulk4(*kc))
    expect4 = np.sort([G1 - s, G1 - s, G1 + s, G1 + s]) * kt
    dev4 = float(np.abs((e4 - expect4) / expect4).max())
    pb = dev < 1e-12 and dev4 < 1e-10
    out["b_bulk_rotation"] = dict(
        max_rel_dev_rotated_vs_mapped=dev,
        m_hh_110=float(round(1.0 / (G1 - s), 4)),
        m_lh_110=float(round(1.0 / (G1 + s), 4)),
        gamma8_110_closed_form_rel_dev=dev4,
        note="x', y', z' and 5 random directions; closed form "
             "1/(g1 -/+ sqrt(g2^2+3 g3^2))",
        passed=bool(pb))
    ok &= pb

    # (c) isotropic limit g2 = g3: rotated == unrotated fin extraction
    gi = (G1, 1.0, 1.0)
    r_rot = extract6_rot(PROD["Ez"], R=R110, g=gi)
    r_id = extract6_rot(PROD["Ez"], R=RID, g=gi)
    diffs = {n: round(_pct(r_rot[i], r_id[i]), 6)
             for i, n in enumerate(NAMES)}
    pc = max(diffs.values()) < 0.5
    out["c_isotropic_limit"] = dict(
        rotated={n: round(r_rot[i], 5) for i, n in enumerate(NAMES)},
        unrotated={n: round(r_id[i], 5) for i, n in enumerate(NAMES)},
        pct_diff=diffs, passed=bool(pc))
    ok &= pc
    print(f"  A/c done t={time.time()-t0:.0f}s", flush=True)

    # (c2) R = identity reproduces kp6_holes.extract6 (production basis)
    mine = extract6_rot(PROD["Ez"], R=RID)
    ref = kp6_holes.extract6(PROD["Ez"])
    d2 = {n: round(_pct(mine[i], ref[i]), 9) for i, n in enumerate(NAMES)}
    pc2 = max(d2.values()) < 1e-6
    out["c2_identity_vs_kp6"] = dict(
        rot_id={n: round(mine[i], 6) for i, n in enumerate(NAMES)},
        kp6={n: round(ref[i], 6) for i, n in enumerate(NAMES)},
        pct_diff=d2, passed=bool(pc2))
    ok &= pc2
    print(f"  A/c2 done t={time.time()-t0:.0f}s", flush=True)

    # (d) Kramers at B = 0 + hermiticity with all fields on
    worst = 0.0
    for Ez in (0.0, PROD["Ez"], 2.5e7):
        ev = np.linalg.eigvalsh(build_fin_H6_rot(0.0, Ez))
        for i in range(30):
            split = ev[2 * i + 1] - ev[2 * i]
            spacing = ev[2 * i + 2] - ev[2 * i]
            if spacing > 0:
                worst = max(worst, split / spacing)
    H = build_fin_H6_rot(2e7, 3e7, (0.05, 0.05, 0.05))
    herm = float(np.abs(H - H.conj().T).max())
    pd = worst < 1e-10 and herm < 1e-25
    out["d_kramers_herm"] = dict(max_split_over_spacing=float(worst),
                                 max_nonhermiticity_J=herm,
                                 passed=bool(pd))
    ok &= pd

    # (e) basis-size convergence at the production point
    base = extract6_rot(PROD["Ez"])
    fine = extract6_rot(PROD["Ez"], ny=PROD["ny"] + 4, nz=PROD["nz"] + 4)
    conv = {n: round(_pct(base[i], fine[i]), 3) for i, n in enumerate(NAMES)}
    pe = max(conv.values()) < 3.0
    out["e_convergence"] = dict(
        prod_11x13={n: round(base[i], 4) for i, n in enumerate(NAMES)},
        change_pct_11x13_vs_15x17_modes=conv, passed=bool(pe))
    ok &= pe

    out["all_passed"] = bool(ok)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def sec_B():
    """[110]-channel extract6 vs Ez at the production cross-section —
    the direct counterpart of kp6_holes section B."""
    t0 = time.time()
    rows = {}
    for EzM in (2, 5, 10, 15, 20, 30, 40, 50):
        rows[f"Ez{EzM}"] = _row(extract6_rot(EzM * 1e6))
        print(f"  B: Ez={EzM} MV/m done t={time.time()-t0:.0f}s", flush=True)
    gx = [r["gx"] for r in rows.values()]
    al = [r["alpha_eVA"] for r in rows.values()]
    kp6B = _kp6().get("B", {}).get("rows", {})
    return dict(rows=rows,
                gx_range=[round(min(gx), 3), round(max(gx), 3)],
                alpha_range_eVA=[round(min(al), 3), round(max(al), 3)],
                gx_100_same_rows={k: v["gx"] for k, v in kp6B.items()},
                runtime_s=round(time.time() - t0, 1))


def sec_C():
    """Seven-geometry bracket, [110] channel (same GEOMS as kp6 section C,
    cross-checked against output/data/kp6.json)."""
    t0 = time.time()
    kp6C = _kp6().get("C", {}).get("rows", {})
    if kp6C and set(kp6C) != set(GEOMS):
        raise SystemExit("GEOMS mismatch vs output/data/kp6.json section C")
    rows = {}
    for name, (wy, wz, ezM) in GEOMS.items():
        rows[name] = _row(extract6_rot(ezM * 1e6, Wy=wy * 1e-9,
                                       Wz=wz * 1e-9, ny=wy + 1, nz=wz + 1))
        print(f"  C: {name} done t={time.time()-t0:.0f}s", flush=True)
    gx = [r["gx"] for r in rows.values()]
    side = {k: dict(gx_110=rows[k]["gx"], gx_100=kp6C.get(k, {}).get("gx"))
            for k in rows}
    return dict(rows=rows,
                gx_range=[round(min(gx), 3), round(max(gx), 3)],
                gx_side_by_side=side,
                fourband_100_gx_range=[0.4, 1.07],
                sixband_100_gx_range=_kp6().get("C", {}).get("gx_range",
                                                             [0.04, 0.93]),
                measured_Geyer_gxx=GEYER_GXX,
                runtime_s=round(time.time() - t0, 1))


def _rms(g, tgt):
    return float(np.sqrt(np.mean([(g[i] - tgt[i]) ** 2 for i in range(3)])))


def sec_D():
    """Benchmark comparison vs Geyer / Camenzind / Venitucci + gap closure.
    Pure post-processing of sections B, C and the kp6 [100] checkpoint."""
    t0 = time.time()
    ck = _load()
    if "B" not in ck or "C" not in ck:
        raise SystemExit("run sections B and C before D")
    rows = {f"B_{k}": v for k, v in ck["B"]["rows"].items()}
    rows.update({f"C_{k}": v for k, v in ck["C"]["rows"].items()})
    kp6 = _kp6()
    kp6B = kp6.get("B", {}).get("rows", {})
    gx100_prod = kp6B.get("Ez10", {}).get("gx", 0.621)
    gx100_all = [r["gx"] for r in kp6B.values()] or [0.621]
    gz100_all = [r["gz"] for r in kp6B.values()] or [4.149]

    # closest [110] row to each Geyer lab-frame tensor
    best = {}
    for q, tgt in GEYER.items():
        name, r = min(rows.items(),
                      key=lambda kv: _rms((kv[1]["gx"], kv[1]["gy"],
                                           kv[1]["gz"]), tgt))
        best[q] = dict(row=name, g_110=[r["gx"], r["gy"], r["gz"]],
                       geyer=list(tgt),
                       rms=round(_rms((r["gx"], r["gy"], r["gz"]), tgt), 3))

    prod = ck["B"]["rows"]["Ez10"]
    gxB = [r["gx"] for r in ck["B"]["rows"].values()]
    gyB = [r["gy"] for r in ck["B"]["rows"].values()]
    gzB = [r["gz"] for r in ck["B"]["rows"].values()]
    gxC = [r["gx"] for r in ck["C"]["rows"].values()]

    meas_mid = 0.5 * (GEYER_GXX[0] + GEYER_GXX[1])      # 2.085
    closure_prod = (prod["gx"] - gx100_prod) / (meas_mid - gx100_prod)
    closure_best = ((max(gxB + gxC) - max(gx100_all))
                    / (meas_mid - max(gx100_all)))

    out = dict(
        production_Ez10_g_110=dict(gx_wire=prod["gx"], gy_inplane=prod["gy"],
                                   gz_001=prod["gz"]),
        wire_axis=dict(gx110_prod=prod["gx"],
                       gx110_range_B=[round(min(gxB), 3), round(max(gxB), 3)],
                       gx110_range_C=[round(min(gxC), 3), round(max(gxC), 3)],
                       gx100_prod=gx100_prod,
                       gx100_range_B=[round(min(gx100_all), 3),
                                      round(max(gx100_all), 3)],
                       measured_Geyer_gxx=GEYER_GXX),
        inplane_perp=dict(gy110_prod=prod["gy"],
                          gy110_range_B=[round(min(gyB), 3),
                                         round(max(gyB), 3)],
                          measured_Camenzind=CAMENZIND_GY),
        gz_001=dict(gz110_prod=prod["gz"],
                    gz110_range_B=[round(min(gzB), 3), round(max(gzB), 3)],
                    gz100_range_B=[round(min(gz100_all), 3),
                                   round(max(gz100_all), 3)],
                    venitucci_peak=VENITUCCI_GZ,
                    measured_Geyer_gzz=[1.46, 1.50]),
        closest_to_geyer=best,
        gap_closure=dict(
            measured_mid_gxx=meas_mid,
            closure_fraction_production=round(float(closure_prod), 3),
            closure_fraction_best_over_bracket=round(float(closure_best), 3),
            note="(gx110 - gx100)/(g_meas - gx100), production point and "
                 "bracket maxima"),
        runtime_s=round(time.time() - t0, 1))
    return out


SECS = {"A": sec_A, "B": sec_B, "C": sec_C, "D": sec_D}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all", choices=list(SECS) + ["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    todo = list(SECS) if args.sec == "all" else [args.sec]
    ck = _load()
    for s in todo:
        if s in ck and not args.force:
            print(f"[{s}] cached, skipping (--force to redo)")
            continue
        if s != "A" and not ck.get("A", {}).get("all_passed"):
            print(f"[{s}] blocked: validation section A has not passed yet")
            continue
        print(f"[{s}] running...")
        ck[s] = SECS[s]()
        _save(ck)
        print(f"[{s}] " + json.dumps(ck[s], indent=1)[:2500])
    # ledger highlights
    led = {}
    if "A" in ck:
        led["validation_all_passed"] = ck["A"]["all_passed"]
        led["isotropic_rot_vs_unrot_pct"] = ck["A"]["c_isotropic_limit"][
            "pct_diff"]
    if "B" in ck:
        led["gx110_range_Ez_2_50"] = ck["B"]["gx_range"]
        led["alpha110_range_eVA"] = ck["B"]["alpha_range_eVA"]
        led["table_10x12_110"] = ck["B"]["rows"]
    if "C" in ck:
        led["bracket_gx_range_110"] = ck["C"]["gx_range"]
        led["bracket_gx_range_100_6band"] = ck["C"]["sixband_100_gx_range"]
        led["measured_Geyer_gxx"] = GEYER_GXX
    if "D" in ck:
        led["benchmark_comparison"] = ck["D"]
    if led:
        save_numbers("kp6_110", led)


if __name__ == "__main__":
    main()
