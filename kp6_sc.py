"""
kp6_sc.py — self-consistent 6-band k.p + Poisson: does REALISTIC TRI-GATE
ELECTROSTATICS change the extracted hole parameters relative to the uniform
vertical field used so far?

Couples the two validated modules
  * kp6_holes.py  — 6-band LKBP fin model, hard-wall sine (dst) production
                    basis, extract6(Ez) -> (m*, alpha_eVA, gx, gy, gz, nso);
  * poisson2d.py  — tri-gate finite-volume Poisson + SCF harness with the
                    density_fn(phi_fin, yf, zf) callback contract,
without modifying either file.  Motivation: drafts/kp6_benchmarks.md, item 4.2
— "realistic, inhomogeneous electrostatic confinement ... enhances HH-LH
mixing and raises in-plane g" (Venitucci 2018; Bosco-Hetenyi-Loss 2021) — one
of the leading candidates for the gx ~ 0.6 (uniform-field hard-wall model) vs
~ 2.0 (measured, [110] devices) gap.

DESIGN / CONVENTIONS
--------------------
* Arbitrary-potential Hamiltonian, WITHOUT touching kp6_holes: the production
  basis is the hard-wall sine (Galerkin) basis, so a scalar potential V(y,z)
  enters through its sine-basis matrix elements
     <m p| V |n q> = int dy dz  s_m(y) s_p(z) V(y,z) s_n(y) s_q(z),
  s_m(x) = sqrt(2/W) sin(m pi x / W), computed by Gauss-Legendre quadrature
  (nq=96 nodes per axis) of the bilinearly interpolated Poisson potential.
  The resulting (ny*nz)x(ny*nz) matrix is added to the diagonal of all six
  bands:  H = build_fin_H6(kx, Ez=0, B) + kron(Vmat, I6).  This is a pure
  wrapper around kp6_holes.build_fin_H6 (imported building block); for a
  uniform field V = e*Ez*(z - Wz/2) the projector must reproduce kp6_holes'
  analytic sine-basis position matrix QE*Ez*Zd to quadrature precision and
  the full extraction must reproduce extract6(Ez) — asserted in section A.
* density_fn (callback contract of poisson2d): given the TOTAL electrostatic
  potential phi on the fin subgrid, the hole potential energy U = +e*phi is
  projected onto the k.p sine basis, H(kx=0) is diagonalized, and the density
  is the doublet-averaged envelope of the lowest Kramers pair,
     n(y,z) = (1/2) sum_{k=0,1} sum_{s=1..6} |psi_k(y,z,s)|^2 ,
  i.e. zero-T single-subband occupancy (appropriate at the line densities
  <= 5e7/m used here, where the 1D subband spacing >> Hartree scale).  The
  doublet SUM is gauge-invariant under the arbitrary unitary within the
  Kramers-degenerate pair.  Grid choice: the sine-basis wavefunction is an
  analytic function of (y,z), so it is EVALUATED DIRECTLY on the (finer)
  Poisson fin subgrid — no inverse interpolation / upsampling error at all.
  Coordinate map: Poisson fin coords are y in [-Wy/2, Wy/2] (centered),
  z in [0, Wz] (bottom-referenced); k.p wall coords are y+Wy/2 and z.
* Self-consistent extraction: poisson2d.FinPoisson.scf converges phi; the
  converged phi is then FROZEN, projected once to Vmat, and the full
  extraction (kx curvature at k1=2e7/m for m*, linear kx splitting for alpha,
  B = 0.05 T finite differences for gx, gy, gz, <J> of the kx-split doublet
  for nso) re-uses that frozen Vmat for every kx and B point.  Frozen
  electrostatics for the field/momentum derivatives is standard practice
  (the Hartree response at fixed total charge is second order in the probe).
  Recipes are line-for-line those of kp6_holes.extract6 so the comparison is
  exact.
* Uniform-field control (apples-to-apples): for each converged case the
  density-weighted effective field Ez_eff = <-dphi/dz>_n (poisson2d.
  effective_field with the converged density as weight) is computed, and
  kp6_holes.extract6(Ez=Ez_eff) is run at the same cross-section/basis.  The
  difference FULL minus UNIFORM isolates the inhomogeneous-electrostatics
  effect the literature points to.  (Sign note: extract6 applies U = +e*Ez*z
  which tilts holes to the BOTTOM, while the real gate pulls them to the TOP;
  the hard-wall rectangular fin is z-mirror symmetric, so the two are
  spectrally identical — verified explicitly in section A by extracting with
  the physically oriented uniform potential.)
* Section E lateral asymmetry: AsymTriGate subclasses FinPoisson in THIS
  file (poisson2d untouched): same mesh, same Dirichlet mask, same LU
  factorization; only the Dirichlet VALUES are overridden so the right
  sidewall (incl. the top-right corner node) sits at V_g + dV_right.  A crude
  linear probe U += e*Ey*y on the symmetric converged potential is also
  reported for comparison (labeled crude).

Checkpointing: per-section in output/data/kp6_sc.json (--force to redo);
ledger highlights via run_analysis.save_numbers("kp6_sc", ...).

Sections (python kp6_sc.py --sec A|B|C|D|E|all):
  A  validation: (a) pipeline check at n_l->0, V_g=-1.2 V (Ez_eff ~ 10 MV/m
     from the poisson2d B-calibration, bare tri-gate slope ~8.3e6 (V/m)/V):
     projector vs analytic uniform-field matrix; uniform-Ez_eff extraction
     through the new machinery vs kp6_holes.extract6 to < 1 %; z-mirror
     orientation check; (b) SCF convergence at V_g=-1.2 V, n_l=2e7/m
     (< 80 iters, residual < 1e-5 V, density nonnegative + unit-normalized);
     (c) k.p basis convergence: production 15x17 vs 19x21 < 3 % (gated);
     the 11x13 vs 15x17 drift (gx ~ 3.4 %, why production is 15x17 here
     rather than kp6_holes' 11x13) is reported alongside.
  B  V_g scan (tri-gate, n_l=2e7/m), Ez_eff ~ 3..30 MV/m: FULL vs UNIFORM
     (m*, alpha, gx, gy, gz) per row — the inhomogeneity effect.
  C  density/screening: n_l in {1e5 (~0), 2e7, 5e7}/m at V_g=-1.2 V.
  D  geometry: top-gate vs tri-gate at matched Ez_eff ~ 10 MV/m.
  E  lateral symmetry breaking: V_right = V_g + 0.2 V (subclass), plus the
     crude e*Ey*y probe (Ey = 1 MV/m) — does breaking y-symmetry raise gx?
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:                       # sandbox fallback, mirrors kp6_holes
    import scipy.sparse.linalg  # noqa: F401
except Exception:
    sys.path.insert(0, os.path.join(_HERE, "compat"))

import kp6_holes
from kp6_holes import (HBAR, M0, QE, MU_B_J, DSO_MEV, I6, J6, NAMES,
                       build_fin_H6, extract6, _dst_ops)
import poisson2d
from poisson2d import (FinPoisson, bilinear_interp, effective_field,
                       center_field, profile_metrics)

DATA = os.path.join(_HERE, "output", "data")
os.makedirs(DATA, exist_ok=True)
CKPT = os.path.join(DATA, "kp6_sc.json")

# production cross-section (identical to kp6_holes PROD) and gate stack.
# Basis note: kp6_holes' uniform-field production basis is 11x13 sine modes;
# the CORNER-LOCALIZED tri-gate potential is harder on the basis (gx moves
# 3.4 % between 11x13 and 15x17 at V_g=-1.2 V, n_l=2e7/m), so production
# here is 15x17, gated < 3 % against 19x21 in section A/c (both drifts are
# reported there).
WY, WZ = 10e-9, 12e-9
NY, NZ = 15, 17
VG0 = -1.2           # V_g giving Ez_eff ~ 10 MV/m (poisson2d B-calibration)
NL0 = 2e7            # production line density (holes/m)
K1, BTEST = 2e7, 0.05


# --------------------------------------------------- lowest-doublet eigensolver
try:
    from scipy.linalg import eigh as _seigh

    def _low2(H, vecs=False):
        """Lowest Kramers doublet of dense Hermitian H (hole picture:
        lowest eigenvalues are the topmost valence states)."""
        if vecs:
            return _seigh(H, subset_by_index=(0, 1))
        return _seigh(H, subset_by_index=(0, 1), eigvals_only=True)
except Exception:                                    # pragma: no cover
    def _low2(H, vecs=False):
        if vecs:
            w, v = np.linalg.eigh(H)
            return w[:2], v[:, :2]
        return np.linalg.eigvalsh(H)[:2]


# ------------------------------------------------------- ledger (run_analysis)
def save_numbers(tag, d):
    try:
        import run_analysis
        run_analysis.save_numbers(tag, d)
        return
    except Exception:
        pass
    path = os.path.join(DATA, "key_numbers.json")
    allnum = {}
    if os.path.exists(path):
        with open(path) as f:
            allnum = json.load(f)
    allnum[tag] = {**allnum.get(tag, {}), **d}
    with open(path, "w") as f:
        json.dump(allnum, f, indent=2)
    print(f"[{tag}] " + json.dumps(d))


# --------------------------------------------- potential -> sine-basis matrix
def _gauss(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _sine_modes(nmodes, W, x):
    """s_m(x) = sqrt(2/W) sin(m pi x / W), x in wall coords [0, W];
    rows m = 1..nmodes."""
    m = np.arange(1, nmodes + 1)[:, None]
    return np.sqrt(2.0 / W) * np.sin(m * np.pi * np.asarray(x)[None, :] / W)


class SineProjector:
    """Projects a scalar potential energy U(y,z) (Joules) onto the hard-wall
    sine basis of kp6_holes (ny x nz modes on Wy x Wz), index convention
    g = (m-1)*nz + (p-1) matching kron(y-op, z-op) in build_fin_H6."""

    def __init__(self, Wy=WY, Wz=WZ, ny=NY, nz=NZ, nq=96):
        self.Wy, self.Wz, self.ny, self.nz, self.nq = Wy, Wz, ny, nz, nq
        self.yq, wy = _gauss(nq, 0.0, Wy)        # wall coords
        self.zq, wz = _gauss(nq, 0.0, Wz)
        Sy = _sine_modes(ny, Wy, self.yq)
        Sz = _sine_modes(nz, Wz, self.zq)
        self.Ay = np.einsum("mi,ni,i->mni", Sy, Sy, wy)
        self.Az = np.einsum("pj,qj,j->pqj", Sz, Sz, wz)

    def vmat_from_U(self, Uq):
        """Uq: potential energy (J) sampled on (yq, zq), shape (nq, nq)."""
        M = np.einsum("mni,pqj,ij->mpnq", self.Ay, self.Az, Uq,
                      optimize=True)
        n = self.ny * self.nz
        return np.ascontiguousarray(M.reshape(n, n))

    def vmat_from_phi_fin(self, phi_fin, yf, zf):
        """Hole potential energy U = +e*phi from the Poisson fin subgrid
        (yf centered, zf bottom-referenced) bilinearly interpolated onto the
        quadrature nodes, then projected."""
        Uq = QE * bilinear_interp(yf, zf, phi_fin,
                                  self.yq - self.Wy / 2, self.zq)
        return self.vmat_from_U(Uq)

    def vmat_uniform_Ez(self, Ez_Vm, sign=+1):
        """Uniform vertical field through the SAME quadrature pipeline:
        U = sign * e*Ez*(z - Wz/2).  sign=+1 matches kp6_holes' +e*Ez*z
        (holes to bottom); sign=-1 is the physical gate orientation."""
        u = sign * QE * Ez_Vm * (self.zq - self.Wz / 2)
        return self.vmat_from_U(np.broadcast_to(u[None, :],
                                                (self.nq, self.nq)).copy())

    def vmat_uniform_Ey(self, Ey_Vm):
        """Crude lateral probe U = +e*Ey*(y - Wy/2) (sign immaterial for a
        y-symmetric base potential)."""
        u = QE * Ey_Vm * (self.yq - self.Wy / 2)
        return self.vmat_from_U(np.broadcast_to(u[:, None],
                                                (self.nq, self.nq)).copy())


# ------------------------------------------------------------ quantum density
def _doublet_density(v2, ny, nz, Wy, Wz, yf, zf):
    """Doublet-averaged 6-spinor envelope density on the Poisson fin subgrid.
    v2: (6*ny*nz, 2) lowest-doublet eigenvectors in the sine basis; the
    sine-basis wavefunction is analytic in (y,z) so it is evaluated exactly
    on the finer Poisson grid (no upsampling interpolation)."""
    c = v2.reshape(ny, nz, 6, 2)                   # (m, p, spinor, doublet)
    Syf = _sine_modes(ny, Wy, np.asarray(yf) + Wy / 2)
    Szf = _sine_modes(nz, Wz, zf)
    psi = np.einsum("mpsk,mi,pj->ijsk", c, Syf, Szf, optimize=True)
    return 0.5 * (np.abs(psi) ** 2).sum(axis=(2, 3))


def make_kp_density(dev, ny=NY, nz=NZ, Dso_meV=DSO_MEV, nq=96):
    """density_fn for poisson2d.scf (callback contract: nonnegative 2D
    density on the fin subgrid; harness renormalizes).  Also returns a state
    dict (last Vmat, last doublet energies) and the projector."""
    proj = SineProjector(dev.Wy, dev.Wz, ny, nz, nq)
    H0 = build_fin_H6(0.0, 0.0, (0.0, 0.0, 0.0), Wy=dev.Wy, Wz=dev.Wz,
                      ny=ny, nz=nz, Dso_meV=Dso_meV, basis="dst")
    state = {}

    def density_fn(phi_fin, yf, zf):
        vmat = proj.vmat_from_phi_fin(phi_fin, yf, zf)
        w, v = _low2(H0 + np.kron(vmat, I6), vecs=True)
        state["vmat"], state["e2"] = vmat, w
        return _doublet_density(v, ny, nz, dev.Wy, dev.Wz, yf, zf)

    return density_fn, state, proj


# ----------------------------------------------- extraction with frozen V(y,z)
def extract_full(vmat, Wy=WY, Wz=WZ, ny=NY, nz=NZ, k1=K1, Btest=BTEST,
                 Dso_meV=DSO_MEV):
    """(m*/m0, alpha_eVA, gx, gy, gz, nso) for the lowest Kramers doublet
    with the arbitrary frozen potential matrix vmat (sine basis) added to all
    six bands.  Recipes line-for-line kp6_holes.extract6; the kx and small-B
    finite differences re-use the SAME frozen electrostatics (standard
    frozen-field derivative)."""
    V6 = np.kron(vmat, I6)
    kw = dict(Wy=Wy, Wz=Wz, ny=ny, nz=nz, Dso_meV=Dso_meV, basis="dst")

    def H(kx, B):
        return build_fin_H6(kx, 0.0, B, **kw) + V6

    e0 = _low2(H(0.0, (0, 0, 0)))
    ek, vk = _low2(H(k1, (0, 0, 0)), vecs=True)
    alpha_Jm = (ek[1] - ek[0]) / (2 * k1)
    dE = ek.mean() - e0.mean()
    mstar = HBAR ** 2 * k1 ** 2 / (2 * dE) / M0 if dE > 0 else np.nan
    gs = []
    for ax in range(3):
        Bv = [0.0, 0.0, 0.0]
        Bv[ax] = Btest
        eb = _low2(H(0.0, tuple(Bv)))
        gs.append((eb[1] - eb[0]) / (MU_B_J * Btest))
    Ng = vk.shape[0] // 6
    nso = []
    for Jm in J6:
        Op = np.kron(np.eye(Ng), Jm)
        nso.append(float(np.real(vk[:, 0].conj() @ Op @ vk[:, 0]
                                 - vk[:, 1].conj() @ Op @ vk[:, 1])))
    nso = np.array(nso)
    nrm = np.linalg.norm(nso)
    nso = nso / nrm if nrm > 1e-9 else nso
    return (float(mstar), float(alpha_Jm / QE / 1e-10),
            float(gs[0]), float(gs[1]), float(gs[2]), nso.tolist())


# ------------------------------------------------- asymmetric tri-gate (sec E)
class AsymTriGate(FinPoisson):
    """Tri-gate with independent sidewall voltages: V_left = V_g (and top
    gate = V_g), V_right = V_g + dV_right.  Subclass-only extension: the
    mesh, Dirichlet MASK and LU factorization of FinPoisson are reused
    untouched; only the Dirichlet VALUE assignment in solve() differs.  The
    top-right corner node lies on the right sidewall column and follows the
    right gate."""

    def __init__(self, dV_right=0.2, **kw):
        kw["gate"] = "tri"
        super().__init__(**kw)
        self.dV_right = dV_right
        m = np.zeros((self.core.Ny, self.core.Nz), bool)
        m[-1, :] = True
        self.right_mask = self.gate_mask & m

    def solve(self, V_g, rho=None):
        dv = np.zeros((self.core.Ny, self.core.Nz))
        dv[self.gate_mask] = V_g
        dv[self.right_mask] = V_g + self.dV_right
        return self.core.solve(dv, rho)


# --------------------------------------------------------------- case harness
def _pct(a, b):
    return 100.0 * abs(a - b) / max(abs(b), 1e-30)


def _resrow(res):
    m, al, gx, gy, gz, nso = res
    return dict(mstar=round(m, 4), alpha_eVA=round(al, 4), gx=round(gx, 4),
                gy=round(gy, 4), gz=round(gz, 4),
                nso=[round(x, 3) for x in nso])


def _cmp(full, uni):
    out = {}
    for i, n in enumerate(NAMES):
        out[n] = dict(full=round(full[i], 4), uniform=round(uni[i], 4),
                      delta=round(full[i] - uni[i], 4),
                      pct=round(100.0 * (full[i] - uni[i])
                                / max(abs(uni[i]), 1e-12), 1))
    return out


def run_case(V_g, n_l, gate="tri", ny=NY, nz=NZ, dev=None, nq=96,
             tol=1e-5, max_iter=150, label=""):
    """SCF at (V_g, n_l, gate) -> converged phi; frozen-phi full extraction;
    density-weighted Ez_eff; uniform-Ez_eff control via kp6_holes.extract6.
    Returns (record_dict, extras) — extras holds arrays (vmat, n2d, dev,
    proj) for follow-up extractions, not serialized."""
    t0 = time.time()
    if dev is None:
        dev = FinPoisson(Wy=WY, Wz=WZ, gate=gate)
    density_fn, state, proj = make_kp_density(dev, ny=ny, nz=nz, nq=nq)
    phi, nit, hist = dev.scf(density_fn, n_l=n_l, V_g=V_g,
                             tol=tol, max_iter=max_iter)
    pf = dev.phi_fin(phi)
    # one final consistent evaluation at the converged phi (also refreshes
    # state['vmat'] to the frozen converged potential)
    n2d = np.clip(density_fn(pf, dev.yf, dev.zf), 0.0, None)
    n2d = n2d / float((n2d * dev.wf).sum())
    Ez_eff, Ey_eff = effective_field(pf, dev.yf, dev.zf, weight=n2d)
    Ez_c, _ = center_field(pf, dev.yf, dev.zf)
    met = profile_metrics(pf, dev.yf, dev.zf)
    zbar = float((n2d * dev.wf * dev.zf[None, :]).sum())
    ybar = float((n2d * dev.wf * dev.yf[:, None]).sum())
    full = extract_full(state["vmat"], dev.Wy, dev.Wz, ny, nz)
    # uniform-Ez_eff control: same machinery with the uniform-field Vmat,
    # validated IDENTICAL to kp6_holes.extract6(Ez_eff) in section A/a
    # (< 1e-4 % there); uses the lowest-doublet solver, ~6x faster than
    # extract6's full-spectrum eigvalsh.
    uni = extract_full(proj.vmat_uniform_Ez(abs(Ez_eff), sign=+1),
                       dev.Wy, dev.Wz, ny, nz)
    rec = dict(label=label, V_g=V_g, n_l=n_l, gate=gate, ny=ny, nz=nz,
               n_iter=int(nit), res_final_V=float(hist[-1]),
               converged=bool(hist[-1] < tol),
               Ez_eff_MVm=round(Ez_eff / 1e6, 4),
               Ey_eff_MVm=round(Ey_eff / 1e6, 5),
               Ez_center_MVm=round(Ez_c / 1e6, 4),
               zbar_nm=round(zbar * 1e9, 3), ybar_nm=round(ybar * 1e9, 3),
               asym_y=round(met["asym_y"], 4), asym_z=round(met["asym_z"], 4),
               full=_resrow(full), uniform_Ezeff=_resrow(uni),
               diff_full_minus_uniform=_cmp(full, uni),
               runtime_s=round(time.time() - t0, 1))
    extras = dict(dev=dev, proj=proj, vmat=state["vmat"], n2d=n2d, phi=phi,
                  full=full, uni=uni, Ez_eff=Ez_eff)
    return rec, extras


def _brief(rec):
    f, u = rec["full"], rec["uniform_Ezeff"]
    return (f"Vg={rec['V_g']:+.2f} nl={rec['n_l']:.0e} {rec['gate']:>3}"
            f" it={rec['n_iter']:3d} Ez_eff={rec['Ez_eff_MVm']:6.2f} MV/m |"
            f" m* {f['mstar']:.3f}/{u['mstar']:.3f}"
            f" a {f['alpha_eVA']:.4f}/{u['alpha_eVA']:.4f}"
            f" gx {f['gx']:.3f}/{u['gx']:.3f}"
            f" gy {f['gy']:.3f}/{u['gy']:.3f}"
            f" gz {f['gz']:.3f}/{u['gz']:.3f}  (full/uniform)")


# ------------------------------------------------------------------- sections
def sec_A():
    """Validation a-c (see module docstring)."""
    t0 = time.time()
    out, ok = {}, True

    # ---- (a) pipeline check at n_l -> 0 (tiny 1e5/m), V_g = VG0
    rec, ex = run_case(VG0, 1e5, label="A_lowdensity")
    Ez = ex["Ez_eff"]
    print("  A/a: " + _brief(rec), flush=True)
    # projector vs kp6_holes' analytic sine-basis position matrix
    _, _, Xz = _dst_ops(NZ, WZ)
    Zd = np.kron(np.eye(NY), Xz)
    vm_u = ex["proj"].vmat_uniform_Ez(Ez, sign=+1)
    quad_err = float(np.abs(vm_u - QE * Ez * Zd).max()
                     / np.abs(QE * Ez * Zd).max())
    # uniform field through the new machinery vs extract6 (pipeline check)
    pipe = extract_full(vm_u)
    ref = extract6(Ez, Wy=WY, Wz=WZ, ny=NY, nz=NZ)
    d_pipe = {n: round(_pct(pipe[i], ref[i]), 4) for i, n in enumerate(NAMES)}
    # physically oriented uniform field (holes to top): z-mirror check
    mirror = extract_full(ex["proj"].vmat_uniform_Ez(Ez, sign=-1))
    d_mir = {n: round(_pct(mirror[i], ref[i]), 4)
             for i, n in enumerate(NAMES)}
    ez_ok = 6e6 < Ez < 1.5e7
    pa = (quad_err < 1e-8 and max(d_pipe.values()) < 1.0
          and max(d_mir.values()) < 1.0 and ez_ok)
    out["a_pipeline"] = dict(
        case=rec, Ez_eff_MVm=round(Ez / 1e6, 4),
        Ez_eff_near_10MVm=bool(ez_ok),
        bare_slope_note="poisson2d B-calibration: ~8.3e6 (V/m)/V tri-gate "
                        "(uniform weight); density-weighted Ez_eff differs "
                        "since the kp hole blob sits near the top gate",
        projector_vs_analytic_relerr=quad_err,
        uniform_pipeline_vs_extract6_pct=d_pipe,
        mirror_orientation_vs_extract6_pct=d_mir,
        passed=bool(pa))
    ok &= pa
    print(f"  A/a: quad_err={quad_err:.2e}, pipe pct={d_pipe}, "
          f"mirror pct={d_mir} -> {'PASS' if pa else 'FAIL'}", flush=True)

    # ---- (b) SCF convergence at production parameters
    rec_b, ex_b = run_case(VG0, NL0, label="A_production")
    print("  A/b: " + _brief(rec_b), flush=True)
    n2d = ex_b["n2d"]
    dev = ex_b["dev"]
    norm = float((n2d * dev.wf).sum())
    pb = (rec_b["converged"] and rec_b["n_iter"] < 80
          and rec_b["res_final_V"] < 1e-5
          and float(n2d.min()) >= 0.0 and abs(norm - 1.0) < 1e-9)
    out["b_scf_production"] = dict(
        case=rec_b, density_min=float(n2d.min()),
        density_norm=norm, passed=bool(pb))
    ok &= pb
    print(f"  A/b: it={rec_b['n_iter']} res={rec_b['res_final_V']:.2e} "
          f"norm={norm:.12f} -> {'PASS' if pb else 'FAIL'}", flush=True)

    # ---- (c) k.p basis-size sanity: full SCF + extraction at 11x13 (the
    # kp6_holes uniform-field production basis) and 19x21; the production
    # basis here (15x17, = rec_b) must agree with 19x21 to < 3 %.  The
    # 11x13 -> 15x17 drift is reported too (gx ~ 3.4 %: the corner-localized
    # tri-gate potential is harder on the sine basis than a uniform field,
    # which is why production was bumped to 15x17).
    rec_c0, _ = run_case(VG0, NL0, ny=11, nz=13, label="A_coarse")
    print("  A/c: " + _brief(rec_c0), flush=True)
    rec_c, _ = run_case(VG0, NL0, ny=NY + 4, nz=NZ + 4, label="A_fine")
    print("  A/c: " + _brief(rec_c), flush=True)
    f0, fc, ff = rec_c0["full"], rec_b["full"], rec_c["full"]
    d_coarse = {n: round(_pct(fc[n], f0[n]), 3) for n in NAMES}
    d_grid = {n: round(_pct(ff[n], fc[n]), 3) for n in NAMES}
    pcv = max(d_grid.values()) < 3.0
    out["c_basis_convergence"] = dict(
        coarse_11x13=f0, prod_15x17=fc, fine_19x21=ff,
        change_pct_11x13_to_15x17=d_coarse,
        change_pct_15x17_to_19x21=d_grid,
        passed=bool(pcv))
    ok &= pcv
    print(f"  A/c: 11x13->15x17 pct={d_coarse}; 15x17->19x21 pct={d_grid} "
          f"-> {'PASS' if pcv else 'FAIL'}", flush=True)

    out["all_passed"] = bool(ok)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def sec_B():
    """V_g scan, tri-gate, n_l = 2e7/m: FULL vs UNIFORM-Ez_eff per row."""
    t0 = time.time()
    rows = {}
    for vg in (-0.35, -0.8, -1.2, -2.0, -3.0, -4.3):
        rec, _ = run_case(vg, NL0, label=f"B_Vg{vg}")
        rows[f"Vg{vg}"] = rec
        print("  B: " + _brief(rec) + f"  t={time.time()-t0:.0f}s",
              flush=True)
    dgx = [r["diff_full_minus_uniform"]["gx"]["delta"] for r in rows.values()]
    dal = [r["diff_full_minus_uniform"]["alpha_eVA"]["delta"]
           for r in rows.values()]
    return dict(rows=rows,
                gx_delta_range=[round(min(dgx), 4), round(max(dgx), 4)],
                alpha_delta_range_eVA=[round(min(dal), 4),
                                       round(max(dal), 4)],
                runtime_s=round(time.time() - t0, 1))


def sec_C():
    """Line-density (screening) effect at V_g = VG0, tri-gate."""
    t0 = time.time()
    rows = {}
    for nl in (1e5, 2e7, 5e7):
        rec, _ = run_case(VG0, nl, label=f"C_nl{nl:.0e}")
        rows[f"nl{nl:.0e}"] = rec
        print("  C: " + _brief(rec) + f"  t={time.time()-t0:.0f}s",
              flush=True)
    return dict(rows=rows, runtime_s=round(time.time() - t0, 1))


def sec_D():
    """Top-gate vs tri-gate at matched Ez_eff ~ 10 MV/m, n_l = 2e7/m."""
    t0 = time.time()
    rec_tri, ex_tri = run_case(VG0, NL0, gate="tri", label="D_tri")
    print("  D: " + _brief(rec_tri), flush=True)
    target = abs(ex_tri["Ez_eff"])
    # two-step V_g calibration for the top gate (weaker capacitive coupling)
    vg1 = VG0
    rec1, ex1 = run_case(vg1, NL0, gate="top", label="D_top_cal")
    print("  D: " + _brief(rec1), flush=True)
    vg2 = vg1 * target / max(abs(ex1["Ez_eff"]), 1.0)
    rec_top, _ = run_case(vg2, NL0, gate="top", label="D_top")
    print("  D: " + _brief(rec_top) + f"  t={time.time()-t0:.0f}s",
          flush=True)
    match_pct = round(100 * abs(rec_top["Ez_eff_MVm"]
                                - rec_tri["Ez_eff_MVm"])
                      / rec_tri["Ez_eff_MVm"], 2)
    return dict(tri=rec_tri, top_calibration=rec1, top=rec_top,
                Ez_eff_match_pct=match_pct,
                runtime_s=round(time.time() - t0, 1))


def sec_E():
    """Lateral symmetry breaking: V_right = V_g + 0.2 V (subclass) and the
    crude e*Ey*y probe on the symmetric converged potential."""
    t0 = time.time()
    rec_sym, ex_sym = run_case(VG0, NL0, label="E_symmetric")
    print("  E: sym  " + _brief(rec_sym), flush=True)
    dev_a = AsymTriGate(dV_right=0.2, Wy=WY, Wz=WZ)
    rec_asym, _ = run_case(VG0, NL0, dev=dev_a, label="E_asym_dV0.2")
    print("  E: asym " + _brief(rec_asym), flush=True)
    # crude probe: linear lateral field added to the frozen symmetric vmat
    Ey_probe = 1e6
    crude = extract_full(ex_sym["vmat"]
                         + ex_sym["proj"].vmat_uniform_Ey(Ey_probe))
    gx_s = rec_sym["full"]["gx"]
    gx_a = rec_asym["full"]["gx"]
    gx_c = crude[2]
    print(f"  E: gx sym={gx_s:.4f} asym(dV=0.2V)={gx_a:.4f} "
          f"crude(Ey=1MV/m)={gx_c:.4f}  t={time.time()-t0:.0f}s", flush=True)
    return dict(
        symmetric=rec_sym, asym_dV_right_0p2V=rec_asym,
        crude_Ey_probe=dict(
            note="crude asymmetry probe: U += e*Ey*y with Ey = 1 MV/m added "
                 "to the FROZEN symmetric converged potential (no SCF "
                 "feedback) — labeled crude, subclass result is primary",
            Ey_MVm=Ey_probe / 1e6, **_resrow(crude)),
        gx_sym=gx_s, gx_asym=gx_a, gx_crude=gx_c,
        gx_raised_by_asymmetry=bool(gx_a > gx_s),
        runtime_s=round(time.time() - t0, 1))


SECS = {"A": sec_A, "B": sec_B, "C": sec_C, "D": sec_D, "E": sec_E}


# ----------------------------------------------------------------------- main
def _load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def _save(d):
    with open(CKPT, "w") as f:
        json.dump(d, f, indent=2)


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
        print(f"[{s}] running...", flush=True)
        ck[s] = SECS[s]()
        _save(ck)
        print(f"[{s}] done ({ck[s].get('runtime_s', '?')} s)", flush=True)
    # ledger highlights
    led = {}
    if "A" in ck:
        led["validation_all_passed"] = ck["A"]["all_passed"]
        led["pipeline_vs_extract6_pct"] = ck["A"]["a_pipeline"][
            "uniform_pipeline_vs_extract6_pct"]
        led["scf_iters_production"] = ck["A"]["b_scf_production"][
            "case"]["n_iter"]
        led["basis_conv_pct"] = ck["A"]["c_basis_convergence"][
            "change_pct_15x17_to_19x21"]
    if "B" in ck:
        led["B_table"] = {
            k: dict(Ez_eff_MVm=r["Ez_eff_MVm"],
                    full=r["full"], uniform=r["uniform_Ezeff"])
            for k, r in ck["B"]["rows"].items()}
        led["B_gx_delta_full_minus_uniform"] = ck["B"]["gx_delta_range"]
    if "C" in ck:
        led["C_gx_vs_nl"] = {k: r["full"]["gx"]
                             for k, r in ck["C"]["rows"].items()}
    if "D" in ck:
        led["D_gx_tri_vs_top"] = dict(
            tri=ck["D"]["tri"]["full"]["gx"],
            top=ck["D"]["top"]["full"]["gx"],
            Ez_match_pct=ck["D"]["Ez_eff_match_pct"])
    if "E" in ck:
        led["E_gx_sym_asym_crude"] = [ck["E"]["gx_sym"], ck["E"]["gx_asym"],
                                      ck["E"]["gx_crude"]]
        led["E_gx_raised_by_asymmetry"] = ck["E"]["gx_raised_by_asymmetry"]
    if led:
        save_numbers("kp6_sc", led)


if __name__ == "__main__":
    main()
