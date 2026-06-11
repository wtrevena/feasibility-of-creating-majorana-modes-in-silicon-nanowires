"""
poisson2d.py — electrostatics half of the 6-band k.p + Poisson upgrade.

2D finite-volume Poisson solver for a tri-gate Si-fin cross-section, plus a
Schrodinger-Poisson self-consistency (SCF) harness with a callback interface
so any quantum model (the 4/6-band k.p modules built in parallel) can supply
the hole density.  This module has NO dependence on the k.p code; it is
validated against analytic and toy densities (run `python poisson2d.py --sec A`).

GEOMETRY (y horizontal, z vertical; SI units, potential in volts)
    silicon fin :  |y| <= Wy/2,  0 <= z <= Wz            (eps_r = 11.7)
    gate oxide  :  shell of thickness t_ox on the two sidewalls and the top
                   (eps_r = 3.9, SiO2)
    tri-gate    :  Dirichlet phi = V_g on the outer oxide surface: the domain
                   boundary at y = +-(Wy/2 + t_ox) for z >= 0, and z = Wz + t_ox
                   (all y).  gate="top" keeps only the top gate (the sidewall
                   boundaries become zero-flux / Neumann).
    buried oxide:  -t_box <= z <= 0 (eps_r = 3.9) terminated at z = -t_box by a
                   grounded substrate (Dirichlet phi = 0).  The lateral boundary
                   of the BOX region (z < 0) is Neumann.

EQUATION   div( eps_r grad phi ) = -rho / eps0   in flux (finite-volume) form:
link permittivities are the harmonic mean of eps_r sampled along each link,
and the grid is snapped to every material interface, so piecewise-constant-eps
stacks are reproduced exactly (validation A.b).

SIGN CONVENTIONS   rho is charge density in C/m^3, holes positive.
E = -grad phi.  For a negative gate, phi decreases from the grounded substrate
toward the gate, so Ez_eff = <-dphi/dz> > 0, and holes (potential energy
U_h = +e*phi) accumulate toward the gate; their positive charge raises phi
there and screens (reduces) |E| in the fin (validation A.e).

PUBLIC API (what the integration / k.p agent should use)
    dev = FinPoisson(Wy=10e-9, Wz=12e-9, t_ox=3e-9, t_box=10e-9,
                     eps_si=11.7, eps_ox=3.9, h=0.25e-9, gate="tri")
    phi = dev.solve(V_g, rho=None)            # full-domain phi(y,z), shape (Ny,Nz)
    phi_f = dev.phi_fin(phi)                  # fin subgrid, coords dev.yf, dev.zf
    dev.phi_on_grid(phi, yq, zq)              # bilinear interp, arbitrary rect grid
    dev.phi_on_kp_grid(phi, ny=11, nz=13)     # lk_holes-convention interior grid
    phi, n_iter, hist = dev.scf(density_fn, n_l=2e7, V_g=-0.5, ...)
    Ez_eff, Ey_eff = effective_field(phi_f, yf, zf, weight=None)
    n2d = toy_density(phi_f, yf, zf)          # documented stand-in ONLY

CALLBACK CONTRACT   density_fn(phi_fin, yf, zf) -> n2d, a (len(yf), len(zf))
nonnegative array; the harness renormalizes it so that the 2D integral of n2d
over the fin cross-section equals 1, then multiplies by the line density n_l
(holes/m) and +e to obtain rho (C/m^3).  phi_fin is the TOTAL electrostatic
potential on the fin subgrid (gate + Hartree); the quantum model should add
hole potential energy U = +e*phi_fin to its Hamiltonian, recompute the lowest
subbands, and return the resulting normalized charge density.

NUMERICAL-STABILITY CAVEATS
  * SCF uses adaptive linear (Kerker-free) potential mixing: beta starts at
    0.3 and halves whenever the residual grows (floor 0.01).  At the line
    densities of interest (<= 5e7 /m) the Hartree term is a few-mV
    perturbation and converges in ~15-40 iterations; much larger n_l or a
    density_fn with strong, discontinuous response (level crossings) may need
    a smaller starting beta.
  * Interface permittivity: harmonic mean over 3 samples per link.  Grids are
    snapped to interfaces, so each link is single-material and layered
    analytic limits are exact; if you pass a custom eps_fn whose interfaces do
    not coincide with grid nodes, accuracy degrades to O(h) at the interface.
  * Boundary choices: the sidewall boundary below z=0 and (for gate="top")
    the full sidewalls are Neumann — an idealization of remote grounds.
    Ez_eff at fixed V_g shifts by a few percent if t_box is changed; the
    grounded substrate at z=-t_box is part of the capacitive divider.
  * Convergence criterion is max|phi_new - phi_old| < 1e-5 V on the FULL grid
    (pre-mixing residual), i.e. ~1% of the Hartree scale at n_l=2e7/m.
  * Screening vs image attraction: the uniform fin-average and fin-center
    fields are always REDUCED by the hole charge (true screening), but the
    DENSITY-WEIGHTED Ez_eff can be slightly enhanced (a few %) at weak gating
    because the hole blob is attracted to its negative image in the gate.
    When comparing screened/unscreened drives, say which average you mean.
"""

import argparse
import json
import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# ----------------------------------------------------------------- constants
EPS0 = 8.8541878128e-12        # F/m
QE = 1.602176634e-19           # C
HBAR = 1.054571817e-34         # J s
M0 = 9.1093837015e-31          # kg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output", "data")


# ------------------------------------------------------------- ledger output
def _save_numbers(tag, d):
    """Ledger via run_analysis.save_numbers if importable (it pulls in
    matplotlib); otherwise an exact-format local fallback."""
    try:
        from run_analysis import save_numbers
        save_numbers(tag, d)
        return
    except Exception:
        pass
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "key_numbers.json")
    allnum = {}
    if os.path.exists(path):
        with open(path) as f:
            allnum = json.load(f)
    allnum[tag] = {**allnum.get(tag, {}), **d}
    with open(path, "w") as f:
        json.dump(allnum, f, indent=2)
    print(f"[{tag}] " + json.dumps(d))


def _checkpoint(section, payload):
    """Merge per-section results into output/data/poisson2d.json."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "poisson2d.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
    blob[section] = payload
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[checkpoint] section {section} -> {path}")


# ------------------------------------------------------------------ core FVM
def _grid(segments, h):
    """1D grid covering contiguous segments [(a0,b0),(b0,b1),...], snapped to
    every segment endpoint, spacing ~h inside each segment."""
    pts = [float(segments[0][0])]
    for a, b in segments:
        n = max(1, int(round((b - a) / h)))
        pts.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.asarray(pts, float)


class Poisson2D:
    """Generic 2D variable-eps Poisson solver, -div(eps_r grad phi) = rho/eps0.

    Finite-volume 5-point stencil on a rectilinear (possibly nonuniform) grid.
    eps_r lives on links (flux form): harmonic mean of eps_fn sampled at the
    1/4, 1/2, 3/4 points of each link.  Nodes flagged in dir_mask are
    Dirichlet; every other boundary node is zero-flux (Neumann).  The free-node
    matrix is LU-factorized once, so repeated solves (SCF) are cheap.
    """

    def __init__(self, y, z, eps_fn, dir_mask):
        y = np.asarray(y, float)
        z = np.asarray(z, float)
        self.y, self.z = y, z
        Ny, Nz = y.size, z.size
        self.Ny, self.Nz = Ny, Nz
        dY, dZ = np.diff(y), np.diff(z)
        cy = np.empty(Ny)
        cy[0], cy[-1] = dY[0] / 2, dY[-1] / 2
        cy[1:-1] = 0.5 * (dY[:-1] + dY[1:])
        cz = np.empty(Nz)
        cz[0], cz[-1] = dZ[0] / 2, dZ[-1] / 2
        cz[1:-1] = 0.5 * (dZ[:-1] + dZ[1:])
        self.cy, self.cz = cy, cz
        self.area = np.outer(cy, cz)            # control volumes (per meter in x)

        fr = (0.25, 0.5, 0.75)                  # harmonic-mean sample fractions
        inv = np.zeros((Ny - 1, Nz))
        for f in fr:
            YY, ZZ = np.meshgrid(y[:-1] + f * dY, z, indexing="ij")
            inv += 1.0 / eps_fn(YY, ZZ)
        epsY = len(fr) / inv                    # eps on y-links
        inv = np.zeros((Ny, Nz - 1))
        for f in fr:
            YY, ZZ = np.meshgrid(y, z[:-1] + f * dZ, indexing="ij")
            inv += 1.0 / eps_fn(YY, ZZ)
        epsZ = len(fr) / inv                    # eps on z-links

        TY = epsY * cz[None, :] / dY[:, None]   # link conductances
        TZ = epsZ * cy[:, None] / dZ[None, :]

        N = Ny * Nz
        Iy, Jy = np.meshgrid(np.arange(Ny - 1), np.arange(Nz), indexing="ij")
        a = (Iy * Nz + Jy).ravel()
        b = ((Iy + 1) * Nz + Jy).ravel()
        t = TY.ravel()
        Iz, Jz = np.meshgrid(np.arange(Ny), np.arange(Nz - 1), indexing="ij")
        a2 = (Iz * Nz + Jz).ravel()
        b2 = a2 + 1
        t2 = TZ.ravel()
        rows = np.concatenate([a, b, a, b, a2, b2, a2, b2])
        cols = np.concatenate([a, b, b, a, a2, b2, b2, a2])
        vals = np.concatenate([t, t, -t, -t, t2, t2, -t2, -t2])
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

        dm = np.asarray(dir_mask, bool)
        self.dir_mask = dm
        self.free = np.flatnonzero(~dm.ravel())
        self.fixed = np.flatnonzero(dm.ravel())
        Aff = A[self.free, :].tocsc()[:, self.free]
        self.Afd = A[self.free, :].tocsc()[:, self.fixed].tocsr()
        self.lu = splu(Aff.tocsc())

    def solve(self, dir_values, rho=None):
        """phi(y,z) with Dirichlet values dir_values (full-grid array; only
        dir_mask entries used) and charge density rho in C/m^3 (or None)."""
        dv = np.asarray(dir_values, float).ravel()
        b = np.zeros(self.Ny * self.Nz)
        if rho is not None:
            b = (np.asarray(rho, float) / EPS0 * self.area).ravel()
        bf = b[self.free] - self.Afd @ dv[self.fixed]
        phi = np.empty(self.Ny * self.Nz)
        phi[self.fixed] = dv[self.fixed]
        phi[self.free] = self.lu.solve(bf)
        return phi.reshape(self.Ny, self.Nz)


def bilinear_interp(yg, zg, F, yq, zq):
    """Bilinear interpolation of F (on rectilinear grid yg x zg) onto the
    tensor grid yq x zq.  Returns array of shape (len(yq), len(zq))."""
    yq = np.atleast_1d(np.asarray(yq, float))
    zq = np.atleast_1d(np.asarray(zq, float))
    iy = np.clip(np.searchsorted(yg, yq) - 1, 0, len(yg) - 2)
    iz = np.clip(np.searchsorted(zg, zq) - 1, 0, len(zg) - 2)
    ty = np.clip((yq - yg[iy]) / (yg[iy + 1] - yg[iy]), 0.0, 1.0)
    tz = np.clip((zq - zg[iz]) / (zg[iz + 1] - zg[iz]), 0.0, 1.0)
    F00 = F[np.ix_(iy, iz)]
    F10 = F[np.ix_(iy + 1, iz)]
    F01 = F[np.ix_(iy, iz + 1)]
    F11 = F[np.ix_(iy + 1, iz + 1)]
    wy0, wy1 = (1 - ty)[:, None], ty[:, None]
    wz0, wz1 = (1 - tz)[None, :], tz[None, :]
    return F00 * wy0 * wz0 + F10 * wy1 * wz0 + F01 * wy0 * wz1 + F11 * wy1 * wz1


# -------------------------------------------------------------- fin geometry
class FinPoisson:
    """Tri-gate Si-fin Poisson solver + SCF harness (see module docstring)."""

    def __init__(self, Wy=10e-9, Wz=12e-9, t_ox=3e-9, t_box=10e-9,
                 eps_si=11.7, eps_ox=3.9, h=0.25e-9, gate="tri"):
        assert gate in ("tri", "top")
        self.Wy, self.Wz, self.t_ox, self.t_box = Wy, Wz, t_ox, t_box
        self.gate = gate
        y = _grid([(-Wy / 2 - t_ox, -Wy / 2), (-Wy / 2, Wy / 2),
                   (Wy / 2, Wy / 2 + t_ox)], h)
        z = _grid([(-t_box, 0.0), (0.0, Wz), (Wz, Wz + t_ox)], h)

        def eps_fn(Y, Z):
            si = (np.abs(Y) <= Wy / 2) & (Z >= 0.0) & (Z <= Wz)
            return np.where(si, eps_si, eps_ox)

        Ny, Nz = y.size, z.size
        gate_mask = np.zeros((Ny, Nz), bool)
        gate_mask[:, -1] = True                          # top gate
        if gate == "tri":
            zi = z >= -1e-15                             # sidewall gates, z >= 0
            gate_mask[0, zi] = True
            gate_mask[-1, zi] = True
        sub_mask = np.zeros((Ny, Nz), bool)
        sub_mask[:, 0] = True                            # grounded substrate
        gate_mask &= ~sub_mask
        self.gate_mask, self.sub_mask = gate_mask, sub_mask
        self.core = Poisson2D(y, z, eps_fn, gate_mask | sub_mask)

        tol = 1e-15
        self.iyf = np.flatnonzero((y >= -Wy / 2 - tol) & (y <= Wy / 2 + tol))
        self.izf = np.flatnonzero((z >= -tol) & (z <= Wz + tol))
        self.yf, self.zf = y[self.iyf], z[self.izf]
        # control-volume weights of the fin nodes (used for all fin integrals,
        # so the deposited line charge is exactly n_l * e)
        self.wf = np.outer(self.core.cy[self.iyf], self.core.cz[self.izf])
        self._last_n2d = None

    # ----------------------------------------------------------- electrostatics
    def solve(self, V_g, rho=None):
        """Solve for phi(y,z) (volts) on the full grid at gate voltage V_g.
        rho: full-grid charge density in C/m^3, or None."""
        dv = np.zeros((self.core.Ny, self.core.Nz))
        dv[self.gate_mask] = V_g
        return self.core.solve(dv, rho)

    def phi_fin(self, phi):
        """Restrict a full-grid phi to the fin subgrid (self.yf, self.zf),
        boundary (Si/SiO2 interface) nodes included."""
        return phi[np.ix_(self.iyf, self.izf)]

    def phi_on_grid(self, phi, yq, zq):
        """Bilinear interpolation of full-grid phi onto an arbitrary
        rectilinear grid yq x zq (e.g. a k.p grid).  Coordinates: y centered
        on the fin, z measured from the fin bottom (fin spans 0..Wz)."""
        return bilinear_interp(self.core.y, self.core.z, phi, yq, zq)

    def kp_grid(self, ny=11, nz=13):
        """Hard-wall interior grid in lk_holes convention (both coordinates
        centered): y_i=(i+1)dy-Wy/2, z_j=(j+1)dz-Wz/2."""
        dy, dz = self.Wy / (ny + 1), self.Wz / (nz + 1)
        return ((np.arange(ny) + 1) * dy - self.Wy / 2,
                (np.arange(nz) + 1) * dz - self.Wz / 2)

    def phi_on_kp_grid(self, phi, ny=11, nz=13):
        """phi interpolated onto the lk_holes ny x nz interior grid (centered
        z convention is shifted internally: z_solver = z_kp + Wz/2)."""
        ykp, zkp = self.kp_grid(ny, nz)
        return self.phi_on_grid(phi, ykp, zkp + self.Wz / 2)

    def rho_from_density(self, n2d, n_l):
        """Full-grid rho (C/m^3) from a fin-subgrid density: clip, renormalize
        to unit cross-section integral, scale by +e * n_l."""
        n = np.clip(np.asarray(n2d, float), 0.0, None)
        s = float((n * self.wf).sum())
        if s <= 0:
            raise ValueError("density_fn returned a non-positive density")
        n = n / s
        rho = np.zeros((self.core.Ny, self.core.Nz))
        rho[np.ix_(self.iyf, self.izf)] = QE * n_l * n
        return rho, n

    # -------------------------------------------------------------------- SCF
    def scf(self, density_fn, n_l=2e7, V_g=-0.5, mix0=0.3, tol=1e-5,
            max_iter=200, verbose=False):
        """Schrodinger-Poisson self-consistency loop.

        density_fn(phi_fin, yf, zf) -> n2d on the fin subgrid (renormalized
        internally); n_l holes/m sets the total charge.  Adaptive linear
        mixing: beta=mix0, halved whenever the residual increases (floor 0.01).
        Converges on max|phi_new - phi_old| < tol (volts) over the full grid.

        Returns (phi, n_iter, residual_history).  The converged density is
        stashed in self._last_n2d (normalized, fin subgrid).
        """
        phi = self.solve(V_g, None)
        beta, res_prev = mix0, np.inf
        hist = []
        for it in range(1, max_iter + 1):
            n2d = density_fn(self.phi_fin(phi), self.yf, self.zf)
            rho, n_norm = self.rho_from_density(n2d, n_l)
            self._last_n2d = n_norm
            phi_new = self.solve(V_g, rho)
            res = float(np.abs(phi_new - phi).max())
            hist.append(res)
            if verbose:
                print(f"  scf it {it:3d}  res {res:.3e}  beta {beta:.3f}")
            if res < tol:
                return phi_new, it, hist
            if res > res_prev:
                beta = max(beta / 2, 0.01)
            phi = phi + beta * (phi_new - phi)
            res_prev = res
        return phi, max_iter, hist


def scf(density_fn, n_l=2e7, V_g=-0.5, device=None, **kwargs):
    """Module-level convenience wrapper: build a default FinPoisson (or use
    `device`) and run device.scf.  Returns (phi, n_iter, residual_history)."""
    if device is None:
        device = FinPoisson()
    return device.scf(density_fn, n_l=n_l, V_g=V_g, **kwargs)


# ------------------------------------------------------- fields & diagnostics
def _trap_w(x):
    w = np.zeros_like(x)
    w[:-1] += 0.5 * np.diff(x)
    w[1:] += 0.5 * np.diff(x)
    return w


def effective_field(phi_fin, yf, zf, weight=None):
    """Density-weighted average field in the fin: (Ez_eff, Ey_eff) in V/m,
    E = -grad phi, weighted by `weight` (any nonnegative fin-subgrid array;
    None = uniform).  This is the equivalent-uniform-field handle for
    apples-to-apples comparison with the old uniform-Ez treatment."""
    dpy, dpz = np.gradient(phi_fin, yf, zf)
    Ey, Ez = -dpy, -dpz
    w = np.ones_like(phi_fin) if weight is None else np.asarray(weight, float)
    w = w * np.outer(_trap_w(yf), _trap_w(zf))
    w = w / w.sum()
    return float((w * Ez).sum()), float((w * Ey).sum())


def center_field(phi_fin, yf, zf):
    """(Ez, Ey) at the fin center (y=0, z=middle) — the quantity the old
    uniform-Ez treatment effectively used."""
    dpy, dpz = np.gradient(phi_fin, yf, zf)
    zc = 0.5 * (zf[0] + zf[-1])
    Ez = float(bilinear_interp(yf, zf, -dpz, [0.0], [zc])[0, 0])
    Ey = float(bilinear_interp(yf, zf, -dpy, [0.0], [zc])[0, 0])
    return Ez, Ey


def profile_metrics(phi_fin, yf, zf):
    """Asymmetry metrics of phi over the fin.
      asym_z  = [phi(0,ztop)-phi(0,zbot)] / (max phi - min phi)  — fraction of
                the fin's potential variation that is a vertical (z) drop;
      asym_y  = [phi(ymax,zmid)-phi(ymin,zmid)] / (max-min)      — left/right
                imbalance (0 for the symmetric tri-gate);
      bowl_y, bowl_z = edge-average minus center curvature proxies (V):
                lateral vs vertical confinement strength."""
    pt = lambda yy, zz: float(bilinear_interp(yf, zf, phi_fin, [yy], [zz])[0, 0])
    zmid = 0.5 * (zf[0] + zf[-1])
    rng = float(phi_fin.max() - phi_fin.min())
    rng = rng if rng > 0 else 1.0
    dphi_z = pt(0.0, zf[-1]) - pt(0.0, zf[0])
    dphi_y = pt(yf[-1], zmid) - pt(yf[0], zmid)
    bowl_y = 0.5 * (pt(yf[0], zmid) + pt(yf[-1], zmid)) - pt(0.0, zmid)
    bowl_z = 0.5 * (pt(0.0, zf[0]) + pt(0.0, zf[-1])) - pt(0.0, zmid)
    return dict(asym_z=dphi_z / rng, asym_y=dphi_y / rng,
                dphi_z_V=dphi_z, bowl_y_V=bowl_y, bowl_z_V=bowl_z)


# ------------------------------------------------------------------ toy model
def make_toy_density(m_eff=0.28, E_smooth_V=0.005, sigma_min=0.5e-9,
                     sigma_frac_max=0.30):
    """TOY density factory — a STAND-IN for the k.p quantum density, used only
    for standalone SCF testing/calibration.  A Gaussian centered at the
    softmin (Boltzmann-weighted, scale E_smooth_V) of the hole potential
    energy U = +e*phi, with widths following a Thomas-Fermi/triangular-well
    response sigma = (hbar^2 / (2 m* e <|E|>))^(1/3) per axis, clipped to
    [sigma_min, sigma_frac_max * fin size].  It cannot represent corner
    lobes, hard-wall nodes, or subband structure — do not use for physics
    conclusions, only for harness validation."""
    mh = m_eff * M0

    def density_fn(phi_fin, yf, zf):
        W = np.outer(_trap_w(yf), _trap_w(zf))
        u = phi_fin - phi_fin.min()              # hole PE / e, volts
        w = np.exp(-u / E_smooth_V) * W
        w = w / w.sum()
        y0 = float((w * yf[:, None]).sum())
        z0 = float((w * zf[None, :]).sum())
        dpy, dpz = np.gradient(phi_fin, yf, zf)
        Ey_abs = float((w * np.abs(dpy)).sum())
        Ez_abs = float((w * np.abs(dpz)).sum())

        def sig(E, L):
            if E < 1e3:
                return sigma_frac_max * L
            s = (HBAR ** 2 / (2 * mh * QE * E)) ** (1.0 / 3.0)
            return min(max(s, sigma_min), sigma_frac_max * L)

        sy = sig(Ey_abs, yf[-1] - yf[0])
        sz = sig(Ez_abs, zf[-1] - zf[0])
        n = np.exp(-0.5 * ((yf[:, None] - y0) / sy) ** 2
                   - 0.5 * ((zf[None, :] - z0) / sz) ** 2)
        return n / (n * W).sum()

    return density_fn


toy_density = make_toy_density()


# ================================================================== sections
def _scf_case(dev, V_g, n_l, **kw):
    """Run SCF (toy density), return phi, metrics dict."""
    phi, nit, hist = dev.scf(toy_density, n_l=n_l, V_g=V_g, **kw)
    pf = dev.phi_fin(phi)
    n2d = dev._last_n2d
    Ez, Ey = effective_field(pf, dev.yf, dev.zf, weight=n2d)
    Ez_c, Ey_c = center_field(pf, dev.yf, dev.zf)
    m = profile_metrics(pf, dev.yf, dev.zf)
    zbar = float((n2d * dev.wf * dev.zf[None, :]).sum()
                 / (n2d * dev.wf).sum())
    return phi, dict(V_g=V_g, n_l=n_l, n_iter=nit,
                     res_final=hist[-1] if hist else 0.0,
                     Ez_eff=Ez, Ey_eff=Ey, Ez_center=Ez_c,
                     zbar_nm=zbar * 1e9, **m)


def section_A():
    print("=== Section A: validations ===")
    out = {}

    # --- (a) uniform-eps parallel plate, quasi-1D --------------------------
    d, V = 12e-9, 1.0
    y = np.linspace(0, 4e-9, 9)
    z = np.linspace(0, d, 49)
    dm = np.zeros((9, 49), bool)
    dm[:, 0] = dm[:, -1] = True
    p = Poisson2D(y, z, lambda Y, Z: np.full(Y.shape, 11.7), dm)
    dv = np.zeros((9, 49))
    dv[:, -1] = V
    phi = p.solve(dv)
    E = -np.diff(phi, axis=1) / np.diff(z)       # should be -V/d everywhere
    err_field = float(np.abs(E + V / d).max() / (V / d))
    err_phi = float(np.abs(phi - V * z[None, :] / d).max())
    out["a_plate"] = dict(err_field_rel=err_field, err_phi_V=err_phi,
                          passed=bool(err_field < 1e-3))
    print(f"(a) plate: field rel err {err_field:.2e}, phi err {err_phi:.2e} V"
          f"  -> {'PASS' if err_field < 1e-3 else 'FAIL'}")

    # --- (b) two-layer dielectric stack ------------------------------------
    eps_si, eps_ox = 11.7, 3.9
    d1 = 6e-9                                    # Si below, oxide above
    z = _grid([(0, d1), (d1, 12e-9)], 0.25e-9)
    y = np.linspace(0, 2e-9, 5)
    dm = np.zeros((y.size, z.size), bool)
    dm[:, 0] = dm[:, -1] = True
    p = Poisson2D(y, z, lambda Y, Z: np.where(Z < d1, eps_si, eps_ox), dm)
    dv = np.zeros((y.size, z.size))
    dv[:, -1] = 1.0
    phi = p.solve(dv)
    k_si = np.searchsorted(z, d1 / 2)            # cell deep in Si layer
    k_ox = np.searchsorted(z, d1 + 3e-9)         # cell deep in oxide layer
    E_si = (phi[0, k_si + 1] - phi[0, k_si]) / (z[k_si + 1] - z[k_si])
    E_ox = (phi[0, k_ox + 1] - phi[0, k_ox]) / (z[k_ox + 1] - z[k_ox])
    ratio = float(E_ox / E_si)
    err = abs(ratio - eps_si / eps_ox) / (eps_si / eps_ox)
    out["b_stack"] = dict(E_ratio=ratio, target=eps_si / eps_ox,
                          rel_err=err, passed=bool(err < 1e-3))
    print(f"(b) stack: E_ox/E_si = {ratio:.6f} (target {eps_si/eps_ox:.6f}),"
          f" rel err {err:.2e} -> {'PASS' if err < 1e-3 else 'FAIL'}")

    # --- (c) line charge vs 2D log Green's function ------------------------
    L, h = 60e-9, 0.5e-9
    n = int(round(L / h)) + 1
    y = np.linspace(-L / 2, L / 2, n)
    z = y.copy()
    dm = np.zeros((n, n), bool)
    dm[0, :] = dm[-1, :] = dm[:, 0] = dm[:, -1] = True
    p = Poisson2D(y, z, lambda Y, Z: np.ones(Y.shape), dm)
    lam = 1e-12                                  # C/m line charge
    ic = n // 2
    rho = np.zeros((n, n))
    rho[ic, ic] = lam / p.area[ic, ic]
    phi = p.solve(np.zeros((n, n)), rho)
    pref = lam / (2 * np.pi * EPS0 * 1.0)
    r0 = 3e-9
    k0 = int(round(r0 / h))
    rels = []
    for r in (4.5e-9, 6e-9, 9e-9):
        k = int(round(r / h))
        for dnum in (phi[ic + k, ic] - phi[ic + k0, ic],
                     phi[ic, ic + k] - phi[ic, ic + k0]):
            dana = -pref * np.log(r / r0)
            rels.append(abs(dnum - dana) / abs(dana))
    err_c = float(max(rels))
    out["c_line"] = dict(max_rel_err=err_c, passed=bool(err_c < 0.05))
    print(f"(c) line charge: max rel err vs log G(r) = {err_c:.3f}"
          f" (r=3..9 nm) -> {'PASS' if err_c < 0.05 else 'FAIL'}")

    # --- (d) grid convergence of the SCF phi -------------------------------
    dev_c = FinPoisson(h=0.25e-9)
    dev_f = FinPoisson(h=0.125e-9)
    phi_c, mc = _scf_case(dev_c, -0.5, 2e7)
    phi_f, mf = _scf_case(dev_f, -0.5, 2e7)
    pc = dev_c.phi_on_kp_grid(phi_c, 21, 25)
    pfin = dev_f.phi_on_kp_grid(phi_f, 21, 25)
    dmax = float(np.abs(pc - pfin).max())
    out["d_grid"] = dict(max_dphi_V=dmax, n_iter_coarse=mc["n_iter"],
                         n_iter_fine=mf["n_iter"], passed=bool(dmax < 1e-3))
    print(f"(d) grid conv: max|dphi| h->h/2 = {dmax*1e3:.3f} mV"
          f" -> {'PASS' if dmax < 1e-3 else 'FAIL'}")

    # --- (e) SCF robustness + Hartree sign / screening ----------------------
    # Screening factors compare the SCF phi against the n_l=0 phi at the same
    # V_g:  f_uniform  = uniform fin-average Ez ratio,
    #       f_center   = Ez ratio at the fin center,
    #       f_weighted = density-weighted Ez_eff ratio (same converged weights
    #                    for both fields).  f_uniform and f_center must be < 1
    #       (charge intercepts gate flux).  f_weighted may exceed 1 at weak
    #       gating: the hole blob is attracted to its image in the gate, a
    #       real electrostatic effect, NOT screening failure — reported as a
    #       caveat, not gated on.
    dev = FinPoisson()
    e_rows, all_ok = [], True
    for Vg in (-0.1, -0.3, -0.5, -1.0):
        phib = dev.solve(Vg)                     # bare (n_l = 0)
        phis, ms = _scf_case(dev, Vg, 2e7)       # screened
        n = dev._last_n2d
        pb, ps = dev.phi_fin(phib), dev.phi_fin(phis)
        Ezb_u, _ = effective_field(pb, dev.yf, dev.zf)
        Ezs_u, _ = effective_field(ps, dev.yf, dev.zf)
        Ezb_w, _ = effective_field(pb, dev.yf, dev.zf, weight=n)
        Ezs_w, _ = effective_field(ps, dev.yf, dev.zf, weight=n)
        Ezb_c, _ = center_field(pb, dev.yf, dev.zf)
        Ezs_c, _ = center_field(ps, dev.yf, dev.zf)
        f_u, f_c, f_w = Ezs_u / Ezb_u, Ezs_c / Ezb_c, Ezs_w / Ezb_w
        conv = ms["res_final"] < 1e-5
        zmid = 0.5 * (dev.zf[0] + dev.zf[-1]) * 1e9
        toward_gate = ms["zbar_nm"] > zmid       # holes pulled up toward gate
        ok = bool(conv and 0 < f_u < 1 and 0 < f_c < 1 and toward_gate)
        all_ok &= ok
        e_rows.append(dict(V_g=Vg, n_iter=ms["n_iter"],
                           Ez_uniform_bare=Ezb_u, Ez_uniform_scr=Ezs_u,
                           screen_uniform=f_u, screen_center=f_c,
                           screen_weighted=f_w, zbar_nm=ms["zbar_nm"],
                           converged=bool(conv), passed=ok))
        print(f"(e) Vg={Vg:+.1f} V: {ms['n_iter']:3d} it, screen"
              f" uniform x{f_u:.4f} center x{f_c:.4f}"
              f" (weighted x{f_w:.4f}), <z>={ms['zbar_nm']:.2f} nm"
              f" -> {'PASS' if ok else 'FAIL'}")
    out["e_scf"] = dict(rows=e_rows, passed=bool(all_ok))

    out["all_passed"] = bool(all(v["passed"] for v in
                                 (out["a_plate"], out["b_stack"], out["c_line"],
                                  out["d_grid"], out["e_scf"])))
    print(f"Section A: {'ALL PASS' if out['all_passed'] else 'FAILURES PRESENT'}")
    return out


def section_B():
    print("=== Section B: V_g -> effective-field calibration map ===")
    Vgs = [-0.05, -0.1, -0.2, -0.3, -0.5, -0.75, -1.0, -2.0]
    nls = [0.0, 2e7, 5e7]
    dev = FinPoisson()
    rows = []
    print(f"{'V_g':>6} {'n_l':>8} {'iters':>5} {'Ez_eff[V/m]':>12}"
          f" {'Ey_eff[V/m]':>12} {'asym_z':>8} {'<z>[nm]':>8}")
    for Vg in Vgs:
        for nl in nls:
            _, m = _scf_case(dev, Vg, nl)
            rows.append(m)
            print(f"{Vg:6.2f} {nl:8.1e} {m['n_iter']:5d} {m['Ez_eff']:12.4e}"
                  f" {m['Ey_eff']:12.4e} {m['asym_z']:8.4f} {m['zbar_nm']:8.2f}")
    print("Section B done.")
    return dict(rows=rows, note="toy-density calibration; Ez_eff/Ey_eff are "
                "density-weighted <-grad phi>; asym_z = vertical phi drop / "
                "(max-min) over fin")


def section_C():
    print("=== Section C: geometry sensitivity at V_g=-0.5 V, n_l=2e7/m ===")
    rows = []
    print(f"{'t_ox[nm]':>8} {'gate':>5} {'iters':>5} {'Ez_eff[V/m]':>12}"
          f" {'Ey_eff[V/m]':>12} {'asym_z':>8} {'asym_y':>8} {'bowl_y[mV]':>10}"
          f" {'bowl_z[mV]':>10} {'screen':>7}")
    for t_ox in (2e-9, 3e-9, 5e-9):
        for gate in ("tri", "top"):
            dev = FinPoisson(t_ox=t_ox, gate=gate)
            phib = dev.solve(-0.5)
            phis, m = _scf_case(dev, -0.5, 2e7)
            Ezb_u, _ = effective_field(dev.phi_fin(phib), dev.yf, dev.zf)
            Ezs_u, _ = effective_field(dev.phi_fin(phis), dev.yf, dev.zf)
            m.update(t_ox_nm=t_ox * 1e9, gate=gate,
                     Ez_uniform_bare=Ezb_u,
                     screen_factor=Ezs_u / Ezb_u)
            rows.append(m)
            print(f"{t_ox*1e9:8.1f} {gate:>5} {m['n_iter']:5d}"
                  f" {m['Ez_eff']:12.4e} {m['Ey_eff']:12.4e}"
                  f" {m['asym_z']:8.4f} {m['asym_y']:8.4f}"
                  f" {m['bowl_y_V']*1e3:10.3f} {m['bowl_z_V']*1e3:10.3f}"
                  f" {m['screen_factor']:7.4f}")
    print("Section C done.")
    return dict(rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--sec", default="all", choices=["A", "B", "C", "all"])
    args = ap.parse_args()
    if args.sec in ("A", "all"):
        res = section_A()
        _checkpoint("A", res)
        _save_numbers("poisson2d", {
            "A_all_passed": res["all_passed"],
            "A_plate_relerr": res["a_plate"]["err_field_rel"],
            "A_stack_relerr": res["b_stack"]["rel_err"],
            "A_line_relerr": res["c_line"]["max_rel_err"],
            "A_gridconv_mV": res["d_grid"]["max_dphi_V"] * 1e3,
            "A_screen_uniform_Vg-0.5": res["e_scf"]["rows"][2]["screen_uniform"],
            "A_screen_center_Vg-0.5": res["e_scf"]["rows"][2]["screen_center"],
        })
    if args.sec in ("B", "all"):
        res = section_B()
        _checkpoint("B", res)
        anchor = [r for r in res["rows"]
                  if r["V_g"] == -0.5 and r["n_l"] == 2e7][0]
        _save_numbers("poisson2d", {
            "B_Ez_eff_Vg-0.5_nl2e7": anchor["Ez_eff"],
            "B_asym_z_Vg-0.5_nl2e7": anchor["asym_z"],
        })
    if args.sec in ("C", "all"):
        res = section_C()
        _checkpoint("C", res)
        tri = [r for r in res["rows"] if r["gate"] == "tri"]
        top3 = [r for r in res["rows"]
                if r["gate"] == "top" and r["t_ox_nm"] == 3.0][0]
        _save_numbers("poisson2d", {
            "C_Ez_eff_tox_spread": max(r["Ez_eff"] for r in tri)
                                   - min(r["Ez_eff"] for r in tri),
            "C_Ez_eff_topgate_over_trigate":
                top3["Ez_eff"] / [r for r in tri if r["t_ox_nm"] == 3.0][0]["Ez_eff"],
        })
    print("poisson2d: requested sections complete.")


if __name__ == "__main__":
    main()
