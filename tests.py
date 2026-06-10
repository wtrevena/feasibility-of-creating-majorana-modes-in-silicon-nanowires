"""
tests.py — reproducible checks of the central exact claims.
Run: python tests.py        (all assertions must pass; <1 min)
"""
import numpy as np
from majorana_sim import (UEV, HBAR, ME, QE, build_wire, build_wire_two_valley,
                          build_wire_two_valley_iv, build_wire_2d,
                          step_phase_profile, bulk_E2_lower, EZ_J, solve_lowest)

rng = np.random.default_rng(7)
ok = lambda name: print(f"  PASS  {name}")

# 1. Particle-hole symmetry of every builder (machine precision)
Hs = {
    "build_wire(+disorder)": build_wire(80, 5e-9, 10, 1.5, 50, 0.05, 0.19, 2.0,
                                        disorder_ueV=30, rng=rng),
    "two_valley toy": build_wire_two_valley(80, 5e-9, 10, 1.5, 50, 0.05, 0.19,
                                            2.0, Ev_ueV=100, delta_iv_ueV=10,
                                            rng=rng),
    "two_valley_iv rashba": build_wire_two_valley_iv(
        80, 5e-9, 10, 1.5, 50, 0.05, 0.19, 2.0,
        vo_profile_ueV=50*np.exp(1j*step_phase_profile(80, 5e-9, 5e-8, rng)),
        valley_pol_ueV=20),
    "two_valley_iv dresselhaus": build_wire_two_valley_iv(
        80, 5e-9, 10, 1.5, 50, 0.05, 0.19, 2.0,
        vo_profile_ueV=50*np.exp(1j*step_phase_profile(80, 5e-9, 5e-8, rng)),
        soc_mode="dresselhaus"),
    "2d strip": build_wire_2d(30, 4, 5e-9, 10e-9, 600, 1.5, 50, 0.05, 0.19, 2.0),
}
for name, H in Hs.items():
    Hd = H.toarray()
    assert np.abs(Hd - Hd.conj().T).max() < 1e-40, f"non-Hermitian: {name}"
    E = np.linalg.eigvalsh(Hd)
    assert np.max(np.abs(E + E[::-1])) / UEV < 1e-6, name
ok("Hermiticity + PHS, all builders")

# 1b. Gauge covariance of the dresselhaus mode: a GLOBAL valley-phase rotation
# of the VO profile is pure gauge and must leave the spectrum invariant,
# including profiles with phases straddling the +-pi branch cut.
rng2 = np.random.default_rng(11)
phi_r = step_phase_profile(120, 5e-9, 5e-8, rng2)
for chi in [1.0, 2.0, np.pi]:
    Ha = build_wire_two_valley_iv(120, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                                  vo_profile_ueV=75*np.exp(1j*phi_r),
                                  soc_mode="dresselhaus")
    Hb = build_wire_two_valley_iv(120, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                                  vo_profile_ueV=75*np.exp(1j*(phi_r+chi)),
                                  soc_mode="dresselhaus")
    Ea = np.linalg.eigvalsh(Ha.toarray()); Eb = np.linalg.eigvalsh(Hb.toarray())
    assert np.max(np.abs(Ea - Eb)) / UEV < 1e-6, chi
ok("dresselhaus gauge covariance under global valley rotation (3 angles)")

# 2. Analytic bulk dispersion vs direct 4x4 Bloch diagonalization
m = 0.19 * ME; mu = 20 * UEV; D = 50 * UEV; EZ = EZ_J(2.0, 1.5)
aSI = 0.05e-10 * QE
s0 = np.eye(2); sx = np.array([[0,1],[1,0]]); sz = np.diag([1.,-1.])
sy = np.array([[0,-1j],[1j,0]]); tz = np.diag([1.,-1.]); tx = np.array([[0,1],[1,0]])
for k in np.linspace(0, 1.2e8, 41):
    xi = HBAR**2*k**2/(2*m) - mu
    Hk = (np.kron(tz, xi*s0 + aSI*k*sz) + np.kron(np.eye(2), EZ*sx)
          + np.kron(tx, D*s0))
    Emin = np.min(np.abs(np.linalg.eigvalsh(Hk)))
    Ean = np.sqrt(bulk_E2_lower(np.array([k]), mu, EZ, D, aSI, m))[0]
    assert abs(Emin - Ean) / UEV < 1e-8
ok("analytic dispersion == Bloch diagonalization (41 k-points)")

# 3. THE EQUIVALENCE THEOREM: inter-valley pairing + ANY uniform in-plane
#    valley-orbit phase == two decoupled bands at mu -/+ |lambda|, pairing +-D.
#    Proof sketch: U = exp(-i phi nu_z / 2) maps lambda e^{i phi} -> |lambda|
#    while preserving nu_x pairing (U nu_x U^T = nu_x for this U combined with
#    the phase gauge); a valley Hadamard then diagonalizes nu_x -> nu_z, giving
#    pairing +-Delta (per-band gauge) and splitting +-|lambda|.
N = 100
rngA = np.random.default_rng(1)
Href = build_wire_two_valley(N, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                             Ev_ueV=150, rng=rngA)         # decoupled-band form
Eref = np.linalg.eigvalsh(Href.toarray())
for phi in [0.0, 0.7, np.pi/2, 2.6, 5.0]:
    for mode in ["rashba", "dresselhaus"]:
        H = build_wire_two_valley_iv(N, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                                     vo_profile_ueV=75*np.exp(1j*phi)*np.ones(N),
                                     soc_mode=mode)
        E = np.linalg.eigvalsh(H.toarray())
        assert np.max(np.abs(E - Eref)) / UEV < 1e-6, (phi, mode)
ok("equivalence theorem: uniform phase benign, BOTH SOC classes, 4 phases")

# 4. Dresselhaus mode hosts a genuine Majorana phase (regression for the old
#    TR-odd bug, which destroyed the topological phase entirely)
H = build_wire_two_valley_iv(300, 5e-9, 35, 1.5, 50, 0.05, 0.19, 2.0,
                             vo_profile_ueV=75*np.ones(300),
                             soc_mode="dresselhaus")
E, _ = solve_lowest(H, k=6)
Ea = np.sort(np.abs(E)) / UEV
assert Ea[0] < 0.5 and Ea[2] > 5.0, Ea[:3]
ok(f"dresselhaus wedge point: E0={Ea[0]:.3f}, gap={Ea[2]:.1f} ueV (topological)")

# 5. Topological criterion: gap closes at EZ = sqrt(D^2+mu^2)
for mu_u in [0.0, 60.0]:
    Bstar = np.sqrt(50**2 + mu_u**2) * 1e-6 / (0.5 * 2.0 * 5.7883818060e-5)
    g_at = lambda B: np.sqrt(bulk_E2_lower(
        np.linspace(0, 1e8, 20001), mu_u*UEV, EZ_J(2.0, B), 50*UEV,
        0.05e-10*QE, 0.19*ME)).min() / UEV
    assert g_at(Bstar) < 0.05 and g_at(Bstar*0.8) > 1 and g_at(Bstar*1.2) > 1
ok("gap closes exactly at EZ = sqrt(Delta^2 + mu^2)")

print("ALL TESTS PASS")
