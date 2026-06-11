"""Review-round-5 item R5-4 (electron side): realistic step/terrace morphology.

Referee demand: distributions of terrace widths and step bunches from actual
miscut angles; wire orientation vs step edges; finite-width phase ramps;
correlated valley-orbit amplitude suppression; full distributions and
rare-event tails, not only medians.

Model. On Si(001) a single atomic step has height a/4 = 0.1358 nm
(a = 0.5431 nm), so a miscut angle theta_m gives a mean terrace width
s_mean = (a/4)/tan(theta_m). Terrace widths are gamma-distributed with shape
kgam (kgam=1: Poisson/exponential, kgam=10: nearly periodic; kgam=4 is the
realistic default for well-prepared vicinal Si). A wire at angle chi to the
step-edge direction crosses steps with effective along-wire spacing
s_eff = s_mean/|sin(chi)| (chi=90 deg: wire perpendicular to edges, worst
case). Each single-step crossing advances the valley-orbit phase by
dphi = 2 k0 (a/4) = 0.85*pi (same-sign staircase, as in fig9 'fixed' mode);
a double-step bunch advances by 1.7*pi == -0.3*pi.

Grids. Sections A/B/C/F use the fig9 baseline grid (N=500, dx=5 nm,
L=2.5 um). Sections D (tanh ramps, w = 0/2/5/10 nm) and E (amplitude dips of
FWHM ~3 nm) involve nm-scale features that a 5 nm grid cannot resolve, so
they run at N=1000, dx=2.5 nm (same L) and include their own same-grid
reference case (w=0 resp. fsup=1.0).

Metrics per realization: E0 (MZM splitting, must stay near zero) and E2
(lowest excitation above the MZM pair), both in ueV, from the 8N BdG
spectrum. Per parameter point we report p5/p50/p95 of E2 and the fraction of
realizations with E2 < 1 ueV (rare/typical failure metric).

Checkpointing: raw per-seed results go to output/data/morphology.json after
every parameter point; rerunning any section resumes. Summaries additionally
go to key_numbers.json via run_analysis.save_numbers("morphology", ...).
"""
import argparse
import json
import os
import time

import numpy as np

from majorana_sim import build_wire_two_valley_iv, solve_lowest, UEV
from run_analysis import (save_numbers, _make_vo_profile, DATA,
                          M_SI, G_SI, DELTA, DX, ALPHA_DEMO)

# fig9 baseline operating point
MU, B, AL = 35.0, 1.5, ALPHA_DEMO        # ueV, T, eV*A
LAM0 = 75.0                              # ueV (|lambda|, splitting 2|lam|)
N0, DX0 = 500, DX                        # baseline grid, L = 2.5 um
NF, DXF = 1000, 2.5e-9                   # fine grid for sections D, E
A_SI = 0.5431e-9
H_STEP = A_SI / 4.0                      # single-step height, 0.1358 nm
DPHI1 = 0.85 * np.pi                     # single-step phase jump
CKPT = os.path.join(DATA, "morphology.json")


def s_mean_from_miscut(theta_deg):
    """Mean terrace width (m) for miscut angle theta_m (deg) on Si(001)."""
    return H_STEP / np.tan(np.radians(theta_deg))


def gen_step_positions(L, s_mean, kgam, chi_deg, rng, bunch=1):
    """Along-wire step-crossing positions: gamma-renewal process with mean
    along-wire spacing s_eff = bunch * s_mean / |sin(chi)|.  bunch>1 keeps
    the same total height budget (miscut) but groups steps into bunches,
    each crossed as ONE composite phase jump."""
    s_eff = bunch * s_mean / abs(np.sin(np.radians(chi_deg)))
    pos, x = [], 0.0
    while True:
        x += rng.gamma(kgam, s_eff / kgam)
        if x >= L:
            break
        pos.append(x)
    return np.asarray(pos)


def morph_profile(N, dx, pos, dphi, w_ramp=0.0, fsup=1.0, wsup=3e-9,
                  lam0=LAM0):
    """Complex valley-orbit profile lambda(x) for step crossings at `pos`.
    dphi: phase advance per crossing (scalar). w_ramp: tanh ramp width (m) --
    phase rises as dphi/2*(1+tanh(2(x-p)/w)) so ~76% of the jump occurs
    within +-w/2 of the step; w_ramp=0 reproduces the abrupt fig9 convention
    phi(x>=p) += dphi. fsup<1: |lambda| dips to fsup*lam0 at each step with a
    Gaussian profile of FWHM wsup (~3 nm: VO coupling weakens near steps)."""
    x = np.arange(N) * dx
    phi = np.zeros(N)
    amp = np.ones(N)
    for p in pos:
        if w_ramp > 0:
            phi += dphi * 0.5 * (1.0 + np.tanh(2.0 * (x - p) / w_ramp))
        else:
            phi += dphi * (x >= p)
        if fsup < 1.0:
            sig = wsup / 2.3548
            amp *= 1.0 - (1.0 - fsup) * np.exp(-0.5 * ((x - p) / sig)**2)
    return lam0 * amp * np.exp(1j * phi)


def solve_E0_E2(vo, N, dx):
    H = build_wire_two_valley_iv(N, dx, MU, B, DELTA, AL, M_SI, G_SI,
                                 vo_profile_ueV=vo)
    E, _ = solve_lowest(H, k=6)
    Ea = np.sort(np.abs(E)) / UEV
    return float(Ea[0]), float(Ea[2])


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def save_ckpt(ck):
    with open(CKPT, "w") as f:
        json.dump(ck, f, indent=1)


def stats(E2, E0):
    E2 = np.asarray(E2, float)
    E0 = np.asarray(E0, float)
    return dict(p5=round(float(np.percentile(E2, 5)), 3),
                p50=round(float(np.percentile(E2, 50)), 3),
                p95=round(float(np.percentile(E2, 95)), 3),
                frac_E2_below_1ueV=round(float(np.mean(E2 < 1.0)), 3),
                E0_med=round(float(np.median(E0)), 4),
                E0_max=round(float(np.max(E0)), 4),
                n=len(E2))


def run_point(ck, sec, key, seeds, make_vo, N, dx, t0):
    """Compute (or resume) one parameter point: per-seed (E0, E2)."""
    ck.setdefault(sec, {})
    if key in ck[sec]:
        return ck[sec][key]
    E0s, E2s = [], []
    for s in seeds:
        vo = make_vo(s)
        e0, e2 = solve_E0_E2(vo, N, dx)
        E0s.append(round(e0, 6))
        E2s.append(round(e2, 6))
    ck[sec][key] = dict(E0=E0s, E2=E2s)
    save_ckpt(ck)
    print(f"  [{sec}/{key}] done  t={time.time()-t0:.0f}s", flush=True)
    return ck[sec][key]


NSEED = 12          # sections B-F
NSEED_A = 14        # section A (matches fig9)


def summarize(sec, ck, label_extra=None):
    out = {}
    for key, rec in ck.get(sec, {}).items():
        if key.startswith("_"):
            out[key] = rec
            continue
        out[key] = stats(rec["E2"], rec["E0"])
    if label_extra:
        out.update(label_extra)
    save_numbers("morphology", {f"sec{sec}": out})
    return out


# ------------------------------------------------------------- section A
def secA():
    """Validation. (1) gamma generator with kgam=1 (exponential terraces),
    chi=90, abrupt jumps, no suppression, s_mean = 50 nm must statistically
    reproduce fig9's Poisson same-sign scan (median E2 ~ 0.65 ueV, seed
    family [71,1,50,s], 14 seeds; different RNG stream -> within ~2x).
    (2) exact pipeline check: _make_vo_profile with identical seeds must
    reproduce the per-seed fig9_scan.npz E2 values to 1e-6."""
    t0 = time.time()
    ck = load_ckpt()
    seeds = list(range(NSEED_A))

    def vo_gamma(s):
        rng = np.random.default_rng([71, 1, 50, s])
        pos = gen_step_positions(N0 * DX0, 50e-9, 1.0, 90.0, rng)
        return morph_profile(N0, DX0, pos, DPHI1)

    run_point(ck, "A", "gamma_kgam1_s50nm", seeds, vo_gamma, N0, DX0, t0)

    def vo_exact(s):
        rng = np.random.default_rng([71, 1, 50, s])
        return _make_vo_profile(N0, DX0, 50e-9, "poisson", "fixed", rng)

    rec = run_point(ck, "A", "exact_fig9_poisson_fixed", seeds, vo_exact,
                    N0, DX0, t0)

    # compare with fig9 checkpoint (scenario 1 = poisson/fixed, 50 nm = js 4)
    cmp_note = {}
    npz = os.path.join(DATA, "fig9_scan.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        ref = np.asarray(z["E2"])[1, 4, :NSEED_A]
        diff = np.abs(np.asarray(rec["E2"]) - ref)
        cmp_note = dict(_exact_check_max_abs_diff_ueV=float(np.max(diff)),
                        _exact_check_pass=bool(np.max(diff) < 1e-5),
                        _fig9_ref_median_ueV=round(float(np.median(ref)), 3))
    else:
        cmp_note = dict(_exact_check="fig9_scan.npz not found; "
                                     "stored per-seed values only")
    ck["A"].update(cmp_note)
    save_ckpt(ck)
    g = stats(ck["A"]["gamma_kgam1_s50nm"]["E2"],
              ck["A"]["gamma_kgam1_s50nm"]["E0"])
    e = stats(ck["A"]["exact_fig9_poisson_fixed"]["E2"],
              ck["A"]["exact_fig9_poisson_fixed"]["E0"])
    summarize("A", ck, dict(_target_median_ueV=0.65,
                            _gamma_vs_exact_median_ratio=round(
                                g["p50"] / max(e["p50"], 1e-9), 3)))


# ------------------------------------------------------------- section B
def secB():
    """Miscut scan: theta_m in {0.05,0.1,0.2,0.5} deg, kgam=4, chi=90."""
    t0 = time.time()
    ck = load_ckpt()
    for th in [0.05, 0.1, 0.2, 0.5]:
        sm = s_mean_from_miscut(th)

        def vo(s, sm=sm, th=th):
            rng = np.random.default_rng([91, 2, int(th * 1000), s])
            pos = gen_step_positions(N0 * DX0, sm, 4.0, 90.0, rng)
            return morph_profile(N0, DX0, pos, DPHI1)

        run_point(ck, "B", f"theta{th}deg_s{sm*1e9:.0f}nm",
                  list(range(NSEED)), vo, N0, DX0, t0)
    summarize("B", ck)


# ------------------------------------------------------------- section C
def secC():
    """Orientation scan: theta_m=0.1 deg, chi in {90,45,20,10,5} deg."""
    t0 = time.time()
    ck = load_ckpt()
    sm = s_mean_from_miscut(0.1)
    for chi in [90, 45, 20, 10, 5]:
        seff = sm / abs(np.sin(np.radians(chi)))

        def vo(s, chi=chi):
            rng = np.random.default_rng([91, 3, chi, s])
            pos = gen_step_positions(N0 * DX0, sm, 4.0, chi, rng)
            return morph_profile(N0, DX0, pos, DPHI1)

        run_point(ck, "C", f"chi{chi}deg_seff{seff*1e9:.0f}nm",
                  list(range(NSEED)), vo, N0, DX0, t0)
    summarize("C", ck)


# ------------------------------------------------------------- section D
def secD():
    """Finite-width tanh phase ramps, w in {0,2,5,10} nm, theta_m=0.1 deg,
    kgam=4, chi=90.  Fine grid (dx=2.5 nm) so w is resolved; w=0 is the
    same-grid abrupt reference."""
    t0 = time.time()
    ck = load_ckpt()
    sm = s_mean_from_miscut(0.1)
    for w_nm in [0, 2, 5, 10]:

        def vo(s, w_nm=w_nm):
            rng = np.random.default_rng([91, 4, s])  # paired: same terraces for all w
            pos = gen_step_positions(NF * DXF, sm, 4.0, 90.0, rng)
            return morph_profile(NF, DXF, pos, DPHI1, w_ramp=w_nm * 1e-9)

        run_point(ck, "D", f"ramp_w{w_nm}nm", list(range(NSEED)),
                  vo, NF, DXF, t0)
    summarize("D", ck, dict(_grid="N=1000, dx=2.5nm (ramps resolved)", _paired="same terrace seeds for all w"))


# ------------------------------------------------------------- section E
def secE():
    """Correlated VO-amplitude suppression: |lambda| dips to fsup*lam0
    (Gaussian, FWHM 3 nm) at each step, fsup in {1.0,0.7,0.5,0.3},
    theta_m=0.1 deg, kgam=4, chi=90, abrupt jumps.  Fine grid."""
    t0 = time.time()
    ck = load_ckpt()
    sm = s_mean_from_miscut(0.1)
    for fs in [1.0, 0.7, 0.5, 0.3]:

        def vo(s, fs=fs):
            rng = np.random.default_rng([91, 5, s])  # paired: same terraces for all fsup
            pos = gen_step_positions(NF * DXF, sm, 4.0, 90.0, rng)
            return morph_profile(NF, DXF, pos, DPHI1, fsup=fs)

        run_point(ck, "E", f"fsup{fs}", list(range(NSEED)), vo, NF, DXF, t0)
    summarize("E", ck, dict(_grid="N=1000, dx=2.5nm (3nm dips resolved)", _paired="same terrace seeds for all fsup"))


# ------------------------------------------------------------- section F
def secF():
    """Step bunching: same height budget as theta_m=0.1 deg, but steps
    arrive in bunches of 2 (mean bunch spacing 2*s_mean, each bunch one
    1.7pi == -0.3pi jump), kgam=4, chi=90.  Compare with secB theta=0.1."""
    t0 = time.time()
    ck = load_ckpt()
    sm = s_mean_from_miscut(0.1)

    def vo(s):
        rng = np.random.default_rng([91, 6, 2, s])
        pos = gen_step_positions(N0 * DX0, sm, 4.0, 90.0, rng, bunch=2)
        return morph_profile(N0, DX0, pos, 2 * DPHI1)

    run_point(ck, "F", "bunch2_theta0.1deg", list(range(NSEED)),
              vo, N0, DX0, t0)
    summarize("F", ck, dict(_note="each bunch = one 1.7pi (== -0.3pi) jump; "
                                  "compare secB theta0.1deg"))


SECS = dict(A=secA, B=secB, C=secC, D=secD, E=secE, F=secF)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all",
                    choices=list(SECS) + ["all"])
    args = ap.parse_args()
    for name in (SECS if args.sec == "all" else [args.sec]):
        print(f"=== section {name} ===", flush=True)
        SECS[name]()
