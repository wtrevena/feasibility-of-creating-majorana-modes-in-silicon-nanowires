#!/usr/bin/env python3
"""Integrity manifest for the silicon-Majorana repository.

Writes MANIFEST.json at the repo root, containing

  * "ledger_files": sha256 + size of every ledger file (output/data/*.json);
  * "figures":      sha256 of every output/fig*.png and output/fig13_convergence.pdf;
  * "documents":    sha256 of paper.pdf, supplement.pdf, paper.tex, supplement.tex;
  * "git":          current commit hash and dirty flag;
  * "paper_numbers": a hand-curated map from the headline numbers quoted in
    paper.tex (and README.md) to the ledger entries that generate them
    (output/data/key_numbers.json and output/data/transport_numbers.json).

The curated map is the CI payload: every entry is re-read from the live ledger
and compared against the value frozen here.  If any entry no longer matches --
i.e. a regeneration changed a number the paper quotes -- the script prints the
mismatches and exits nonzero, so drift between the manuscript and the code is
caught mechanically.

Usage:
    python tools/manifest.py            # verify curated numbers, then (re)write MANIFEST.json
    python tools/manifest.py --check    # verify only; writes nothing (used by CI)

stdlib only (json/hashlib/subprocess); no numpy/scipy required.

Ledger-path syntax: dot-separated keys, with ['quoted'] segments for keys that
contain dots/spaces/unicode and [int] for list indices.  Paths beginning with
"transport." resolve inside output/data/transport_numbers.json; everything else
resolves inside output/data/key_numbers.json.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Hand-curated headline numbers (paper.tex / README.md -> ledger).
# "value" is the ledger-exact number (the paper quotes it rounded); "paper"
# records where and how it is quoted.  The LEDGER is the source of truth:
# if a regeneration changes a ledger value, this script fails and the curated
# entry (and the manuscript!) must be revisited.
# --------------------------------------------------------------------------
PAPER_NUMBERS = {
    # --- conduction band: SOC shortfall (Sec 3.1, fig3) ---
    "soc_intrinsic_optimistic_gap_ueV": {
        "value": 1.5506,
        "ledger": "fig3.gap_at_intrinsic_Si_opt_1e3_ueV",
        "paper": "abstract & Sec 3.1: best intrinsic topological gap ~1.5 ueV (optimistic alpha=1e-3 eV.A)"},
    "soc_intrinsic_typical_gap_ueV": {
        "value": 0.15565,
        "ledger": "fig3.gap_at_intrinsic_Si_typ_1e4_ueV",
        "paper": "Sec 3.1: 0.16 ueV at typical alpha=1e-4 eV.A"},
    "soc_alpha_for_20ueV_idealized_eVA": {
        "value": 0.0153,
        "ledger": "fig3.alpha_needed_for_20ueV",
        "paper": "Sec 3.1: robust 20 ueV requires alpha ~ 0.015 eV.A (idealized)"},
    "soc_alpha_for_20ueV_DeltaB_eVA": {
        "value": 0.0205,
        "ledger": "fig3.alpha_needed_for_20ueV_with_DeltaB",
        "paper": "Sec 3.1: 0.021 eV.A with parent-gap suppression Delta(B); abstract 'alpha >~ 0.02'"},
    # --- conduction band: g=2 catch-22 (Sec 3.2, fig3) ---
    "catch22_max_Delta_2T_ueV": {
        "value": 115.8,
        "ledger": "fig3.max_Delta_allowing_topology_at_2T_ueV",
        "paper": "abstract & Sec 3.2: induced gaps above 116 ueV cannot reach topology at B<=2 T (g=2)"},
    "catch22_optimal_Delta_ueV": {
        "value": 98.0,
        "ledger": "fig3.optimal_Delta_at_alpha005_ueV",
        "paper": "Sec 3.2: optimal parent gap Delta0 ~ 98 ueV with Delta(B) suppression"},
    # --- valley physics / step junctions (Sec 3.3, fig9) ---
    "clean_wedge_gap_ueV": {
        "value": 22.22,
        "ledger": "fig9.single_step_E2_vs_dphi_over_pi['0.00']",
        "paper": "Sec 3.3: 22 ueV clean gap of the inter-valley-paired wire"},
    "single_step_gap_ueV": {
        "value": 4.03,
        "ledger": "fig9.single_step_E2_vs_dphi_over_pi['0.85']",
        "paper": "abstract & Sec 3.3: single step binds a 4.0 ueV state = 18% cap of the 22 ueV clean gap"},
    "ensemble50nm_vicinal_median_ueV": {
        "value": 0.65,
        "ledger": "fig9.median_E2_at_50nm_by_scenario_ueV['Poisson steps, same-sign Δφ (vicinal miscut)']",
        "paper": "Sec 3.3: ensembles at 50 nm step spacing retain ~0.7-2.3 ueV (lower end)"},
    "ensemble50nm_rough_median_ueV": {
        "value": 2.29,
        "ledger": "fig9.median_E2_at_50nm_by_scenario_ueV['Poisson steps, random phase (rough, zero winding)']",
        "paper": "Sec 3.3: ensembles at 50 nm step spacing retain ~0.7-2.3 ueV (upper end)"},
    "vicinal_single_steps_median_ueV": {
        "value": 0.79,
        "ledger": "fig9.extensions['vicinal_single_steps_0.85pi'].median",
        "paper": "Sec 3.3: vicinal single-step ensemble baseline (vs bunched bi-steps)"},
    "bistep_bunched_median_ueV": {
        "value": 1.69,
        "ledger": "fig9.extensions['bunched_bisteps_1.7pi'].median",
        "paper": "Sec 3.3: bunched double-height steps ~2x milder than single steps"},
    # --- transport verification (Sec 5, transport_numbers.json) ---
    "transport_gap_clean_ueV": {
        "value": 37.34,
        "ledger": "transport.clean.transport_gap_ueV",
        "paper": "Sec 5: clean transport gap 37.3 ueV"},
    "transport_bulk_gap_ueV": {
        "value": 36.83,
        "ledger": "transport.clean.bulk_gap_ueV",
        "paper": "Sec 5: vs analytic bulk gap 36.8 ueV"},
    "transport_W_half_ueV": {
        "value": 548.3,
        "ledger": "transport.comparison.W_half_transport_ueV",
        "paper": "Sec 5: transport half-gap disorder threshold ~548 ueV"},
    "spectral_W_half_ueV": {
        "value": 410.6,
        "ledger": "fig5.case1.W_half_ueV",
        "paper": "Sec 5: spectral half-gap threshold ~410 ueV (engineered-electron case)"},
    "anderson_L1p5um_median_ueV": {
        "value": 0.5,
        "ledger": "transport.valley.vicinal_50nm_vs_L['L1.5um'].median",
        "paper": "Sec 3.3: Anderson-insulator transport gap 0.5/5.6/23 ueV at L=1.5/3/6 um (L=1.5)"},
    "anderson_L3um_median_ueV": {
        "value": 5.64,
        "ledger": "transport.valley.vicinal_50nm_vs_L['L3.0um'].median",
        "paper": "Sec 3.3: ... (L=3)"},
    "anderson_L6um_median_ueV": {
        "value": 23.04,
        "ledger": "transport.valley.vicinal_50nm_vs_L['L6.0um'].median",
        "paper": "Sec 3.3: ... (L=6)"},
    # --- valence band: LK / k.p constraints (Sec 4.1, fig10, kp6, kp6_110) ---
    "lk_lso_range_nm": {
        "value": [32.0, 101.0],
        "ledger": "fig10.lso_range_nm",
        "paper": "Sec 4.1: reproduces measured spin-orbit-length window l_so = 32-101 nm"},
    "lk4_gx_bracket": {
        "value": [0.4, 1.07],
        "ledger": "kp6.bracket_gx_range_4band",
        "paper": "Sec 4.1: 4-band seven-geometry bracket keeps g_x in [0.4, 1.1]"},
    "kp6_gx_bracket_100": {
        "value": [0.04, 0.93],
        "ledger": "kp6.bracket_gx_range_6band",
        "paper": "Sec 4.1: six-band [100] moves g_x DOWN, to [0.04, 0.93]"},
    "measured_geyer_gxx": {
        "value": [1.86, 2.31],
        "ledger": "kp6.measured_Geyer_gxx",
        "paper": "Sec 4.1: measured FinFET g_xx ~ 1.9-2.3 (Geyer et al.)"},
    "kp6_110_gx_bracket": {
        "value": [1.386, 1.658],
        "ledger": "kp6_110.bracket_gx_range_110",
        "paper": "abstract & Sec 4.1: [110] channel gives gate-stable g_x' = 1.4-1.7"},
    "kp6_110_mstar_production": {
        "value": 0.1944,
        "ledger": "kp6_110.table_10x12_110.Ez10.mstar",
        "paper": "abstract & Sec 4.1: [110] m* ~ 0.20 (production point Ez=10 MV/m)"},
    "kp6_110_alpha_range_eVA": {
        "value": [0.015, 0.075],
        "ledger": "kp6_110.alpha110_range_eVA",
        "paper": "abstract & Sec 4.1: [110] alpha up to 0.075 eV.A"},
    "kp6_110_gap_closure_fraction": {
        "value": 0.634,
        "ledger": "kp6_110.benchmark_comparison.gap_closure.closure_fraction_production",
        "paper": "Sec 4.1: channel orientation closes ~60% of the g_x discrepancy"},
    # --- platform center points & optimizer-grid convergence (Sec 4.2, convergence.G_grid) ---
    "sib_meas_center_bare_ueV": {
        "value": 11.012,
        "ledger": "convergence.G_grid.SiB_meas.10x10x20.gap",
        "paper": "Sec 4.2: measured-field Si:B center point 11.0 ueV bare on finer optimizer grids"},
    "sib_meas_center_renorm_lo_ueV": {
        "value": 9.817,
        "ledger": "convergence.G_grid.SiB_meas_renorm.14x14x28.gap",
        "paper": "Sec 4.2: 9.8-10.1 ueV renormalized on finer grids (lower end)"},
    "sib_meas_center_renorm_hi_ueV": {
        "value": 10.129,
        "ledger": "convergence.G_grid.SiB_meas_renorm.10x10x20.gap",
        "paper": "Sec 4.2: 9.8-10.1 ueV renormalized on finer grids (upper end)"},
    "sib_pauli_center_bare_ueV": {
        "value": 30.615,
        "ledger": "convergence.G_grid.SiB_pauli.10x10x20.gap",
        "paper": "README headline: Pauli-limited thin-film Si:B ~30 ueV static-bare (grid-stable)"},
    "sib_pauli_center_renorm_ueV": {
        "value": 19.682,
        "ledger": "convergence.G_grid.SiB_pauli_renorm.10x10x20.gap",
        "paper": "Sec 6 chain: Pauli Si:B 19.7 ueV static-renormalized (grid-stable)"},
    "parent_model_variation_lo_ueV": {
        "value": 7.98,
        "ledger": "convergence.H_parent['D0x0.8_Bc2=0.3T'].gap",
        "paper": "Sec 4.2: center point moves over 8.0-10.6 ueV under +-20% parent-model variations (lower)"},
    "parent_model_variation_hi_ueV": {
        "value": 10.59,
        "ledger": "convergence.H_parent['D0x1.2_Bc2=0.4T'].gap",
        "paper": "Sec 4.2: ... (upper)"},
    # --- parents & orientation (Sec 4.2, fig8/fig10/fig11) ---
    "al_center_renorm_ueV": {
        "value": 34.4,
        "ledger": "fig8.center_Al_renormalized",
        "paper": "abstract & Sec 4.2: Al film 34 ueV metallization-renormalized at the center point"},
    "al_best_bare_orientations_ueV": {
        "value": 55.0,
        "ledger": "fig11['empirical (Geyer-class) — Al film (B$_{c∥}$=2 T, B$_{c⊥}$=0.1 T)'].best",
        "paper": "abstract & Sec 4.2: up to 55 ueV bare over field orientations (empirical tensor)"},
    "sib_thick_tilted_best_ueV": {
        "value": 19.0,
        "ledger": "fig10.best_gap_by_scenario['Si:B thick, B∥ẑ (uses g_z)'][0]",
        "paper": "abstract & Sec 4.2: thick Si:B with tilted field supports 10-19 ueV (upper end)"},
    "film_pauli_limit_BP_T": {
        "value": 1.112,
        "ledger": "fig8.BP_T",
        "paper": "Sec 4.2: Pauli limit B_P = 1.11 T"},
    "film_xi_min_nm": {
        "value": 28.7,
        "ledger": "fig8.film_design['Hc2perp_0.4T'].xi_nm",
        "paper": "Sec 4.2: coherence lengths xi = 29-57 nm from measured B_c2,perp (lower)"},
    "film_xi_max_nm": {
        "value": 57.4,
        "ledger": "fig8.film_design['Hc2perp_0.1T'].xi_nm",
        "paper": "Sec 4.2: ... (upper)"},
    "film_gap_min_ueV": {
        "value": 20.9,
        "ledger": "fig8.film_design['Hc2perp_0.1T'].d15nm.gap_ueV",
        "paper": "Sec 4.2: d=10-20 nm Pauli-limited films support 21-27 ueV (lower end)"},
    "film_gap_max_ueV": {
        "value": 26.6,
        "ledger": "fig8.film_design['Hc2perp_0.4T'].d10nm.gap_ueV",
        "paper": "Sec 4.2: ... (upper end)"},
    # --- quasiparticle poisoning (Sec 4.2, fig8.qp_poisoning) ---
    "qp_parent_gap_sib_meas_ueV": {
        "value": 29.1,
        "ledger": "fig8.qp_poisoning['SiB_measured_B0.33T'].parent_gap_ueV",
        "paper": "Sec 4.2: at operating fields the Si:B parent gap is 29-33 ueV (lower)"},
    "qp_parent_gap_sib_pauli_ueV": {
        "value": 33.1,
        "ledger": "fig8.qp_poisoning['SiB_pauli_hyp_B1.0T'].parent_gap_ueV",
        "paper": "Sec 4.2: ... (upper)"},
    "qp_Teff_sib_meas_mK": {
        "value": 25.0,
        "ledger": "fig8.qp_poisoning['SiB_measured_B0.33T'].Teff_for_xqp_1e-6_mK",
        "paper": "Sec 4.2: x_qp = 1e-6 requires T_eff <= 25-29 mK (measured Si:B)"},
    "qp_Teff_sib_pauli_mK": {
        "value": 29.0,
        "ledger": "fig8.qp_poisoning['SiB_pauli_hyp_B1.0T'].Teff_for_xqp_1e-6_mK",
        "paper": "Sec 4.2: ... (Pauli Si:B)"},
    "qp_Teff_al_mK": {
        "value": 130.0,
        "ledger": "fig8.qp_poisoning['Al_film_B1.0T'].Teff_for_xqp_1e-6_mK",
        "paper": "Sec 4.2: Al reference 130 mK"},
    # --- dynamic self-energy control (Sec 6, realism.SE_selfenergy) ---
    "dynSE_sib_meas_ueV": {
        "value": 7.61,
        "ledger": "realism.SE_selfenergy.SiB_meas.best_gap",
        "paper": "abstract & Sec 6: dynamic self-energy center-point gap 7.6 ueV (measured Si:B, Gamma*=10)"},
    "dynSE_sib_pauli_ueV": {
        "value": 14.92,
        "ledger": "realism.SE_selfenergy.SiB_pauli.best_gap",
        "paper": "abstract & Sec 6: 14.9 ueV (Pauli Si:B, Gamma*=30)"},
    "dynSE_al_ueV": {
        "value": 24.22,
        "ledger": "realism.SE_selfenergy.Al.best_gap",
        "paper": "abstract & Sec 6: 24.2 ueV (Al, Gamma*=45)"},
    # --- [110] platform re-evaluation (Sec 4.2 / Supplement S10.3, platform110) ---
    "p110_sib_meas_dynamic_ueV": {
        "value": 3.77,
        "ledger": "platform110.C_dynamic.Ez10_SiB_meas.best",
        "paper": "abstract & Sec 4.2: [110] measured-field Si:B drops to 3.8 ueV dynamic (below Tier S)"},
    "p110_pauli_renorm_lo_ueV": {
        "value": 16.62,
        "ledger": "platform110.A_inplane_gx.Ez10_SiB_pauli_renorm.gap",
        "paper": "Sec 4.2: [110] Pauli film 17-19 ueV static renormalized (lower)"},
    "p110_pauli_renorm_hi_ueV": {
        "value": 18.69,
        "ledger": "platform110.A_inplane_gx.Ez30_SiB_pauli_renorm.gap",
        "paper": "Sec 4.2: ... (upper)"},
    "p110_pauli_dynamic_ueV": {
        "value": 12.95,
        "ledger": "platform110.C_dynamic.Ez30_SiB_pauli.best",
        "paper": "Sec 4.2: [110] Pauli film 13 ueV dynamic"},
    "p110_al_renorm_lo_ueV": {
        "value": 26.02,
        "ledger": "platform110.A_inplane_gx.Ez10_Al_renorm.gap",
        "paper": "Sec 4.2: [110] Al 26-32 ueV static renormalized (lower)"},
    "p110_al_renorm_hi_ueV": {
        "value": 32.08,
        "ledger": "platform110.A_inplane_gx.Ez30_Al_renorm.gap",
        "paper": "Sec 4.2: ... (upper)"},
    "p110_al_dynamic_lo_ueV": {
        "value": 16.93,
        "ledger": "platform110.C_dynamic.Ez10_Al.best",
        "paper": "Sec 4.2: [110] Al 17-18 ueV dynamic (lower)"},
    "p110_al_dynamic_hi_ueV": {
        "value": 18.45,
        "ledger": "platform110.C_dynamic.Ez30_Al.best",
        "paper": "Sec 4.2: ... (upper)"},
    # --- realism controls: correlated disorder, pairing mix, morphology, orbital (Sec 6) ---
    "corr_disorder_median_lo_ueV": {
        "value": 3.52,
        "ledger": "realism.CD_correlated_disorder.mu_grf_lc100nm_rms50.p50",
        "paper": "Sec 6: correlated mu(x) at 50 ueV RMS collapses median gap to 3.5-6.8 ueV (lower)"},
    "corr_disorder_median_hi_ueV": {
        "value": 6.81,
        "ledger": "realism.CD_correlated_disorder.mu_grf_lc25nm_rms50.p50",
        "paper": "Sec 6: ... (upper)"},
    "iid_disorder_median_ueV": {
        "value": 28.94,
        "ledger": "realism.CD_correlated_disorder.mu_iid_rms50.p50",
        "paper": "Sec 6: where the iid baseline retains 29 ueV"},
    "pairing_mix_eta_star": {
        "value": 0.242,
        "ledger": "pairing_mix.E_threshold_refine.eta_star_2x_deepening",
        "paper": "Sec 6: 2x-deeper bound state by eta* ~ 0.24"},
    "yield_failure_frac_0p05deg": {
        "value": 0.25,
        "ledger": "morphology.secB['theta0.05deg_s156nm'].frac_E2_below_1ueV",
        "paper": "Sec 6: 25-33% failure fraction at 0.05-0.1 deg miscut (0.05 deg)"},
    "yield_failure_frac_0p1deg": {
        "value": 0.333,
        "ledger": "morphology.secB['theta0.1deg_s78nm'].frac_E2_below_1ueV",
        "paper": "Sec 6: ... (0.1 deg)"},
    "orbital_suppression_15nm_pct": {
        "value": 0.044,
        "ledger": "orbital.design_point.suppression_pct_vs_W['15nm']",
        "paper": "abstract & Sec 6: orbital gap suppression <0.05% at fin widths (W=15 nm)"},
}

DOCUMENTS = ["paper.pdf", "supplement.pdf", "paper.tex", "supplement.tex"]


# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(path: str):
    """'a.b['c.d'][0].e' -> ['a', 'b', 'c.d', 0, 'e']"""
    toks, i, n = [], 0, len(path)
    while i < n:
        c = path[i]
        if c == ".":
            i += 1
        elif c == "[":
            j = path.index("]", i)
            seg = path[i + 1:j].strip()
            if seg[:1] in ("'", '"'):
                toks.append(seg[1:-1])
            else:
                toks.append(int(seg))
            i = j + 1
        else:
            j = i
            while j < n and path[j] not in ".[":
                j += 1
            toks.append(path[i:j])
            i = j
    return toks


def resolve(path: str, ledgers):
    toks = tokenize(path)
    if toks and toks[0] == "transport":
        cur, toks = ledgers["transport"], toks[1:]
    else:
        cur = ledgers["key_numbers"]
    for t in toks:
        cur = cur[t]
    return cur


def close(a, b, rtol=1e-9):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a, b = float(a), float(b)
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception as exc:  # git absent / not a checkout: record, do not fail
        return {"commit": None, "dirty": None, "error": str(exc)}


def verify_paper_numbers(ledgers):
    """Returns (n_ok, failures); failures is a list of printable strings."""
    n_ok, failures = 0, []
    for name in sorted(PAPER_NUMBERS):
        spec = PAPER_NUMBERS[name]
        try:
            live = resolve(spec["ledger"], ledgers)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            failures.append("[MISSING ] %s: cannot resolve ledger path %r (%s)"
                            % (name, spec["ledger"], exc))
            continue
        if close(spec["value"], live):
            n_ok += 1
        else:
            failures.append("[MISMATCH] %s: curated=%r ledger=%r path=%s"
                            % (name, spec["value"], live, spec["ledger"]))
    return n_ok, failures


def main(argv):
    check_only = "--check" in argv

    ledgers = {
        "key_numbers": json.loads((ROOT / "output/data/key_numbers.json").read_text(encoding="utf-8")),
        "transport": json.loads((ROOT / "output/data/transport_numbers.json").read_text(encoding="utf-8")),
    }

    n_ok, failures = verify_paper_numbers(ledgers)
    print("Curated paper numbers: %d checked, %d OK, %d failing."
          % (len(PAPER_NUMBERS), n_ok, len(failures)))
    if failures:
        print()
        for line in failures:
            print(line)
        print()
        print("MANIFEST FAIL: the paper quotes numbers the ledger no longer supports.")
        print("The ledger (output/data/*.json) is the source of truth: update the")
        print("curated map in tools/manifest.py AND the manuscript text together.")
        return 1

    if check_only:
        print("MANIFEST CHECK OK (no file written).")
        return 0

    # ---- hashes -----------------------------------------------------------
    ledger_files = {}
    for p in sorted((ROOT / "output" / "data").glob("*.json")):
        rel = p.relative_to(ROOT).as_posix()
        ledger_files[rel] = {"sha256": sha256_of(p), "size": p.stat().st_size}

    figures = {}
    fig_paths = sorted((ROOT / "output").glob("fig*.png")) + [ROOT / "output" / "fig13_convergence.pdf"]
    for p in fig_paths:
        if not p.exists():
            print("MANIFEST FAIL: missing figure %s" % p)
            return 1
        figures[p.relative_to(ROOT).as_posix()] = {"sha256": sha256_of(p)}

    documents = {}
    for name in DOCUMENTS:
        p = ROOT / name
        if not p.exists():
            print("MANIFEST FAIL: missing document %s" % name)
            return 1
        documents[name] = {"sha256": sha256_of(p)}

    manifest = {
        "_comment": ("Generated by tools/manifest.py. paper_numbers maps headline numbers "
                     "quoted in paper.tex/README.md to their ledger entries; the script "
                     "exits nonzero if any curated value stops matching the ledger. "
                     "Regenerate with: python tools/manifest.py"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_info(),
        "ledger_files": ledger_files,
        "figures": figures,
        "documents": documents,
        "paper_numbers": PAPER_NUMBERS,
    }

    out = ROOT / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print("Wrote %s: %d ledger files, %d figures, %d documents, %d curated paper numbers."
          % (out.name, len(ledger_files), len(figures), len(documents), len(PAPER_NUMBERS)))
    print("git: %s%s" % (manifest["git"].get("commit"),
                         " (dirty)" if manifest["git"].get("dirty") else ""))
    print("MANIFEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
