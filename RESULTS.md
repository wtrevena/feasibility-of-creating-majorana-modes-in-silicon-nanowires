# Results: Feasibility of Majorana Zero Modes in Silicon

**TL;DR — After three adversarial review rounds (the last of which audited our own
positive claims as hard as the negative ones): conduction-band silicon is dead for
quantified, multiply-confirmed reasons. A hole-based all-silicon design survives, but at
recalibrated numbers: using only *measured* Si:B critical fields and the Luttinger–Kohn
constrained (not free) hole parameters, the accessible topological gap is **~10–19 µeV**
(at tilted-to-out-of-plane field with a thick Si:B parent), rising to 30+ µeV only under
the so-far-unmeasured hypothesis of Pauli-limited thin-film Si:B. Si holes proximitized
by an Al film would reach 34 µeV (renormalized center; up to 55 µeV bare
over orientations) if such an interface can be made. The decisive
unknowns are now two measurements, not theory: the parallel critical field of thin-film
Si:B, and the wire-axis hole g-factor/SOC axis in the actual device geometry.**

All numbers regenerate via `python run_analysis.py --fig 1..11` and `python transport.py`;
ledger in `output/data/key_numbers.json` + `transport_numbers.json`; exact-claim tests in
`tests.py`. Section 6 logs every correction from the review rounds.

---

## 1. Why this repository was rebuilt

The original notebook (kept in `Majorana_Si_Code.ipynb`; its figures quarantined in
`output/legacy/`) had no spin, never used its spin-orbit parameter, and broke
particle-hole symmetry with a scalar Zeeman shift — its spectrum is exactly
E_Z ± sqrt(eps²+Δ²), a trivial superconductor, and its "Majorana" was a bulk state at
−21 µeV peaked mid-wire. Its own outputs said so: a 9.76 *meter* localization length and
nine-orders-inconsistent overlaps. No result from it survives.

## 2. Methods and validation

`majorana_sim.py`: spinful BdG (Lutchyn–Oreg) with exact PHS by construction; analytic
bulk dispersion verified against direct Bloch diagonalization; two-valley models with
the physical inter-valley pairing channel and arbitrary complex valley-orbit profiles
(gauge-covariant Dresselhaus SOC); 2D multi-subband strip. `transport.py`: recursive
Green's-function transmission and a scattering-matrix invariant sign(det r) (kwant is
not buildable in this environment — no root, source-only package; the RGF engine is
validated against `build_wire` to 10⁻¹³ and against Sancho–Rubio lead self-energies).
`lk_holes.py`: 4-band Luttinger–Kohn fin model (validated: exact bulk masses along
[100]; α(E=0)=0 by symmetry; Hermiticity exact). `tests.py` pins the exact claims:
PHS+Hermiticity for every builder, dispersion vs Bloch, the valley equivalence theorem
(both SOC classes, five phases), Dresselhaus gauge covariance across branch cuts, and
the gap-closing point. Fig. 1: gap closes at the analytic B* = 0.864 T; end-localized
zero mode; exponential splitting (ξ ≈ 166 nm); finite-wire gap tracks bulk.

## 3. Findings

### F1. The magnetic field is not silicon's main problem (Fig. 2)
Phase boundary identical regardless of SOC; topological phase opens at 0.86 T. What
collapses is protection: max gap 1.6 µeV (B ≤ 3 T) at optimistic intrinsic α = 10⁻³ eV·Å
vs 50 µeV at InSb-class α — 32× at identical phase diagrams. (Model-internal: clean,
single subband, no orbital effects.)

### F2. Electron SOC shortfall (Fig. 3)
Best topological gap over (µ, B ≤ 2 T): 0.16 µeV at typical intrinsic α = 10⁻⁴ eV·Å;
1.55 µeV at optimistic 10⁻³ (1.4 with Δ(B) pair-breaking). Reaching 20 µeV needs
α ≈ 0.015 (idealized) / 0.021 eV·Å (with Δ(B)) — an optimistic lower bound (orbital and
metallization corrections push it up). Micromagnet proposals span α_eff ≈ 2×10⁻³–3×10⁻²
eV·Å (Turcotte et al. 2020), and a 2025 InAs measurement (arXiv:2505.06040 — to be
human-verified) suggests the upper end is real, so engineered-electron-Si is marginal,
not absurd — but the valley physics of F6/F9 stacks on top of it.

### F3. The g = 2 catch-22 (Fig. 3)
Any induced gap ≥ E_Z(2 T) = 116 µeV can never go topological at B ≤ 2 T. In the
SOC-limited regime relevant to engineered devices (α ≳ 0.01 eV·Å) the Δ = 100 µeV curve
saturates far below the Δ = 50 µeV curve; only in the deep small-α regime (both gaps
sub-5 µeV anyway) does larger Δ help slightly — the curves cross near α ≈ 0.01 eV·Å.
With Δ(B) included consistently the optimal parent gap is Δ₀ ≈ 98 ± 4(grid) µeV
(effective ~50 µeV at the operating field). *Prior art:* the "don't maximize coupling"
rule is Cole–Das Sarma–Stanescu (PRB 92, 174511 (2015)); ours is only the sharp g=2
quantification.

### F4. Valley degeneracy, toy model (Fig. 4)
Two-band wedge picture: exactly one valley in the window |µ ∓ E_v/2| < √(E_Z²−Δ²) gives
a protected pair; numerics reproduce the analytic boundaries. Superseded in physical
content by F6/F9.

### F5. Disorder, with a transport-level check (Figs. 5, 12)
Spectral study (per-case converged dx/L, 16 seeds, per-case baselines): gap halves at
W ≈ 100–300 µeV (intrinsic, clean gap 1.4 µeV), ≈ 410 µeV (α=0.05), ≈ 520 µeV (α=0.15).
The RGF transport upgrade shows the spectral proxy is *conservative* at moderate
disorder (localized subgap states carry no current): transport W_half ≈ 548 µeV for the
α=0.05 case, and the collapse at W = 800 µeV is abrupt — a class-D zero-energy
transmission anomaly with 7/10 seeds flipping to trivial (invariant sign), not a soft
gap. Intrinsic-Si translation: the tolerance corresponds to mean free paths of order
the µm coherence length at ~0.1 meV Fermi energy — far beyond demonstrated low-density
Si wires, to our knowledge. W is an iid-onsite amplitude (corr. length = dx);
compare via scattering rates, not raw W.

### F6. The physical valley channel: inter-valley pairing (Fig. 6)
(i) Exact statement, tested in-repo for both SOC classes and five phases: for ANY
uniform valley-orbit phase the inter-valley-paired wire is unitarily equivalent to two
decoupled bands at µ ∓ |λ| with pairing ±Δ — smooth interfaces are benign. (ii) Steps
destroy the protection: at the wedge point the uniform-phase gap is 22 µeV; a sparse
1 step/µm profile already yields median 14 µeV; at 50 nm mean spacing ~1.9 µeV
(valley-scalar SOC) / 2.8 µeV (phase-locked Dresselhaus, gauge-covariant), and the
spectral proxy keeps falling with L (2.6/2.1/1.0 µeV at 1.5/3/6 µm) — upper bounds.
(iii) The TRS-breaking valley polarization ν_z is the channel-level pair-breaker.
The MZM splitting stays small (median 0.09–0.25 µeV); damage appears in the gap.

### F7. Multi-subband: 1D treatment is safe for Si electrons (Fig. 7)
Domes at each subband bottom with max gap 38.0/37.9 µeV vs 36.8 µeV 1D prediction;
subband spacing (1.5 meV) ≫ topological window (0.14 meV). First two subbands scanned.

### F8. The all-silicon hole stack — recalibrated (Fig. 8)
The Si valence band has no valleys (F4/F6/F9 do not apply) and strong gate-tunable SOC.
The free-box numbers (median ~31 µeV; literal corner 19.3 µeV — the earlier "100%
clears 20 µeV" was a grid artifact, true fraction ~97–99%) survive only under two
assumptions that round-3 review broke: (a) "Pauli-limited Si:B" — every published Si:B
critical field is 0.1–0.4 T and orbital-limited; our B* ≈ 1 T operating points assume
unmeasured thin-film physics. With measured-class Bc2 = 0.4 T: center point
**9.9 µeV** (B = 0.33 T, Δ_ind = 11 µeV); with metallization renormalization
(Z = 1−Δ_ind/Δ_p): 9.1. The hypothetical Pauli film gives 30.6 raw / 19.7 renormalized.
The thickness analysis (`sib_film_design`, item 5) makes the hypothesis concrete: with
coherence lengths from *measured* perpendicular critical fields (ξ = 29–57 nm), films of
d ≤ 18–36 nm have parallel orbital critical fields above the 1.11 T Pauli limit, and
d = 10–20 nm films support **21–27 µeV** in the film-compatible in-plane configuration
(empirical g = 2.1). The laser-doped SOI process already produces films in this range —
so "Pauli-limited Si:B" is a fabrication target, not speculation. Unmodeled risk:
T_c itself degrades in thin Si:B epilayers (measured thickness dependence exists);
the d-target must beat both constraints simultaneously.
Holes + Al film: 50.4 raw / 34.4 renormalized. (b) The "self-tuning below the catch-22
ceiling" framing was wrong — removing the suppression changes nothing; the optimizer is
simply applying Cole's weak-coupling rule, which works for any parent. What the
same-crystal Si:B interface genuinely buys is disorder-free, *controllable* coupling —
not metallization immunity. Finite-wire check (in-repo generator): E₀ = 0,
E₂ = 31.3 µeV ≈ bulk (2%), 95% end-localization at 3 µm, gap to W ≈ 300 µeV disorder.
Quasiparticle poisoning, now quantified (`qp_poisoning.py`): at the
operating fields the Si:B parent gap is 29–33 µeV, so x_qp = 10⁻⁶ requires
T_eff ≤ 25–29 mK — beyond demonstrated QP thermalization (typical T_eff ≳ 100 mK gives
x_qp ≈ 3–5%, no usable parity lifetime). The Al-film reference (150 µeV at field) needs
only T_eff ≤ 130 mK. **This makes the all-Si stack, as bounded by measured critical
fields, a candidate for MZM *detection* experiments but not, without parent-gap
engineering, for parity-coherent qubits.**

### F9. Step physics: two channels, and the winding claim retired (Fig. 9)
Round-3 control test (run in-repo): same-sign steps (vicinal miscut, median E₂ =
0.65 µeV at 50 nm) vs sign-randomized ±Δφ (zero net winding, 0.73 µeV) are statistically
indistinguishable — **net phase winding is NOT the dominant mechanism**; per-step
junction physics is. Each single-atomic step (Δφ = 2k₀a/4 ≈ 0.85π, accidentally near π
in Si) acts as a Josephson-like junction binding a subgap state: a single step caps the
gap at 4.0 µeV (18% of clean), reaching a Kitaev domain-wall zero mode at Δφ = π
(in-repo scan: 22.2 → 19.8 → 13.5 → 4.0 → 0.01 µeV at Δφ/π = 0/0.25/0.5/0.85/1).
The miscut dependence is therefore a *cliff at the first step* plus density-driven
decay — not a winding threshold. Smooth phase gradients are a real but subdominant
second channel (in-repo ramp: gap 22.2 → 16.9 → 6.3 → 0.9 µeV at q = 0/4/8/16 ×10⁶ m⁻¹,
suppression scale set by the SOC velocity and E_v-*protected*, per the round-3
analysis). Extensions (in-repo, `fig9_extensions`): double-height bi-steps
(Δφ = 1.7π ≡ −0.3π) are ~2× milder (median 1.7 vs 0.8 µeV at 50 nm; single bi-step caps
at 18.6 vs 4.0 µeV) — step-BUNCHED vicinal surfaces are preferable; |λ| suppression near
steps (Hosseinkhani–Burkard) worsens things (0.31 µeV), so the phase-only model is an
upper bound; step density along the wire scales as |cos χ| of the wire–miscut azimuth —
wires laid along step edges see no steps. Transport-level check (`transport_valley.py`,
8-orbital RGF validated to 10⁻¹⁴ against the closed chain; clean wedge E_T = 22.7 vs
spectral 22.0 µeV): a single step transmits resonantly through its bound state
(E_T = 2.7 µeV), and step-dense wires become Anderson insulators whose END-TO-END
transport gap *grows* with length (medians 0.5 / 5.6 / 23.0 µeV at L = 1.5/3/6 µm at
50 nm spacing) even as localized subgap states persist (spectral ~1 µeV) — i.e.
transport spectroscopy can show a hard-looking gap while true local protection is
degraded. Both metrics are therefore reported; neither alone equals topological
protection in the step-disordered regime. Experimental upshot: induced-
gap hardness in electron devices should degrade with miscut/step *density* from the
very first step — and Si holes are immune by construction.

### F10. Luttinger–Kohn constrains the hole box — the α–g covariation is real (Fig. 10)
4-band LK fin model (10×12 nm hard wall): α(E) rises to 0.017–0.054 eV·Å with
l_so = 32–101 nm (validating against the measured 20–60 nm FinFET range), m* ≈ 0.44
(heavier than the 0.25 assumed), and the SOC axis comes out in-plane-transverse (ŷ) for
this geometry. The covariation is unfavorable exactly as feared: the wire-axis g
collapses (1.1 → 0.5) as the same gate field builds α, while g_z grows to 2.4–4.2.
Consequently the *physical operating curve* misses the free box's high-gap corner: along
it, the accessible gaps are 19.0 µeV (thick measured-Si:B, B ∥ ẑ via g_z), 15.2 µeV
(hypothetical Pauli Si:B, B ∥ x̂), 25.2 µeV (Al film, B ∥ x̂). Caveats: hard-wall
rectangle vs real tri-gate; 4-band (no split-off, Δ_SO = 44 meV); no strain. The
geometry/field bracket (`fig10.g_bracket`, 7 configurations spanning 7–16 nm cross
sections and 3–30 MV/m): g_x stays in [0.4, 1.1] everywhere — the measured
g_xx ≈ 1.9–2.3 is NOT reproducible within 4-band hard-wall LK, so the discrepancy is
formally attributed to tri-gate electrostatics, strain, and 6-band/cubic corrections,
and every platform number is carried under BOTH tensors (here and F11). Resolving it
needs device-specific 6-band+Poisson input — flagged as the top theory item.

### F11. Field orientation decides the platform (Fig. 11)
Tilted-field gap maps (exact 4×4 dispersion; criterion on the SOC-perpendicular Zeeman
component — a parallel-only field is the trivial configuration) for LK and
empirical-Geyer-class tensors × {thick measured-Si:B, Al film}: the all-Si stack is
orientation-tolerant but modest (best 13.8–14.6 µeV at θ ≈ 25–40° tilt; only 3–9 µeV for
in-plane-along-wire); the Al film is in-plane-locked but strong (41–55 µeV). Out-of-
plane fields kill film parents; thick Si:B does not care — **the all-silicon design that
survives is: thick laser-doped Si:B parent, field tilted toward out-of-plane, riding
g_z.**

## 4. Verdict

1. **Si conduction band: infeasible** — every channel (SOC, disorder-vs-gap, valley
   steps from the first atomic step, orbital/metallization headroom) fails independently.
2. **The all-silicon hole stack survives at ~10–19 µeV** using only measured Si:B
   critical fields, thick-parent geometry, tilted field — equivalent to 115–220 mK,
   demanding but not absurd; 30+ µeV requires demonstrating Pauli-limited thin-film
   Si:B — which the thickness analysis turns into a concrete target: 10–20 nm films
   are orbitally consistent with Pauli limiting at measured coherence lengths and
   support 21–27 µeV (modulo T_c(d) suppression, unmodeled). QP poisoning caps the
   role: detection platform yes; parity-coherent qubit only with parent-gap
   engineering (T_eff ≤ ~30 mK otherwise unreachable). **The two decisive measurements:** (i) B_c∥ of <20 nm laser-doped Si:B films;
   (ii) g-tensor + SOC axis of a gated Si hole wire in the actual geometry.
3. **Si holes + Al film: 34 µeV renormalized (center), up to 55 µeV bare over
   orientations** — fig11 maps are NOT renormalized — if a hard-gap Al/Si-hole
   interface can be made
   (no demonstration exists; metallization-renormalized numbers quoted).
4. Before submission: 6-band LK with tri-gate electrostatics; transport invariant
   extended to all disorder figures; QP-poisoning estimate; verify flagged references.

## 5. Caveats
Single-band wire caricature downstream of LK inputs; LK model is 4-band hard-wall;
parent gap models are caricatures (GL orbital / Zeeman-linear); spectral proxy retained
where transport not yet wired (figs 6, 9); no finite temperature; no electrostatic
self-consistency; novelty statements rest on automated literature searches
(Ge-hole proposals — Maier–Klinovaja–Loss PRB 90, 195421 — are the close precedent;
the Si-hole/Si:B *combination* appears unexamined).

## 6. Verification and corrections log
Round 1 (electron results): fig5 finite-size/convergence redo; inter-valley pairing
(F6); Δ(B); interp/seed/checkpoint fixes. Round 2: TR-odd "Dresselhaus" replaced by
phase-locked operator (regression-tested); L-convergence bounds; optimal-Δ consistency;
prior-art repositioning. Round 3 (four agents, including against our own positive
claims): (a) Dresselhaus bond-phase branch-cut bug → circular mean + gauge-covariance
test; (b) two evidence blocks had no generating code → in-repo generators + merge-safe
ledger; (c) "100% of box" was a grid artifact → corner = 19.3 µeV; (d) winding
mechanism falsified by sign-randomization control → F9 rewritten around per-step
junctions; (e) Si:B "Pauli-limited" contradicted by all published critical fields →
measured-field scenario now primary; "self-tuning" claim retired (suppression-free
optimizer reproduces the numbers); (f) metallization renormalization quantified;
(g) F3 "everywhere" corrected (curves cross); stale 0.17/0.156 and "≤0.1 µeV" fixed.
In-house: tilted-field criterion bug (B ∥ SOC axis misclassified as topological) caught
and fixed before release. Transport upgrade: spectral W_half 410 → 548 µeV (transport),
abrupt class-D breakdown identified. kwant: not buildable here (documented); RGF
validated in its place.

*Provenance statement (for the eventual manuscript):* the verification process used
independent adversarial review rounds — numerical re-derivation of headline numbers,
controlled counter-experiments, raw-data audits, and literature checks — run by AI
agents, with all corrections logged above and every quantitative claim regenerable from
committed code. To be disclosed in the acknowledgments or supplement.

## 7. Key references
- Kitaev (2001); Lutchyn–Sau–Das Sarma PRL 105, 077001; Oreg–Refael–von Oppen PRL 105,
  177002 — model. Mourik et al. Science 336, 1003; Zhang et al. retraction (2021);
  Pan–Das Sarma PRR 2, 013377 — disorder history. Aghaee et al. (Microsoft) PRB 107,
  245423 — measurement standard.
- Cole–Das Sarma–Stanescu PRB 92, 174511 (2015) — optimal coupling (F3 prior art).
- Kjaergaard et al. PRB 85, 020503(R); Turcotte et al. PRB 102, 125425; arXiv:2505.06040
  (verified: permalloy arrays on InAs/Al, synthetic α = 0.022 eV·nm = 0.22 eV·Å — 10× our
  silicon requirement of ~0.02 eV·Å; unit conflation caught in review round 4) — engineered SOC.
- Losert et al. PRB 108, 125405; Hosseinkhani–Burkard PRB 100, 125309 — valley
  splitting, interface steps (F6/F9 context).
- Osca, Ruiz & Serra, PRB 89, 245405 (2014) (verified) — tilted-field criterion
  (F11 context); see also PRB 90, 115429 (critical angle).
- Bosco–Hetényi–Loss PRX Quantum 2, 010348; Camenzind et al. Nat. Electron. 5, 178;
  Geyer et al. Nat. Phys. (2024); Voisin et al. Nano Lett. 16, 88 — Si hole SOC and
  g-tensors (F8/F10/F11 inputs).
- Bustarret et al. Nature 444, 465 (2006) (Si:B superconductivity discovery);
  Grockowiak et al./C2N line, PRB 81, 020501(R) (2010) (GILD (001) films, T_c up to
  ~0.6 K at ~8 at% B, H_c2 ≈ 0.1 T class); Chiodi et al. line: thickness dependence of
  T_c (HAL hal-00957153), all-Si SQUID (Duvauchelle et al., arXiv:1508.04075 — corrected
  attribution), laser-annealed superconducting polycrystalline SOI, T_c 0→0.5 K
  (arXiv:2404.02748, 2024); review: "Superconductivity in silicon", arXiv:2108.03031 —
  the Si:B parent. All four previously flagged references now hand-verified.
- Maier–Klinovaja–Loss PRB 90, 195421 (2014) — Ge-hole MZMs (closest precedent).
- Reeg–Loss–Klinovaja PRB 97, 165425 — metallization. Nijholt–Akhmerov PRB 93, 235434 —
  orbital effects (unmodeled).

## 7. Convergence and sensitivity (reviewer round 4, convergence.py)

All headline numbers were re-derived under systematic refinement
(output/data/convergence.json; fig13_convergence.pdf/.png rendered from the
same JSON by tools/fig13_pgf.py — pgfplots, network/matplotlib-free; the
matplotlib version regenerates via `python convergence.py`):

- **dx**: clean wedge gap 22.165 / 22.221 / 22.234 µeV at dx = 10/5/2.5 nm;
  single-step (0.85π) 4.037/4.034/4.033. Production dx = 5 nm is converged to
  <0.3% / <0.1%.
- **L**: clean and single-step values are L-independent above 2.5 µm. The
  50-nm step-ensemble median is L-dependent BY PHYSICS (Anderson
  localization): 2.54 / 0.65 / 0.08 µeV at L = 1.25/2.5/5 µm — consistent
  with the transport-section warning; ensemble medians are quoted as
  L-tagged upper bounds, never as bulk gaps.
- **seeds**: 56-seed median 0.64 µeV, bootstrap 95% CI [0.52, 0.81];
  the published 14-seed value 0.65 sits inside the CI.
- **µ**: clean gap and step state vary smoothly (9.7→27.6 and 1.9→6.2 µeV
  over µ = 15–60); the wedge point µ=35 is not fine-tuned.
- **E_v**: the wedge point sits in a finite topological window. E_v = 100/150
  µeV behave as expected; E_v ≳ 225 µeV at fixed µ=35, B=1.5 T exits the
  topological phase (the "clean 33.8" at E_v=300 is a TRIVIAL gap) — single-
  valley topology requires re-centering µ as E_v grows, as stated in F6.
- **nk**: bulk-gap k-grid fully converged at production nk (≤0.02% drift).
- **optimizer grid**: SiB_pauli 30.6 µeV exact across grids; renormalized
  19.7→20.2 (+2.7%) at the finest grid. SiB_meas center point is the one
  number that moves: 9.9 → 11.0 µeV bare (10.1 renormalized) on finer
  optimizer grids — the published 9.9/9.1 were ~10% conservative. Quoted
  ranges "10–19 µeV" are unaffected (the lower edge rises slightly).
- **parent model**: ±20% on Si:B Δ0 and Bc2 ∈ {0.3, 0.4} T move the
  measured-field center point over 8.0–10.6 µeV (Bc2 dominates; Δ0 is
  second order). The platform verdict is insensitive to the GL caricature.

Engine note: in sandboxes without scipy, compat/ provides a validated
pure-numpy block-tridiagonal shift-invert Lanczos (compat/test_shim.py:
matches dense eigh to ~1e-12 relative; H-space residual validation guards
against spurious Ritz values; shift offset 1e-7·scale bounds the solve
condition number). Production runs with scipy are unaffected.

## 8. Round-5 realism controls (realism.py, orbital.py, pairing_mix.py, morphology.py)

- **Dynamic self-energy** (validated: static limit 0.07%, grid 0.001%): optimal
  dynamic gaps 7.6 / 14.9 / 24.2 µeV (measured-Si:B Γ*=10, Pauli-Si:B Γ*=30,
  Al Γ*=45) vs static bare 11.0 / 30.6 / 50.4 — static bare DEMOTED to upper
  bound; static-renormalized caricature directionally validated (dynamic sits
  24–30% below it; matched-Δind agreement ≲7%). Dynes: 50 mK operation needs
  γ_D/Δp ≲ 1e-3 (Si:B). Parent Δ(x) inhomogeneity benign ≤25%.
- **Orbital coupling**: <0.05% gap suppression at 7–16 nm fin widths at the
  tilted design point (∝ W^1.5 B_perp²); even 1 T full out-of-plane costs
  0.3/1.4/8.9% at W=10/20/40 nm. Orientation maps survive.
- **Pairing mixing**: step mechanism never weakens with intra-valley fraction η
  (gauge argument: the step reappears as a pairing-phase junction); single-step
  state 2× deeper by η*≈0.24, near-zero at η=1. Ensembles HEAL with η
  (0.57→4.5 µeV at η=0→0.3): our η=0 ensembles are conservative. η≪η*
  expected (2k0 momentum transfer suppression).
- **Morphology**: yield, not median, is binding — failure fraction P(E2<1µeV)
  = 25–33% at 0.05–0.1° miscut even where medians look viable; χ≲10° wires
  along step edges or step-bunched templates empty the tail; ramps ≤10 nm and
  VO dips shift medians only 10–15%.
- **Correlated disorder**: smooth µ(x) at 50 µeV RMS collapses hole median gap
  to 3.5–6.8 µeV (λc=25–100 nm) where iid retains 28.9 — iid overstates
  robustness 4–8×; gate smoothness is the binding fabrication constraint.
  α fluctuations negligible; g fluctuations ~12%.

Manuscript: title softened ("quantified constraints"), master assumption table
(Table 1) + operational tiers S/N/Q (Table 2) added; abstract carries the
dynamic-corrected numbers. F8 clarification: the 21–27 µeV film range is over
Pauli-limited thicknesses only (d below the per-branch Pauli thickness).
