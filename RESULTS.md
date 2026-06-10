# Results: Feasibility of Majorana Zero Modes in Silicon Nanowires

**TL;DR — Silicon CAN host Majorana zero modes — but not the way everyone tried.
The conduction band fails for quantified, multiply-confirmed reasons (tiny SOC, a g=2
catch-22, and valley physics whose true killer turns out to be valley-phase WINDING:
a substrate miscut of just ~0.02 degrees acts as Fulde-Ferrell-like depairing on the
physical inter-valley pairing channel). The valence band evades every one of these
blockers: using measured Si FinFET hole parameters, 100% of the (alpha, g) parameter
box clears a 20 ueV topological gap, with ~30 ueV at the center. Pairing it with
boron-doped superconducting silicon (Si:B, Tc ~ 0.6 K) — whose Pauli-limited,
Zeeman-suppressed gap self-tunes below the catch-22 ceiling — yields an all-silicon,
CMOS-compatible Majorana stack with no valleys, no micromagnets, and 0.95
end-localization in a 3 um wire. We found no prior proposal of either ingredient.**

---

## 1. Why this repository was rebuilt

The original notebook (`Majorana_Si_Code.ipynb`, kept for the record) could not address the
title question: its Hamiltonian had no spin, never used the spin-orbit parameter it defined,
and added the Zeeman term as a scalar shift to both BdG blocks, explicitly breaking
particle-hole symmetry. Its spectrum is exactly E = E_Z ± sqrt(eps² + Δ²) — a trivial
s-wave superconductor, rigidly shifted (verified to 10⁻¹⁰ µeV). The state it called a
"Majorana mode" sits at −21 µeV with a bulk envelope peaked at the wire's *middle*
(Fig. 1b). Its own outputs flagged the problem: a 9.76 *meter* fitted localization length
and electron-hole overlaps differing by nine orders of magnitude between cells.
**No figure from the original notebook should be used.**

## 2. Corrected model and validation (Fig. 1)

`majorana_sim.py` implements the single-band Lutchyn–Oreg model (4N×4N BdG: spin, Rashba
SOC, Zeeman, singlet pairing), constructed as [[h, Δiσ_y], [h.c., −h*]] so particle-hole
symmetry is exact by construction. Validation: spectrum symmetric to 10⁻¹⁰ µeV; gap closes
and reopens at the analytic boundary (B* = 0.864 T at µ=0, Δ=50 µeV, g=2); end-localized
zero mode (94% end weight) with splitting decaying exponentially in length (ξ ≈ 166 nm at
demonstration parameters, consistent with ħv/Δ_top); finite-wire gap tracks the analytic
bulk gap. The analytic dispersion was independently verified against direct diagonalization
of the Bloch Hamiltonian to 10⁻¹¹ µeV.

## 3. Findings

### F1. The magnetic field is *not* silicon's main problem (Fig. 2)
With g = 2 and Δ = 50 µeV the topological criterion is met above B = 0.86 T, inside a
thin-film Al parent's parallel critical field. What collapses is *protection*: maximum
topological gap 1.6 µeV at optimistic intrinsic SOC (α = 10⁻³ eV·Å) vs 50 µeV at
InSb-class SOC — identical phase boundary, 32× weaker protection. (Both numbers are
clean, single-subband, orbital-free model statements; the 1.6 here allows B ≤ 3 T, the
1.5 in F2's table B ≤ 2 T.)

### F2. Quantified SOC shortfall (Fig. 3) — the central result
Maximizing the topological gap over µ and B (B ≤ 2 T):

| scenario | α (eV·Å) | best gap, Δ const | with Δ(B) pair-breaking |
|---|---|---|---|
| typical intrinsic Si (measured) | ~10⁻⁴ | 0.17 µeV | lower |
| optimistic intrinsic Si | 10⁻³ | 1.5 µeV | 1.4 µeV |
| **required for 20 µeV (robust operation)** | **0.015** | — | **0.021** |
| engineered-SOC scenario | 0.05 | 43 µeV | — |

Intrinsic Si falls short by 15× (optimistic) to 150× (typical). Including pair-breaking
suppression Δ(B) = Δ₀[1−(B/B_c)²] of the Al parent raises the requirement to
α ≈ 0.021 eV·Å. Orbital magnetic-field effects and superconductor-induced renormalization
of g and α (metallization) are *not* included and both push the requirement further up —
**treat α ≥ 0.02 eV·Å as an optimistic lower bound, plausibly 2–5× too small.** Micromagnet
proposals give synthetic α_eff ≈ 2×10⁻³–3×10⁻² eV·Å (optimized geometries: Turcotte et
al., PRB 102, 125425 (2020)), and a 2025 experiment measured α_eff ≈ 0.22 eV·Å in
InAs/Al with permalloy arrays (arXiv:2505.06040) — naive g-scaling to Si suggests
0.03–0.05 eV·Å may be reachable, putting the requirement *within* rather than at the
edge of the engineered window. The micromagnet→uniform-α mapping remains idealized
(texture disorder becomes field inhomogeneity). All of this is moot for holes (F8).

### F3. A g = 2 catch-22: better interfaces make it worse (Fig. 3)
E_Z(2 T) = 116 µeV for g = 2, so any induced gap Δ ≥ 116 µeV can never reach the
topological phase at B ≤ 2 T, and the Δ = 100 µeV curve lies below the Δ = 50 µeV curve
everywhere. With Δ(B) pair-breaking included consistently, the optimal parent gap is
Δ₀ ≈ 98 µeV (effective induced gap ~50 µeV at the operating field). *Prior art:* the
"don't maximize coupling" insight is established (Cole, Das Sarma & Stanescu, PRB 92,
174511 (2015); metallization literature) and the Δ < E_Z(B_c) requirement is
review-standard — what is ours is only the sharp g=2 quantification. Large-g materials
never face this trade-off; F8 shows a Pauli-limited parent *automatically* satisfies it.

### F4. Valley degeneracy, toy model (Fig. 4)
Two independent valley copies (intra-valley pairing, ν_z splitting, white-noise
inter-valley scattering δ_iv = 10 µeV) give the wedge picture: a single protected Majorana
pair requires exactly one valley inside the topological window, |µ ∓ E_v/2| < √(E_Z²−Δ²);
the numerical map reproduces the analytic boundaries. The often-quoted "E_v ≳ 80–140 µeV"
from this picture is window geometry at B = 1.5 T, not an optimized threshold, and the
both-valleys splitting (~0.1–1 µeV here) scales with the assumed δ_iv. F6 replaces this toy
with the physical pairing channel.

### F5. Disorder, redone after review (Fig. 5)
Per-case converged geometry (k_so·dx < 0.1, L up to 20 µm, 16 realizations, per-case clean
baselines, first-crossing thresholds):

| case | clean E₂ | W at which median E₂ halves |
|---|---|---|
| intrinsic Si, α=10⁻³, L=20 µm | 1.4 µeV | ~100–300 µeV (noisy, wide IQR) |
| engineered Si, α=0.05, L=2 µm | 37.7 µeV | ≈ 410 µeV |
| strong SOC, α=0.15, L=6 µm | 37.7 µeV | ≈ 520 µeV |

Two honest conclusions replace the earlier overclaim: (i) disorder tolerance mostly tracks
the clean gap — extra SOC at *equal* gap buys only a modest improvement (≈1.3× for 3× α);
(ii) the intrinsic-Si case fails not at "sub-µeV disorder" but because its tolerance,
translated to transport language, requires a mean free path of order the Majorana coherence
length (~µm) at a Fermi energy of ~0.1 meV — orders of magnitude beyond any demonstrated Si
wire at that density. The zero mode also degrades directly: median |E₀| rises from 0.02 to
0.29 µeV across the intrinsic scan (E₀/E₂ → 0.5, i.e. no usable Majorana). W is an
iid-onsite amplitude with correlation length = dx; thresholds are lattice-convention
dependent and should be compared via scattering rates, not raw W.

### F6. The physical valley channel: inter-valley pairing (Fig. 6) — new
Because the two relevant valleys are time-reversed partners, a uniform s-wave parent
induces *inter-valley* singlet pairing Δ(ν_x ⊗ iσ_y). Three results:

1. **Smooth interfaces are exactly benign.** For any uniform valley-orbit phase, the
   inter-valley-paired wire is unitarily equivalent to two decoupled bands at µ ∓ E_v/2
   with pairing ±Δ (verified numerically to 3×10⁻¹⁰ µeV). The wedge picture of F4 is the
   *correct* physics for a perfect interface — the worry that the pairing channel would
   qualitatively change it resolves in favor of the toy model, with one big exception:
2. **Interface steps convert valley splitting into gap-killing disorder.** Atomic steps
   scramble the valley-orbit phase. At E_v = 150 µeV and mean step spacing 50 nm the
   protecting gap collapses from ~14 µeV (smooth interface; median over seeds, wide IQR)
   to ~2 µeV (valley-scalar SOC: 1.9; phase-locked Dresselhaus SOC: 2.5), and the
   spectral-gap proxy *keeps falling with wire length* (medians 2.6/2.1/1.0 µeV at
   L = 1.5/3/6 µm) — these numbers are upper bounds, so the conclusion only strengthens.
   Valley splitting is a double-edged sword: single-valley topology needs E_v large, but
   at stepped interfaces large E_v amplifies the damage. F9 identifies the dominant
   underlying mechanism (coherent phase winding). The MZM-pair splitting itself stays
   ≤ 0.1 µeV — the damage appears in the gap, not the splitting.
3. **The true pair-breaker is valley polarization** (a ν_z energy imbalance between the
   ±k₀ valleys, which breaks time-reversal in valley space): it suppresses the
   inter-valley-paired gap directly (Fig. 6c). Physically it is small in planar devices,
   but axis-aligned wire orientations that disfavor coherent valley-orbit coupling have no
   E_v knob at all — **planar (MOS/SiGe) geometries with the valley axis out of the wire
   direction are the right design.**

### F7. Multi-subband check: the 1D treatment is safe for Si (Fig. 7) — new
A 5-subband strip (60 nm hard wall, 2D Rashba, B in-plane along the wire) shows clean
topological domes at each subband bottom with max gap 38.0/37.9 µeV vs the 1D prediction
36.8 µeV, and zero modes at 0.01–0.02 µeV inside the predicted windows |µ−E_n| < 71 µeV.
Si's heavy transverse mass makes the subband spacing (~1.5 meV) ≫ the topological window
(~0.14 meV), so subbands decouple and inter-subband SOC mixing at α = 0.05 eV·Å is
harmless. In this planar geometry with B ∥ wire, no flux threads the lattice — one more
argument for planar Si devices; full 3D orbital effects (superconductor shell, finite
thickness) remain the largest unquantified correction.

### F8. The all-silicon stack: hole channel + superconducting Si:B (Fig. 8) — new
Every conduction-band blocker is absent in the Si valence band: the maximum is at Γ (no
valleys — F4/F6/F9 do not apply), quasi-1D hole channels have strong gate-tunable
direct-Rashba-type SOC (measured via fast EDSR in Si FinFET hole qubits: spin-orbit
lengths 20–100 nm → α ≈ 0.03–0.15 eV·Å at m* ≈ 0.25 m_e), and hole g-factors are 1.5–3+
and anisotropic. Running the same optimizer used for electrons over the measured
(α, g) box:

| operating point | max topological gap | optimum (B, Δ_ind) |
|---|---|---|
| conservative (α=0.03, g=1.8) | 23.2 µeV | 1.0 T, 29 µeV |
| center (α=0.06, g=2.2) | 30.6 µeV | 1.0 T, 33 µeV |
| favorable (α=0.15, g=3) | 35.7 µeV | 0.86 T, 36 µeV |
| even g = 2.0 exactly | 28.1 µeV | — |

**100% of the measured box clears 20 µeV** (box median 30.8 µeV) — 20× the intrinsic
electron ceiling, in the same tier as the III-V platforms, with no engineering of SOC
at all. A finite-wire check at the center point: E₀ = 0 (machine zero), E₂ = 31.3 µeV
(= bulk), 95% end-localization at L = 3 µm, and the gap survives onsite disorder to
W ≈ 300 µeV.

The parent: boron-doped superconducting silicon (Si:B, laser-doped, T_c up to ~0.6 K →
Δ₀ ≈ 91 µeV; demonstrated epitaxial all-silicon junctions and CMOS-compatible 300 mm
processing). Two structural advantages: (i) parent and channel are the *same band of the
same crystal* — the most transparent proximity interface conceivable, blunting the
metallization worry; (ii) Si:B is a light element, so its thin films are Pauli-limited
(B_P ≈ 1.11 T) and the parent gap Zeeman-suppresses as Δ₀ − µ_B B — which *automatically*
keeps the induced gap below the F3 catch-22 ceiling at the operating field. The
worry that g ≈ 2 cannot beat a Pauli-limited parent (naively g > 2√2 at full
transparency) dissolves in the suppressed-parent model: the optimizer finds 28 µeV at
g = 2.0. Holes + epitaxial Al would give ~50 µeV if such an interface can be made; the
Si:B number (30 µeV) buys interface quality and full CMOS compatibility for a ~40% gap
price. Caveats: single-band caricature of the hole subband (no Luttinger–Kohn 4-band
treatment, no k³ SOC, no α–g covariation under gate field yet); hole g-anisotropy means
the field must be oriented along a large-g axis that is simultaneously ⊥ the SOC field
and in-plane for the film — the binding design constraint to be computed next.

### F9. The true valley killer is phase *winding*, not step density (Fig. 9) — new
We tested whether gap damage tracks step density or step randomness and found neither:
the dominant variable is the **net valley-phase winding**. Each single-atomic step shifts
the valley-orbit phase by ≈ 0.85π with a *sign fixed by the step direction*. A vicinal
(miscut) surface is a staircase of same-sign steps → coherent winding φ(x) ≈ qx, which
under inter-valley pairing acts exactly like Fulde–Ferrell finite-momentum depairing.
Verified directly with a pure linear phase ramp (no steps at all): the gap collapses at
q_c ≈ 2Δ_top/(ħv_F) ≈ 4×10⁶ m⁻¹ as predicted (22.2 → 16.9 → 6.3 → 0.9 µeV at
q = 0/4/8/16 ×10⁶ m⁻¹). Consequences, all counterintuitive and falsifiable:

- a substrate miscut of only **~0.02° halves the gap; ≳0.05° destroys it** (step spacing
  maps to miscut angle via tan θ = a/4 ÷ spacing);
- at fixed mean spacing 50 nm, *periodic same-sign* steps give gap ≈ 0 while
  *random-phase Poisson* steps leave 2.4 µeV — randomness is **less** damaging than
  regularity, the opposite of ordinary disorder intuition;
- position jitter at fixed density *restores* the gap monotonically (0 → 0.6 µeV as
  jitter → 100%).

The experimental prediction is then sharp and cheap: induced-gap hardness in a
proximitized Si *electron* wire should anti-correlate with miscut angle with a threshold
near 0.01–0.02°, distinguishable from generic disorder by its dependence on step
*coherence* rather than density. (Tunneling spectroscopy of an induced gap on
0°/0.1°/0.5° miscut wafers — catalog items — would test it; no working Majorana device
needed.) It also implies that "wiggle-well"-type deterministic valley-orbit engineering,
which adds a large phase-rigid component, should parametrically suppress the
winding — and that holes (F8) are immune by construction.

## 4. Verdict and recommended next steps

1. **Pristine Si conduction-band wires: infeasible** — gap ≤ 1.6 µeV best case, with
   orbital effects, Δ(B), metallization, disorder, and now miscut-winding (F9) all
   pushing the same direction.
2. **Engineered-SOC electron Si: marginal** — α ≥ 0.02 eV·Å (optimistic lower bound)
   is plausibly reachable with micromagnet arrays (F2 update), but the valley
   requirements (E_v ≳ 100 µeV *and* sub-0.02° effective miscut / phase-rigid
   valley-orbit engineering) stack on top.
3. **Si holes + Si:B (F8) is the recommendation** — every electron blocker absent or
   automatic, 20–36 µeV gaps across the *measured* parameter box, all-silicon
   CMOS-compatible stack, and apparently unproposed. This is the paper (see
   PAPER_OUTLINE.md).
4. **Before submission:** Luttinger–Kohn justification of the hole parameter box
   including α(E)–g(E) covariation under gate field; scattering-matrix invariant +
   transport gap (kwant) to replace the E₂ proxy in disorder figures; orbital-field
   estimate for the planar geometry; g-anisotropy/field-orientation analysis for the
   Si:B film constraint; engage prior art (Cole 2015; Turcotte 2020; arXiv:2505.06040;
   Ge-hole literature as method precedent).

## 5. Caveats

Now included: physical inter-valley pairing with phase-disordered valley-orbit coupling
(F6); multi-subband (F7); Δ(B) pair-breaking (F2); lattice-convergence and finite-size
controls (F5); seed-independent ensembles with IQRs. Still missing, all pushing the verdict
in the same (negative) direction: 3D orbital B effects, superconductor metallization,
correlated and electrostatic disorder, finite temperature, transverse-mass anisotropy,
strain. Parameter provenance: Si Rashba 10⁻⁵–10⁻³ eV·Å (Si/SiGe ESR, MOS
weak-antilocalization); valley splitting 30–300 µeV typical, up to ~0.5–0.8 meV in the best
MOS devices; Al thin-film B_c∥ ≈ 2–3 T; micromagnet synthetic α_eff ≈ 2×10⁻³–3×10⁻² eV·Å.
All platform α values are representative, not device-specific.

## 6. Verification and corrections log

The repository was subjected to an independent adversarial review (numerical re-derivation
of headline numbers on finer grids — all reproduced within ~4%; analytic dispersion checked
against direct Bloch diagonalization; PHS verified to machine precision; checkpoint raw
data audited). The review found, and this revision fixed: (1) the original Fig. 5 compared
a localized-Majorana wire against an unconverged wire (k_so·dx = 0.62) whose Majoranas
were not separated (L ≈ coherence length) and whose baseline "clean gap" was a finite-size
level spacing — fully redone with per-case dx/L and baselines, which *changed the F5
conclusions*; (2) the toy valley model's pairing channel — replaced by F6; (3) a
">1600 µeV" threshold that was an interpolation clamp — now reported honestly; (4)
non-monotonic-array interpolation and seed-collision bugs — replaced with first-crossing
logic and sequence seeding; (5) checkpoint files now carry parameter signatures; (6) the
α-requirement is now explicitly an optimistic lower bound with the Δ(B) variant computed.

A second independent review round then found, and this revision fixed: (7) the
"valley-antisymmetric SOC" option was a time-reversal-odd, unphysical operator that
destroyed the topological phase entirely — replaced by the physical phase-locked
Dresselhaus operator (TR-even; restores the exact two-band equivalence, regression test
in tests.py), and fig6(b)/F6 were recomputed; (8) the F6 step-disorder gap was shown not
to be L-converged — now quoted as an L-dependent upper bound with IQRs; (9) the optimal
induced gap is now computed with Δ(B) suppression (98 µeV, was inconsistently 62);
(10) k-grid bias at the smallest α removed via refined recompute (0.156, was 0.172 µeV);
(11) the central equivalence claim now has an in-repo reproducible test (tests.py,
4 phases × both SOC classes); (12) F3 repositioned against prior art; legacy-notebook
figures quarantined to output/legacy/. The E₂ spectral proxy limitation (vs a transport
gap) remains open and is listed as required work in §4.

## 7. Key references

- A. Kitaev, Phys.-Usp. 44, 131 (2001); R. Lutchyn, J. Sau, S. Das Sarma, PRL 105, 077001
  (2010); Y. Oreg, G. Refael, F. von Oppen, PRL 105, 177002 (2010) — the model.
- V. Mourik et al., Science 336, 1003 (2012); retraction of Zhang et al. (Nature, 2021);
  Pan & Das Sarma, PRR 2, 013377 (2020) — disorder and false positives.
- M. Kjaergaard, K. Wölms, K. Flensberg, PRB 85, 020503(R) (2012) — micromagnet synthetic SOC.
- C. Kloeffel, M. Trif, D. Loss, PRB 84, 195314 (2011); F. Maier, J. Klinovaja, D. Loss,
  PRB 90, 195421 (2014) — Ge/Si hole wires.
- F. Zwanenburg et al., Rev. Mod. Phys. 85, 961 (2013) — Si valley physics and SOC scales.
- B. Nijholt, A. Akhmerov, PRB 93, 245412 (2016) — orbital field effects (unmodeled here).
- C. Reeg, D. Loss, J. Klinovaja, PRB 97, 165425 (2018) — metallization/renormalization.
- Microsoft Quantum (M. Aghaee et al.), PRB 107, 245423 (2023) — the measurement standard.
- W. Cole, S. Das Sarma, T. Stanescu, PRB 92, 174511 (2015) — optimal (non-maximal)
  semiconductor–superconductor coupling: prior art for F3.
- S. Turcotte et al., PRB 102, 125425 (2020); arXiv:2505.06040 (2025) — micromagnet
  synthetic SOC, proposal and measurement.
- M. P. Losert et al., PRB 108, 125405 (2023) — valley splitting, interface steps, and
  deterministic valley-orbit engineering ("wiggle well") in Si.
- S. Bosco, B. Hetényi, D. Loss, PRX Quantum 2, 010348 (2021); L. Camenzind et al.,
  Nat. Electron. 5, 178 (2022); B. Voisin et al., Nano Lett. 16, 88 (2016) — Si hole
  direct-Rashba SOC, FinFET hole qubits, hole g-factors (the F8 parameter box).
- E. Bustarret et al., Nature 444, 465 (2006); J. E. Duvauchelle et al., PRB 96, 024503
  (2017); F. Chiodi et al., APL Mater. (2024) — superconducting Si:B, all-silicon
  Josephson junctions, CMOS-compatible superconducting SOI (the F8 parent).
