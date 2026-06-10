# Paper outline (pivoted)

## Working title
**"An all-silicon route to Majorana zero modes: hole nanowires proximitized by
superconducting Si:B"**
(alt: "Majorana zero modes in silicon: why electrons fail and holes do not")

## Target
PRB (regular article) or PRX Quantum / PRApplied if the device section is strengthened.
SciPost Phys. as alternative. Post to arXiv cond-mat.mes-hall first.

## Abstract (draft)
Semiconductor–superconductor Majorana platforms have avoided silicon for good reasons:
weak spin–orbit coupling, a small g-factor, and valley degeneracy. We quantify all three
for conduction-band wires within a validated Bogoliubov–de Gennes framework and show
they are individually fatal: the best intrinsic topological gap is ~1.5 µeV; any induced
gap above E_Z(B_c) = 116 µeV can never reach the topological phase (g = 2 "catch-22");
and — using the physical inter-valley pairing channel, for which we prove an exact
unitary equivalence making smooth-interface valley splitting benign — we identify
valley-phase *winding* from miscut-staircase interface steps as a Fulde–Ferrell-like
depairing field that destroys the gap above ~0.02° miscut. We then show the silicon
*valence* band evades every blocker: with measured Si FinFET hole parameters
(α ≈ 0.03–0.15 eV·Å, g ≈ 1.5–3), the full parameter box supports 20–36 µeV topological
gaps. Pairing the hole channel with boron-doped superconducting silicon — whose
Pauli-limited gap self-tunes below the catch-22 ceiling — yields an all-silicon,
CMOS-compatible Majorana stack. We give the optimal operating points, a falsifiable
miscut prediction for electron devices, and the design constraints (field orientation
vs hole g-anisotropy) that decide the platform.

## Figure plan (8 figures from current repo, renumbered)
1. Model validation (current fig1, compressed; partly appendix)
2. Electron verdict: gap vs α with Δ(B), platform bands (fig3)
3. Phase diagrams + catch-22 (fig2 + optimal-Δ inset; cite Cole 2015)
4. Valley physics I: equivalence theorem + step disorder (fig6, corrected SOC)
5. Valley physics II: winding/depairing mechanism + miscut prediction (fig9 + linear-ramp check)
6. **The all-silicon stack** (fig8) — centerpiece
7. Hole wire robustness: finite-size, end-localization, disorder (extend fig8 check)
8. Design constraints: g-anisotropy / field orientation map for Si:B film (TO COMPUTE)

## Required work before submission (honest list)
1. **Luttinger–Kohn 4-band justification** of the hole parameter box; α(E)–g(E)
   covariation under gate field (Bosco–Loss sweet-spot physics) → replace independent
   (α, g) scan with constraint-curve scan. Biggest theory gap.
2. **Scattering-matrix invariant + transport gap** (kwant) replacing the E₂ spectral
   proxy in all disorder figures (referee-proofing post-Pan–Das Sarma).
3. **g-anisotropy / field-orientation analysis** (new Fig. 8): the field must be
   in-plane for the Si:B film, ⊥ SOC field, ∥ large-g hole axis. This is the binding
   constraint; compute the feasible solid angle.
4. Orbital-field estimate for the planar hole geometry; metallization discussion
   (mitigated by same-crystal interface — argue, don't just assert).
5. Si:B parent details: pair-breaking model beyond Zeeman-linear caricature; check
   literature for measured Si:B critical fields in thin films.
6. Prior-art engagement: Cole 2015 (optimal coupling), Turcotte 2020 + arXiv:2505.06040
   (micromagnets), Maier–Klinovaja–Loss 2014 (Ge holes — method precedent),
   Losert 2023 (steps/wiggle well), Pan–Das Sarma (disorder standards).
7. Verify Si:B/Si-hole interface physics with an experimentalist coauthor if possible
   (laser-doped Si:B on SOI + gate-defined hole channel — is the integration credible?).

## Claims discipline
- "100% of measured box > 20 µeV" — always cite the box provenance (EDSR l_so → α; g
  from FinFET magnetospectroscopy) and the single-band caveat.
- The miscut prediction is for *electron* devices; holes are immune (no valleys) — keep
  the two threads clearly separated.
- The equivalence theorem and winding mechanism are exact statements within the model;
  state the model.
