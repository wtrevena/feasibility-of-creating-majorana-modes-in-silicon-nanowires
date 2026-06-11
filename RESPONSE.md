# Response to Adversarial Review Round 4

Manuscript: *Majorana zero modes in silicon: quantified failure of the
conduction band and a conditional hole-based route* — draft v2.

Round 4 was an AI red-team review conducted in the style of a journal
referee report (recommendation: major revision). It is not a human peer
review; it was used to close gaps before submission. All ten of its
revision demands were accepted. Summary of what changed, keyed to the
report's numbering.

## The ten specific demands

**1. SOC unit issue (0.022 eV·nm vs eV·Å).** Confirmed and corrected
(§3.1). The reviewer's reading is right and the correction *strengthens*
the marginality statement: 0.022 eV·nm = 0.22 eV·Å is an order of magnitude
**above** our 0.015–0.021 eV·Å requirement. The revised text says so
explicitly, with the caveats that the demonstration is in InAs (lighter
mass, mature stack) and that micromagnet fringe-field disorder is not
modeled here. Our code is unit-consistent throughout (α in eV·Å,
converted once; `majorana_sim.py` line 22 documents the convention).

**2. Full valley-theorem proof.** Supplement S2: basis, the rotation
U_φ = e^{−iφν_z/2}, the key algebraic fact U_φ D U_φ^T = D (the
inter-valley pairing is *exactly* invariant under the valley rotation for
any φ), band decoupling, treatment of both SOC classes, and validity
limits. The physical-channel argument (valleys are time-reversed partners;
intra-valley pairing would be an FFLO condensate) is S1.2.

**3. Single-step bound state: analytic where possible.** Supplement S3.2
derives, exactly, that band-projected pairing acquires the phase −φ(x) —
so a VO phase step *is* a Josephson junction; this is the rigorous part.
The bound-state depth is numerical; the zero-parameter transparent
short-junction formula Δ_t|cos(Δφ/2)| is tabulated against the lattice
(Supplement Table 1: ≲15% for Δφ ≲ 0.5π, 25–30% near 0.85π, exact at
Δφ → 0 and π). The text
now distinguishes the exact mapping from the numerically determined depth,
as requested ("different levels of claim").

**4. Complete parameter tables.** Supplement S4 (Tables 2–4): every
figure's m*, α, g, Δ₀, μ, B, L, dx, valley splitting, disorder model,
and seed family. Seeds are integer-list `default_rng` families,
documented in README.

**5. Convergence tests.** New `convergence.py` (8 sections, checkpointed):
dx, L, seed count (with bootstrap CI), μ, valley splitting, k-grid,
optimizer grid, and parent-model (±20% Δ₀, B_c2) sensitivity. Supplement
S5. One number moved: the measured-field Si:B center point is ~10%
*higher* (9.9 → 11.0 µeV) on finer optimizer grids; everything else is
converged at production settings. The step-ensemble L-dependence is
physical (Anderson localization) and is now labeled as such everywhere.

**6. Parent-superconductor model.** Methods + Supplement S6.1: Δ is a
static induced gap (ω = 0 limit of the tunneling self-energy); field
suppression via Δ_ind ≤ Δ_p(B); metallization via quasiparticle weight Z.
The omitted dynamical self-energy lowers gaps; the renormalized numbers
bound the direction.

**7. Measured / derived / hypothetical separation.** The abstract now
carries explicit labels; the hole section quotes the three scenarios with
their status, and the conditional-feasibility framing is stated in both
abstract and discussion.

**8. Abstract moderated.** "Conditional feasibility result, decided by two
standard measurements, not an established platform."

**9. Archival code access.** README: per-figure regeneration commands,
environment, seed ledger, Zenodo checklist; CITATION.cff added; DOI
placeholder in the Data-availability section (deposit at submission —
requires the author's account and a license choice).

**10. AI provenance.** Moved off the author line into a standard
Acknowledgments-and-provenance statement; the text now states explicitly
that AI-assisted review supplements but does not substitute for the
reproducibility materials.

## The reviewer's five decisive questions

**Q1 (electron failure robust?).** The new sensitivity sections cover μ,
discretization, length, seeds, valley splitting, and parent model. Multi-
subband occupancy was already in fig. 7; orbital effects, self-energy, and
correlated disorder remain open and are listed as such (the conclusion is
stated as robust *directionally*, with the failure quantified under the
stated caricatures).

**Q2 (step mechanism new and correct?).** The exact part is now proven
(S2–S3); positioning relative to the quantum-dot step literature
(Hosseinkhani–Burkard; Losert et al.) is in the main text: the phase shift
is known physics, the consequence under inter-valley pairing is the new
statement.

**Q3 (hole route realistic?).** Recast as conditional throughout, with
uncertainty bands (grid, parent-model, dual g-tensors). The LK-vs-measured
g_x discrepancy remains the top open theory item and is flagged in
abstract, §5, and discussion.

**Q4 (reproducible?).** README + CITATION.cff + ledger + seeded runs +
tests; archival DOI at submission.

**Q5 (journal-ready prose?).** Sweeping phrases removed ("validated …
adversarial rounds" out of the abstract); claims tied to supplement
sections; provenance conventional.

## Not yet done (honest list)

- 6-band k·p + electrostatics for g_x (top theory item, acknowledged in text).
- Dynamical parent self-energy; orbital magnetic effects; correlated disorder.
- Zenodo deposit + license (author action at submission).
- (fig13 now rendered: vector PDF from the committed JSON via
  tools/fig13_pgf.py and embedded in Supplement S5.)
