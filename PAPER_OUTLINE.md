# Paper outline (post-round-3)

## Working title
**"Majorana zero modes in silicon: quantified failure of the conduction band and a
conditional hole-based route"**
(The earlier all-Si-triumphant framing is retired: the surviving numbers are
conditional and the paper is honest about it.)

## Target
PRB regular article (the natural home for a careful feasibility + mechanism paper).

## Abstract (draft, recalibrated)
We assess Majorana feasibility in silicon within a validated BdG framework subjected to
three adversarial review rounds. Conduction-band wires fail independently on three
axes: spin-orbit coupling (best intrinsic gap ~1.5 µeV; ≥0.02 eV·Å of engineered SOC
required), a g=2 ceiling forbidding induced gaps above 116 µeV at attainable fields
(quantifying Cole et al.'s weak-coupling rule), and valley physics — where we prove an
exact two-band equivalence for smooth interfaces and show each single-atomic interface
step acts as a near-π Josephson junction (Δφ = 2k₀a/4 ≈ 0.85π) whose bound state caps
the topological gap at ~18% from the very first step. For the valence band we derive
the gate-field covariation of (α, g-tensor, m*) from a 4-band Luttinger–Kohn fin model:
the field that builds direct-Rashba SOC (α up to 0.054 eV·Å, l_so = 32–101 nm, matching
FinFET measurements) suppresses the wire-axis g below 1.1 while raising g_z to 2.4–4.2.
Field-orientation maps then select the viable designs: a thick boron-doped
superconducting-silicon parent with tilted field supports 10–19 µeV using only measured
Si:B critical fields (30+ µeV if Pauli-limited thin films can be demonstrated); an Al
film parent would support 34–55 µeV (metallization-renormalized) if a hard-gap Al/Si-
hole interface existed. Transport calculations show our spectral disorder thresholds
are conservative and identify an abrupt class-D breakdown mode. We end with the two
measurements that decide the platform.

## Figure plan
1. Validation (fig1, partly appendix) 2. Electron verdict (fig3) 3. Catch-22 (fig2+3)
4. Valley equivalence + steps (fig6) 5. Step junctions, two channels (fig9)
6. LK constraint (fig10) 7. Orientation maps (fig11) — centerpiece
8. All-Si operating points + scenarios (fig8, reframed) 9. Transport (fig12, methods)

## Pre-submission work list
1. 6-band LK + tri-gate electrostatics (Poisson) — the LK-vs-measured g_x discrepancy
   (0.5–1.1 vs 1.9–2.3) must be resolved or bracketed with device-specific input.
2. Transport invariant extended to figs 6/9 (valley cases); L-scaling shown.
3. QP-poisoning estimate for the T_c = 0.6 K parent (Δ_op/k_B T_eff analysis).
4. Verify flagged references by hand: arXiv:2505.06040; laser-annealed SOI (2024);
   Duvauchelle attribution; Osca–Ruiz–Serra volume/page.
5. Si:B parent model beyond caricature (orbital+Pauli Maki combination; film-thickness
   dependence) with experimental input if any new H_c2 data exists.
6. Wire-orientation (cos-factor) and step-bunching extensions of F9; |λ| suppression at
   steps (Hosseinkhani–Burkard).
7. Statement of review provenance: the methodology (adversarial agent rounds with
   numerical re-derivation) belongs in the acknowledgments or supplement honestly.

## Claims discipline (enforced)
- Si:B numbers always labeled measured-field vs Pauli-hypothetical; renormalized
  variants quoted alongside raw.
- LK-curve numbers quoted as the physical baseline; free-box numbers as the optimistic
  envelope bracketed by measured g-tensors.
- "100%" language retired; box statistics quoted with the literal corner value.
- The winding mechanism is retired; per-step junction physics is the F9 claim.
- Every novelty statement hedged with search methodology.
