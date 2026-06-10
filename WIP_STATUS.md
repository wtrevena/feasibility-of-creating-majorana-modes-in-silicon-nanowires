# Work-in-progress status (checkpoint)

**Date:** 2026-06-09/10 session checkpoint.

## State: four-agent vet round complete; round-3 fixes partially applied

### Done since last stable point
- **Transport upgrade (transport.py, fig12, transport_numbers.json):** custom RGF
  (kwant unbuildable in sandbox). Clean transport gap 37.3 µeV validates vs bulk 36.8;
  scattering invariant Q = sign(det r) implemented (flips at B = 0.8–0.95 T vs analytic
  0.864). Verdict: the E₂ spectral proxy was *conservative* — fig5 case-1 W_half revises
  410 → ≈548 µeV; breakdown at W=800 is an abrupt class-D zero-energy anomaly, not a
  soft gap. Standalone: `python3 transport.py`.
- **Dresselhaus gauge bug fixed** (circular-mean bond phase in majorana_sim.py);
  tests.py hardened (Hermiticity asserts, gauge-covariance regression with branch-cut
  phases, equivalence at φ=5.0). All tests pass.
- **save_numbers now merges** (no tag clobbering). Orphan evidence blocks now have
  in-repo generators: `_hole_wire_check()` in fig8; linear-ramp check still to move
  into fig9 (pending).
- **fig6 rerun** with gauge-correct SOC: dresselhaus wedge E₂ = 2.78 µeV (was 2.46,
  inside the gauge artifact). fig8 rerun: refined corner of the claimed box =
  **19.3 µeV < 20** → "100% of box clears 20 µeV" must become "~97–99% (corner 19.3)".

### Vet findings NOT yet addressed (the to-do list)
1. **fig9 rework (CRITICAL):** winding/FF mechanism attribution falsified by
   sign-randomization control (vet 1). Reframe as two channels: per-step near-π
   Josephson junctions (dominant for staircases; bound state ~4 µeV at Δφ=0.85π;
   cliff-at-first-step, E_v-amplified) + smooth-winding depairing (second-order,
   E_v-protected, q_c ≈ 2Δ/α not 2Δ/ħv_F). Add scenarios: poisson+same-sign (vicinal),
   sign-randomized control, single-step junction scan; move linear-ramp into code;
   ≥16 seeds; wire-orientation cosine factor for the miscut conversion.
2. **F8/Si:B repositioning (CRITICAL):** measured Si:B critical fields are 0.1–0.4 T,
   orbital-limited — the Pauli-limited B_P=1.11 T scenario is a hypothesis, not data.
   "Self-tuning" claim does no work (verified: suppression-free optimizer gives same
   numbers — it's Cole-2015 weak coupling). Metallization at full transparency cuts
   center 30.6 → ~20.3. Reframe Si:B as contingent; consider Al-on-Si comparison
   as primary quantitative scenario.
3. **g-orientation constraint (user-requested, was task #22):** Geyer-2024 g-tensors +
   n_so geometry imply accessible g ≈ 1.9–2.3 along the only Majorana-compatible axis
   (B ∥ fin, in-plane); the g=3 corner is out-of-plane (orbitally fatal for film parent).
   Compute the orientation-sphere gap map under three parent scenarios (Si:B measured
   Hc2≈0.4 T / Si:B Pauli-hypothetical / Al).
4. **Luttinger–Kohn module (user-requested, was task #21):** 4-band LK on fin
   cross-section vs k_z + E-field + Zeeman/orbital terms → m*, α(E), g-tensor(E),
   α–g covariation; overlay accessible operating curve on fig8(a). Validation anchors:
   bulk masses m_HH=0.277/m_LH=0.20 (γ1=4.285, γ2=0.339, γ3=1.446, κ≈−0.42),
   Camenzind l_so = 20–60 nm.
5. **Docs overhaul v3:** F3 "everywhere" false (curves cross at α≈0.01); stale 0.17 vs
   0.156; F6 "≤0.1 µeV" vs ledger 0.254; "~14 µeV smooth" mislabeled (true smooth = 22);
   abstract miscut thresholds conflated; "no prior proposal" needs hedging (Maier 2014
   Ge-holes is close precedent); metallization sentence backwards; Nijholt citation
   should be PRB 93, 235434; Chiodi APL Mater. incomplete; arXiv:2505.06040 needs human
   verification; precision claims should quote test tolerances; note hole verdict has
   now had its first adversarial round (this one). Incorporate transport W_half revision.

### Key numbers ledger
All current numbers: output/data/key_numbers.json + output/data/transport_numbers.json.
Full vet reports are in the conversation log (4 agents: F9 physics, F8 physics+lit,
code audit, claims audit) — their substance is summarized in items 1–5 above.
