# Response to Adversarial Review Round 6

(AI red-team referee-style report, 2026-06-10 22:35 PST; recommendation:
major revision. All blocking items addressed; disposition below.)

## Blocking items
1. **Version-control problem** — FIXED: paper and supplement both v4, single
   analysis state tied to the tagged commit via MANIFEST.json (65 curated
   paper numbers machine-verified against the ledger in CI).
2. **Abstract overclaims / too long** — REWRITTEN: ~40% shorter, tier-led,
   dynamic self-energy numbers as the headline central estimates, static
   values explicitly labeled upper bounds, "qubit relevance not established".
3. **"Conduction-band silicon is dead" too broad** — README + paper now use
   the bounded wording prescribed by the review.
4. **Hole story internally unstable** — RESTRUCTURED: six-band [110] is the
   primary model in the main text; 4-band [100] demoted to a pedagogical
   covariation caution; frozen [100]/static master table moved to Supplement
   S10.3 for provenance.
5. **Scenario table with reviewer's columns** — DONE (Table t:scenario):
   demonstrated?/unproven requirement/bare/renorm/dynamic/tier/failure
   mode/decisive measurement; tiers now introduced BEFORE the platforms.

## Requested computational experiments (10)
1. Dynamic SE for tilted orientations — DONE (8.4–10.5 µeV; statics retired).
2. Al-specific disorder — DONE (iid halving 310–390 µeV; correlated binding).
3. Hole transport + finite-T conductance — DONE (validation passes; ±35%
   bracketing at 50 mK; nonlocal conductance explicitly scoped as absent).
4. Electrostatics-calibrated correlated disorder — DONE (trap calibration;
   oxide charge identified as THE binding constraint — promoted to abstract).
5. Step morphology → wafer yields — DONE (alignment dominates).
6. Finite step-width × valley-splitting sweeps — DONE (junction survives;
   ratio 0.18–0.28 over the full grid).
7. Multi-subband holes — DONE (2nd/3rd windows identical; effective-model
   caveat stated; LK-BdG multiband left as future work).
8. Self-consistent spatial proximity — DONE in the static limit (Γ(x), dead
   zones, uncovered ends — all mild/benign; dynamic Σ(x,ω) scoped out).
9. Gate-tuning protocol — DONE (tolerance windows per knob per parent).
10. Failure-mode library — DONE (no trivial mimics at this operating point;
    smooth-confinement false negative identified; diagnostic pair stated).

## Presentation fixes
Falsification paragraph added; Ge-hole comparison added (PRB 109, 035433);
epistemic three-level statement in Methods; intro reframed as computational
feasibility and falsification study; "none of them applies to holes"
qualified; self-referential "review round" prose removed from the main text;
hypothetical scenarios daggered in the scenario table; reproduce.sh/ps1 +
MANIFEST + CI manifest check + bounded README language (round-6 A5).

## Honest remainders
Nonlocal conductance protocols; full LK-BdG multiband wire; dynamic Σ(x,ω);
strain in kp6; Zenodo DOI (author action at submission).
