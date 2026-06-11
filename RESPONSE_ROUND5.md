# Response to Review Round 5

(Reviewer recommendation: major revision; five acceptance conditions plus ten
itemized weaknesses. Round 5, like round 4, was used to close gaps before
human peer review. All five conditions are addressed; the disposition of
every numbered weakness is below.)

## The five acceptance conditions

**1. Toned-down no-go language — DONE.** Title: "quantified failure" →
"quantified constraints". "fail independently" → "face three independent,
quantified obstructions"; "can never reach" → "cannot reach … within the
single-channel g=2 model"; "destroy protection" → "sharply degrade
protection (single-step cap ~18%)"; "decided by two measurements" →
"would do the most to settle"; "holes are immune by construction" →
explicitly limited to this mechanism with all other imperfections retained.
The hole route is now labeled a "hypothesis-generating design study" in the
abstract.

**2. Master assumption/gap table — DONE.** Table 1 (paper): one row per
platform scenario with parent + demonstration status, g-tensor source,
orientation, B*, assumed Bc, Δp(B*), Δind, α, g, µ, bare and renormalized
gaps; every cell traced to a key_numbers.json tag (LaTeX comments);
undemonstrated items daggered.

**3. At least one major additional control — THREE DONE.**
(a) *Dynamic self-energy* (realism.py): full Σ(ω), validated against the
static limit to 0.07%; Γ-optimal gaps 7.6/14.9/24.2 µeV vs static-bare
11.0/30.6/50.4 — static bare demoted to upper bounds, static-Z validated
directionally (dynamic 24–30% below). Includes the requested Γ sensitivity
scan, Dynes broadening (γ_D/Δp ≲ 1e-3 for 50 mK), and parent-gap disorder.
(b) *Orbital coupling* (orbital.py): Peierls control; <0.05% suppression at
fin widths; orientation maps survive (reviewer weakness #6 resolved in the
favorable direction).
(c) *Pairing-channel mixing* (pairing_mix.py): the requested inter/intra
interpolation; mechanism robust, η=0 ensembles conservative, threshold
η*≈0.24 identified (weakness #3 resolved).
UPDATE: the six-band k·p + Poisson model has since been completed
(Supplement S10; kp6_holes/kp6_110/poisson2d/kp6_sc). Outcome: the split-off
band LOWERS g_x; the 4-band-vs-measured discrepancy is resolved as
predominantly a channel-orientation ([110] vs [100]) effect closing ~60% of
the gap, with electrostatic asymmetry/strain sized for the remainder; the
platform was re-evaluated at the [110] parameters (Supplement S10.3). The
hole platform remains labeled hypothesis-generating (hard-wall, no strain).

**4. Realistic disorder/step morphology — DONE.** morphology.py: gamma
terrace ensembles from miscut angle, orientation scan, finite-width ramps,
correlated VO-amplitude suppression, bunching, with p5/p50/p95 and failure
fractions (yield is the binding constraint). realism.py CD: correlated
µ(x)/α(x)/g(x)/Δ(x) disorder; the iid baseline overstates robustness 4–8×
at matched RMS — stated prominently (abstract + Sec. controls).

**5. Archival reproducibility — DONE except the DOI.** MIT LICENSE;
CITATION.cff updated; requirements.lock (pip freeze); GitHub Actions CI
running tests.py + the shim cross-validation on every push; public repo
cited in the paper; tagged release. Zenodo DOI minting requires the
author's account and is the single remaining manual step (placeholder
retained in the paper).

## The ten weaknesses, briefly

1 (usable undefined) → operational tiers S/N/Q with quantitative thresholds
and a per-platform decision table (paper Sec. controls, Table 2).
2 (electron no-go absolute) → reframed as "highly constrained/noncompetitive
under ordinary interface conditions"; micromagnet mitigation retained.
3 (pairing channel) → control done; conservative direction.
4 (step morphology) → done, incl. rare-event tails and the design rule.
5 (hole model misses g_x) → RESOLVED: 6-band + Poisson + [110] rotation done
(Supplement S10); discrepancy was predominantly an axes mismatch in the
comparison; platform re-evaluated; remains hypothesis-generating (no strain).
6 (orbital effects) → control done; designs survive (<0.05% at fin widths).
7 (parent modeling) → dynamic self-energy + Γ/Dynes/Δ-disorder scans done;
Tc(d) suppression still flagged hypothetical for thin films.
8 (disorder under-modeled) → correlated µ/α/g/Δ ensembles done; smooth
electrostatics identified as the binding constraint. Local/nonlocal
conductance maps remain future work (weakness #9, acknowledged).
9 (transport observables) → partially: existing RGF + invariant + Anderson
warning; full conductance-map protocol simulation listed as future work.
10 (holes-are-immune oversell) → softened as quoted above.
