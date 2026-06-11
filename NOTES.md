# Working notes (maintained by the AI assistant; updated after every completed item)

Last updated: 2026-06-11, after round-6 closeout (draft v4 pushed, tag
v4.0-round6, MANIFEST refresh 32bdf81). Working tree clean; main == origin.

## QUEUED NEXT STEP (user instruction 2026-06-11 ~7:25am PST — NOT yet started)

- [ ] FINAL PRE-PUBLICATION PASS: review feedback file
      `20260611.7.25am_PST_review_comments.txt` (repo root; NOT yet read).
      Per the user: treat as FEEDBACK only — make whatever additional
      revisions I deem appropriate, then wrap up and declare
      ready-to-publish. This is the closing round.
      Plan on pickup: read fully -> triage against everything below (much
      may already be addressed by rounds 4-6) -> judgment call on which
      items genuinely improve the paper (no obligation to do everything;
      this is editorial discretion, not a referee gate) -> implement
      (parallel agents for any compute; me for manuscript) -> final vet ->
      recompile -> commit/push -> tag (suggest v4.1-final or v5.0) ->
      publication-readiness statement to user (remaining author-only items:
      Zenodo DOI; their final read as author).

## CURRENT STATE (all pushed; remote = github.com/wtrevena/feasibility-of-creating-majorana-modes-in-silicon-nanowires)

Manuscript: **draft v4** — paper.pdf 10pp + supplement.pdf 13pp (S1-S12),
both compiled from the same analysis state, version-sync statement in Data
availability tied to MANIFEST.json (65 curated numbers, machine-verified,
CI-checked on every push via tools/manifest.py --check).
Tags: v3.0-round5, v3.1-kp6, v4.0-round6. Tests: ALL PASS. MANIFEST OK
(records content commit 7304c93).

Paper v4 structure (post round-6 restructure):
- Tier-led abstract (~40% shorter): dynamic-SE numbers as central
  estimates, statics labeled upper bounds, oxide-charge constraint and
  yield statistics in headline, "qubit relevance not established".
- Operational tiers (S/N/Q, drafts/usable_criterion.tex) introduced at end
  of Methods, BEFORE all platform discussion.
- Epistemic three-level statement (exact / numerical / extrapolation).
- Six-band [110] k.p + Poisson = PRIMARY hole model; 4-band [100] demoted
  to pedagogical covariation caution; frozen [100]/static master table in
  Supplement S10.3 for provenance.
- Scenario table (drafts/scenario_table.tex, t:scenario) with reviewer
  columns: demonstrated?/unproven/bare/renorm/dynamic/tier/failure
  mode/decisive measurement.
- Discussion: Ge-hole comparison (PRB 109, 035433) + "What would falsify
  this study" paragraph (4 falsifiers).
- Provenance: 6 internal AI red-team review rounds disclosed; NOT human
  peer review (stated).

## COMPLETED WORK LOG (chronological, with ledger tags)

Round 4 (first referee-style report): SOC unit fix (0.022 eV*nm = 0.22
eV*A); supplement created (theorem proof S2, step-junction mapping S3,
parameter tables S4, convergence S5 [tag convergence], parent model S6, QP
S7, LK S8); claims relabeled; provenance standardized; archival package.
Vet round caught Al 34/55 renorm-vs-bare mislabel + QP inputs — fixed.

Round 5 (ten demands): realism.py [tag realism] — dynamic self-energy
(7.61/14.92/24.22 ueV optima; static-Z validated directionally), Dynes,
parent-Delta(x), correlated mu(x) 4-8x harsher than iid; orbital.py
[orbital] — Peierls control, <0.05% at fin widths; pairing_mix.py
[pairing_mix] — eta*=0.242, eta=0 conservative; morphology.py [morphology]
— miscut/terrace/ramps/bunching, yield framing; master table + tiers;
title softened to "quantified constraints"; MIT license, requirements.lock,
CI tests.yml; RESPONSE_ROUND5.md.

Six-band program: kp6_holes.py [kp6] — 6-band Gamma8+Gamma7, sine/Galerkin
basis (FD non-convergent at Dso=44 meV — documented), validation gauntlet
incl. 4-band limit 0.006%; FINDING: split-off LOWERS gx (bracket
0.04-0.93). kp6_110.py [kp6_110] — [110] rotation; FINDING: gx'=1.43-1.66,
~60% of gap closed — axes mismatch was the dominant effect; m*~0.20, alpha
to 0.075. poisson2d.py [poisson2d] — tri-gate Poisson+SCF validated.
kp6_sc.py [kp6_sc] — self-consistent; electrostatics +-25% geometry-signed;
lateral asymmetry raises gx+31%/alpha+56%. platform110.py [platform110] —
[110] operating points; in-plane film designs revive. drafts/
kp6_benchmarks.md — lit anchors; all benchmark devices [110]; gamma-set
provenance Hensel-Feher/Winkler/Venitucci.

Round 6 (this latest): r6_scenarios.py [r6_scenarios] — tilted dynamic
8.4-10.5 ueV SUPERSEDES static 12.5-17.1 (theta*=90deg; dyn = 69% of
matched static; note: platform110 C-values were at static-optimal B, true
optima ~25-40% higher); tuning windows mu +-14-30+, B >=35%, Gamma
factor >=3; Al disorder (iid halving 310-390 ueV; correlated mu binding
at 25-50 ueV RMS). r6_transport.py [r6_transport] — hole RGF validation
passes (E_T = bulk +-2%, invariant flips); 50 mK spectroscopy brackets
gap +-35%, 100 mK unresolvable; failure-mode library (no trivial mimics
at this point; smooth-confinement FALSE NEGATIVE; diagnostic pair =
sign det r + E2; nonlocal conductance NOT implemented — scoped).
r6_morph2.py [r6_morph2] — wafer yields (premium+chi<=10deg: 68%;
unaligned 2-5%); step junction survives lam x ramp grid (ratio
0.18-0.28); CHARGE-TRAP CALIBRATION: one +-e interface trap ~ 14 meV at
channel; good-oxide densities (1e10-1e11/cm2) -> 3-14 meV RMS = ~100x
beyond fatal; studied RMS levels = trap-free wire => OXIDE CHARGE IS THE
BINDING MATERIALS CONSTRAINT (now in abstract). r6_multiband.py
[r6_multiband] — 2nd/3rd subband windows identical to 1st; Gamma(x)
mild (5-14%); dead zones benign; uncovered end = shortened wire.
A5 infra: tools/manifest.py (65 curated numbers; --check in CI),
reproduce.sh/.ps1, .github/workflows/manifest.yml, README bounded
language + Reproducing-the-paper section. Manuscript restructured to v4
(see CURRENT STATE). RESPONSE_ROUND6.md. RESULTS.md sections 1-10.

## OPEN ITEMS (after the queued final pass)

- [ ] Zenodo DOI mint from final tag (AUTHOR ACTION; placeholder in paper).
- [ ] Author's own read-through as the human author before submission.
- Known scoped-out physics (disclosed in paper/responses, optional):
  nonlocal conductance protocols; full LK-BdG multiband wire; dynamic
  Sigma(x,omega); strain (Bir-Pikus) in kp6; [110] orientation maps
  (fig11 rerun with kp6_110 tensors); triangular/rounded fin sections.
- Minor: fig10/fig11 figures still show [100] LK content (paper text
  points to them as the pedagogical case — acceptable, but a fig with
  [110] curves would strengthen S10 if ever needed).

## ENVIRONMENT QUIRKS (mandatory reading for resumption in fresh sessions)

- Sandbox mount serves STALE content for files written on the real
  machine; bash-side writes DO propagate host-ward. NEVER edit existing
  repo files with host-side Write/Edit tools — use bash heredoc/python
  string-replacement. Git on the mount corrupts .git/index — on the real
  machine use Desktop Commander git; in sandbox set
  GIT_INDEX_FILE=/tmp/gitindex if unavoidable.
- Real machine: venv at .\venv (Python 3.12, scipy 1.17.1, matplotlib
  3.10.9); run all compute via Desktop Commander start_process, shell
  powershell.exe, timeout <=200000 ms, keep each call <3 min; ALL heavy
  scripts checkpoint per-section to output/data/<name>.json and resume;
  read ledgers via DC (sandbox copies stale).
- Sandbox: no scipy/network (PyPI blocked by allowlist); compat/ shim
  (validated vs dense eigh ~1e-12) runs the whole test suite; pdflatex
  exists ONLY in sandbox — compile in /tmp/build (needs paper.tex,
  supplement.tex, drafts/*.tex, output/fig{8,9,10,11,12}*.png,
  output/fig13_convergence.pdf), transport PDFs to the repo via NEW
  filenames (output/syncN/), then DC Move-Item into place.
- Commit messages: plain ASCII (powershell quoting); LF/CRLF warnings are
  benign. After every completed item: commit, push, update this file.
