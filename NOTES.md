# Working notes (maintained by the AI assistant; updated after every completed item)

Last updated: 2026-06-11, six-band program COMPLETE and pushed
(commit 54d2459, tag v3.1-kp6; tests ALL PASS; trees clean).

## Current state (committed through this commit)

Paper: draft v3 (paper.tex/pdf, 10pp) + supplement (10pp, S1-S10) — round-5
revision complete and pushed earlier (tag v3.0-round5). Public repo:
https://github.com/wtrevena/feasibility-of-creating-majorana-modes-in-silicon-nanowires
CI (GitHub Actions) runs tests.py + compat/test_shim.py on push. MIT license.
Zenodo DOI: NOT yet minted (needs author account) — placeholder in paper.

## Six-band k.p + Poisson program (IN PROGRESS — this commit adds the code+results)

New modules (all validated, all results in output/data/<name>.json + key_numbers.json tags):
- kp6_holes.py  (tag kp6): 6-band Γ8+Γ7 LKBP fin model, sine/Galerkin basis
  (FD does NOT converge at Δso=44 meV — documented), full validation gauntlet
  passed (bulk, 4-band limit 0.006%, Kramers, hermiticity, grid <3%).
  FINDING: split-off coupling moves g_x DOWN (0.62 at production; bracket
  0.04-0.93 vs measured 1.9-2.3) — split-off is NOT the resolution.
- kp6_110.py  (tag kp6_110): channel rotated to [110] (all benchmark devices
  are [110]!). FINDING: g_x' = 1.4-1.7 across all geometries/fields — closes
  ~60% of the gap; discrepancy was largely an AXES MISMATCH in our comparison.
  Side effects favorable: m* ~0.20 (was 0.85), α up to 0.075 eV·Å,
  g_x' nearly Ez-independent (the harmful α-g_x covariation largely lifts).
- poisson2d.py (tag poisson2d): tri-gate fin Poisson + SCF harness, validated
  (plate/two-layer/line-charge/convergence). Calibration: bare slope
  ~8.3 MV/m per V (tri-gate); top-gate ~2.5x stronger at same V_g.
- kp6_sc.py   (tag kp6_sc): full self-consistent 6-band+Poisson. FINDING:
  inhomogeneous electrostatics moves gx -23%..+24% (sign set by gate
  geometry); lateral asymmetry (0.2 V) raises gx+31%, α+56%, rotates SOC
  axis toward -z. Does NOT close the [100] gap alone.
- drafts/kp6_benchmarks.md: literature sheet. Key: all benchmark devices are
  [110]; literature ranks orientation > electrostatics/strain > split-off;
  "Lawaetz 1971" citation for our γ set is shaky — cite Hensel-Feher/Winkler/
  Venitucci instead. Venitucci g_z anchor ~4.66-5 ([100] HH-like) consistent
  with our [100] g_z=4.15-4.8.

## NEXT ITEMS (in order)

1. [x] Platform rerun with [110] 6-band parameters — DONE (platform110.py,
   tag platform110): in-plane g_x'~1.6 usable; SiB_meas dynamic 3.8 ueV
   (below Tier S); Pauli 12.5-13; Al 17-18 dynamic.: _best_gap_hole + tilted
   analysis at (m*=0.19-0.21, α=0.034-0.075, g=(1.43-1.66, 0.4-2.2, 2.1-3.9))
   — [110] numbers look FAVORABLE vs the old empirical-tensor assumptions;
   quantify new operating points for SiB_meas / SiB_pauli / Al (+ dynamic-SE
   correction factor from realism.py tag).
2. [x] Manuscript integration — DONE (supplement S10 + S10.3 table; paper
   LK section rewritten, parents addendum, abstract updated, master-table
   supersession note, geyer/venitucci/hensel bibitems; RESULTS sec 9; README
   module rows; RESPONSE_ROUND5 updated). Was: supplement new section (6-band+Poisson:
   methods, validation, the three findings); paper: revise LK section + the
   g_x-discrepancy story (axes mismatch primary; electrostatics/strain
   remainder), update master table rows 3-4 sources, abstract sentence,
   citation fix (Lawaetz -> Hensel-Feher/Winkler/Venitucci + cite Geyer
   arXiv:2212.02308, Venitucci arXiv:1807.09185, Camenzind arXiv:2103.07369).
   Decide: hole platform stays "hypothesis-generating" but the 6-band [110]
   result UPGRADES the parameter outlook — frame carefully, channel-by-
   channel covariation caveat from kp6_sc.
3. [x] RESULTS.md section 9 etc. — DONE. Was: README table rows for the 4 new modules;
   RESPONSE addendum (reviewer's "six-band k.p + Poisson" demand now MET).
4. [x] Recompile + commit + push + retag — DONE (54d2459, v3.1-kp6).
   Original item: (sandbox /tmp/build; PNGs there current; use
   sync-dir trick for PDF transport), commit, push, retag (v3.1-kp6).
5. [ ] Strain (Bir-Pikus b,d) — flagged optional future work in benchmarks.
6. [ ] Zenodo DOI — author action.

## REMAINING OPEN ITEMS (post six-band program)

- [ ] Zenodo DOI mint from tag (author action; placeholder in paper).
- [ ] Strain (Bir-Pikus b,d) in kp6 — the last unmodeled candidate for the
      residual 0.3-0.6 g_x' shortfall (params in drafts/kp6_benchmarks.md).
- [ ] [110] orientation maps (fig11 rerun with kp6_110 tensors) + 6-band
      dynamic-SE at tilted g_z (platform110 sec B has static only).
- [ ] Triangular/rounded fin cross-sections (real devices are not
      rectangles; literature points at apex localization).
- [ ] Local/nonlocal conductance-map protocol simulation (round-5
      weakness 9, acknowledged as future work in RESPONSE_ROUND5).
- [ ] Master-table regeneration with [110] rows replacing the supersession
      footnote (cosmetic until numbers settle).

## Environment quirks (for resumption in fresh sessions)

- Sandbox mount serves STALE content for files written on the real machine;
  bash-side writes DO propagate host-ward. NEVER edit existing repo files
  with host-side Write/Edit tools — use bash heredoc/python. Git on the
  mount corrupts .git/index — on real machine use Desktop Commander git;
  in sandbox set GIT_INDEX_FILE=/tmp/gitindex if needed.
- Real machine: venv at .\venv (scipy 1.17.1); run compute via Desktop
  Commander start_process, shell powershell.exe, <3 min per call; all heavy
  scripts checkpoint per-section to output/data/*.json and resume.
- Sandbox: no scipy/network; compat/ shim (validated) runs everything;
  pdflatex available IN SANDBOX only — compile in /tmp/build, transport
  PDFs via NEW filenames (output/syncN/), move into place with DC.
- LaTeX builds in /tmp/build need: paper.tex, supplement.tex, drafts/*.tex,
  output/fig{8,9,10,11,12}*.png + output/fig13_convergence.pdf.
