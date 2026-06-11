# Feasibility of Majorana Zero Modes in Silicon Nanowires

A quantitative feasibility study of Majorana zero modes (MZMs) in proximitized silicon
nanowires, built on a validated spinful Bogoliubov-de Gennes (Lutchyn-Oreg) model.

**Headline result (post-adversarial-review):** within single-channel proximitized
conduction-band Si wires with g~2, realistic intrinsic SOC, and generic stepped (001)
interfaces, usable protection is not achievable; engineered-SOC variants remain
marginal and face the valley-step constraint unless step morphology is specifically
controlled — quantified across SOC, disorder, and valley channels, including
a new mechanism: each single-atomic interface step is a near-π Josephson junction that
caps the topological gap from the very first step. The silicon *valence* band supports a
conditional design: with Luttinger–Kohn-constrained hole parameters and only *measured*
Si:B critical fields, a thick boron-doped-Si parent with tilted field reaches
**10–19 µeV** static (7.6 µeV under the dynamic self-energy control); Pauli-limited
thin-film Si:B (unmeasured) raises this to ~15 µeV dynamic / 30 µeV static-bare, and an
Al-film parent to 24 µeV dynamic / 34 µeV renormalized-center / up to 55 µeV bare over
orientations. Two measurements would do the most to settle it: thin-film Si:B B_c∥ and
the device-geometry hole g-tensor.
See **[RESULTS.md](RESULTS.md)** (nine findings + corrections log)
and **[PAPER_OUTLINE.md](PAPER_OUTLINE.md)**.

## Repository structure

| file | purpose |
|---|---|
| `majorana_sim.py` | model library: bulk dispersion/topological gap, sparse finite-wire BdG solver, two-valley extension, disorder |
| `run_analysis.py` | reproduces every figure and number (`python run_analysis.py --fig 1..11` or `--fig all`) |
| `RESULTS.md` | findings, verdict, caveats, references |
| `output/fig1..fig9*.png` | generated figures |
| `tests.py` | reproducible checks of the central exact claims (`python tests.py`) |
| `transport.py` | RGF transmission + scattering invariant (standalone) |
| `lk_holes.py` | 4-band Luttinger–Kohn fin model for hole parameters (standalone validation) |
| `transport_valley.py` | dimension-generic RGF for the 8-orbital valley cells (Anderson-insulator analysis) |
| `qp_poisoning.py` | quasiparticle-poisoning temperature bounds |
| `convergence.py` | discretization/length/seed/grid/parent-model convergence studies (Supplement S5) |
| `realism.py` | round-5 controls: dynamic self-energy, Dynes, parent-gap + correlated disorder (Supplement S9) |
| `orbital.py` | round-5 control: Peierls orbital coupling in the 2D strip (Supplement S9) |
| `pairing_mix.py` | round-5 control: inter/intra-valley pairing interpolation (Supplement S9) |
| `morphology.py` | round-5 control: miscut/terrace/bunching/ramp ensembles (Supplement S9) |
| `kp6_holes.py` | six-band Γ8⊕Γ7 LKBP fin model, sine/Galerkin basis (Supplement S10) |
| `kp6_110.py` | [110]-channel rotation of the 6-band model (the experimentally relevant axis) |
| `poisson2d.py` | tri-gate fin Poisson + Schrödinger–Poisson SCF harness |
| `kp6_sc.py` | self-consistent 6-band + Poisson extraction |
| `platform110.py` | hole-platform operating points re-evaluated with [110] 6-band parameters |
| `paper.tex` / `paper.pdf` | manuscript draft v2 |
| `supplement.tex` / `supplement.pdf` | Supplementary Material: proofs, parameter tables, convergence |
| `compat/` | pure-numpy scipy/matplotlib fallback (validated; see compat/scipy docstring) |
| `PAPER_OUTLINE.md` | the pivoted paper: title, abstract draft, figure plan, pre-submission work list |
| `output/data/` | scan checkpoints and `key_numbers.json` |
| `Majorana_Si_Code.ipynb` | **legacy** — the original notebook, kept for the record; its model is not a valid Majorana Hamiltonian (see RESULTS.md §1) and its figures should not be used |

## Quickstart

`./setup_env.sh` does all of the below (Windows commands in its header), or manually:

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python tests.py                 # assertion suite: must end with ALL PASS
python run_analysis.py --fig all
```

Figures land in `output/`. Long scans checkpoint to `output/data/` (with parameter
signatures) and resume if interrupted.

## Reproducing the paper

One command re-runs the full pipeline (test suites, then every analysis script in
dependency order, then the integrity manifest):

```bash
./reproduce.sh          # Linux/macOS
```
```powershell
.\reproduce.ps1         # Windows (uses .\venv if present)
```

All heavy scripts checkpoint per-section to `output/data/*.json` (with parameter
signatures) and resume, so a full pass over the committed checkpoints is mostly
cache reads; delete a script's checkpoint JSON to force genuine recomputation.

`tools/manifest.py` (run last) writes **`MANIFEST.json`** at the repo root:
SHA-256 + size of every ledger file (`output/data/*.json`), SHA-256 of every
figure (`output/fig*.png`, `output/fig13_convergence.pdf`) and manuscript file
(`paper.tex`/`.pdf`, `supplement.tex`/`.pdf`), the git commit + dirty flag, and a
hand-curated map from the headline numbers quoted in the paper to the ledger
entries that generate them. The script **fails (nonzero exit, listing every
mismatch)** if any curated number no longer matches the ledger, so manuscript /
code drift is caught mechanically; `python tools/manifest.py --check` performs
the same verification without rewriting `MANIFEST.json` and runs in CI
(`.github/workflows/manifest.yml`, alongside the test workflow).

**Legacy-notebook quarantine.** `Majorana_Si_Code.ipynb` is kept for the
historical record only: its model is not a valid Majorana Hamiltonian
(RESULTS.md §1), it is executed by no test, script, or CI workflow, and its
outputs appear in **no current figure or quoted number** — its only surviving
images sit in `output/legacy/`, which `MANIFEST.json` deliberately excludes.

## Exact regeneration commands (per figure / claim)

| output | command |
|---|---|
| figs 1–11 + key numbers | `python run_analysis.py --fig N` (N = 1..11, or `all`) |
| fig 8 film-design block | `python -c "import run_analysis as r; r.sib_film_design()"` |
| fig 9 extensions (bi-steps, λ-suppression) | `python -c "import run_analysis as r; r.fig9_extensions()"` |
| fig 12 (transport, invariant, W-scan) | `python transport.py` |
| valley transport / L-scaling | `python transport_valley.py` |
| QP-poisoning bounds | `python qp_poisoning.py` |
| fig 13 + convergence tables | `python convergence.py` (sections checkpoint to `output/data/convergence.json`); matplotlib-free render: `python tools/fig13_pgf.py` |
| round-5 controls | `python realism.py` / `python orbital.py` / `python pairing_mix.py` / `python morphology.py` (each checkpoints to `output/data/<name>.json`) |
| six-band program | `python kp6_holes.py` / `python kp6_110.py` / `python poisson2d.py` / `python kp6_sc.py` / `python platform110.py` (checkpointed likewise) |
| assertion suite | `python tests.py` |

Every quoted number in the paper lives in `output/data/key_numbers.json`, written only
by these scripts (`save_numbers` merges per-tag; nothing is hand-edited).

## Randomness / seed ledger

All stochastic runs use `numpy.random.default_rng` with explicit integer-list seeds:
disorder figs 5/12 family `[fig-specific, W, seed]`; fig 6 family `[61, Ev, seed, L_step]`;
fig 9 family `[71, scenario, spacing_nm, seed]` (convergence reuses `[71, 1, 50, seed]`);
hole-wire check `[81, W, seed]`. Re-running any script bit-reproduces the published
medians; bootstrap CIs in `convergence.py` are seeded with `default_rng(7)`.

## Environment

Developed and run with Python 3.10–3.12, `numpy >= 1.26`, `scipy >= 1.11`,
`matplotlib >= 3.8` (pinned working set in `requirements.txt`). If scipy is
unavailable, `compat/` provides a validated pure-numpy block-tridiagonal
shift-invert Lanczos (`python compat/test_shim.py` checks it against dense
`numpy.linalg.eigh` to ~1e-12 relative) — sufficient for `convergence.py` and
`tests.py`; production figures should use real scipy/matplotlib.

## Archival checklist (Zenodo deposit at submission)

1. Choose a license (required by Zenodo; not yet chosen — see below).
2. Tag the submission commit; record the hash in the paper's Data-availability section
   (public repo: https://github.com/wtrevena/feasibility-of-creating-majorana-modes-in-silicon-nanowires).
3. `git archive` the tag; deposit with `CITATION.cff` metadata; mint DOI.
4. Insert the DOI in `paper.tex` (placeholder marked `[Zenodo DOI: ...]`).

## Model

Single-band semiconductor nanowire with Rashba SOC, Zeeman field, and proximity-induced
s-wave pairing:

```
h(k)  = (hbar^2 k^2 / 2m* - mu) sigma_0 + alpha k sigma_z + E_Z sigma_x
H_BdG = [[h, Delta i sigma_y], [h.c., -h*]]
```

Particle-hole symmetry is exact by construction; the topological phase obeys
E_Z > sqrt(Delta^2 + mu^2). Validation against four known results (gap closing point,
end localization, exponential length-splitting, bulk-gap agreement) is Figure 1.

## License

No license specified yet — add one before sharing or publishing.
