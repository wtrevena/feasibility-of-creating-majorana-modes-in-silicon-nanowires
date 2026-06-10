# Feasibility of Majorana Zero Modes in Silicon Nanowires

A quantitative feasibility study of Majorana zero modes (MZMs) in proximitized silicon
nanowires, built on a validated spinful Bogoliubov-de Gennes (Lutchyn-Oreg) model.

**Headline result:** silicon *can* host Majorana zero modes — in the valence band.
Conduction-band wires fail for quantified reasons (best intrinsic gap ~1.5 µeV; a g=2
catch-22; and valley-phase *winding* from ≳0.02° substrate miscut acting as
Fulde–Ferrell-like depairing — a new, falsifiable mechanism). Silicon *holes* evade all
three blockers: measured FinFET hole parameters give 20–36 µeV topological gaps across
the entire parameter box, and boron-doped superconducting silicon (Si:B) as the parent
yields an all-silicon, CMOS-compatible stack whose Pauli-limited gap self-tunes below
the catch-22 ceiling. See **[RESULTS.md](RESULTS.md)** (nine findings + corrections log)
and **[PAPER_OUTLINE.md](PAPER_OUTLINE.md)**.

## Repository structure

| file | purpose |
|---|---|
| `majorana_sim.py` | model library: bulk dispersion/topological gap, sparse finite-wire BdG solver, two-valley extension, disorder |
| `run_analysis.py` | reproduces every figure and number (`python run_analysis.py --fig 1..9` or `--fig all`) |
| `RESULTS.md` | findings, verdict, caveats, references |
| `output/fig1..fig9*.png` | generated figures |
| `tests.py` | reproducible checks of the central exact claims (`python tests.py`) |
| `PAPER_OUTLINE.md` | the pivoted paper: title, abstract draft, figure plan, pre-submission work list |
| `output/data/` | scan checkpoints and `key_numbers.json` |
| `Majorana_Si_Code.ipynb` | **legacy** — the original notebook, kept for the record; its model is not a valid Majorana Hamiltonian (see RESULTS.md §1) and its figures should not be used |

## Quickstart

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py --fig all
```

Figures land in `output/`. Long scans (figs 4-5) checkpoint to `output/data/` (with parameter
signatures) and resume if interrupted; everything runs in well under a minute per figure.

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
