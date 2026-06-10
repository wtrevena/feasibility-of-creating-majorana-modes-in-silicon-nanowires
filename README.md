# Feasibility of Majorana Zero Modes in Silicon Nanowires

A quantitative feasibility study of Majorana zero modes (MZMs) in proximitized silicon
nanowires, built on a validated spinful Bogoliubov-de Gennes (Lutchyn-Oreg) model.

**Headline result (post-adversarial-review):** conduction-band silicon cannot host
usable Majorana modes — quantified across SOC, disorder, and valley channels, including
a new mechanism: each single-atomic interface step is a near-π Josephson junction that
caps the topological gap from the very first step. The silicon *valence* band supports a
conditional design: with Luttinger–Kohn-constrained hole parameters and only *measured*
Si:B critical fields, a thick boron-doped-Si parent with tilted field reaches
**10–19 µeV**; Pauli-limited thin-film Si:B (unmeasured) would raise this to 30+ µeV,
and an Al-film parent to 34–55 µeV. Two measurements decide it: thin-film Si:B B_c∥ and
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
