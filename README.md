# Majorana States in Silicon Nanowire Simulation

This repository contains code to simulate and analyze Majorana states in proximitized silicon nanowires. The simulation implements a Bogoliubov-de Gennes (BdG) Hamiltonian to study the energy spectrum and wavefunctions of Majorana modes.

## Prerequisites

- Python 3.12 or higher
- Virtual environment management

## Installation

1. Create and activate a virtual environment:

On Linux / MacOS:
```bash
python3.12 -m venv venv
source venv/bin/activate
``` 

On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Simulation

The simulation is contained in a Jupyter notebook `Majorana_Si_Code.ipynb`. To run it:

1. Ensure your virtual environment is activated
2. Launch Jupyter:

3. Open `Majorana_Si_Code.ipynb`
4. Run the cells in sequence

## Simulation Details

The notebook simulates:
- Energy spectrum of a proximitized silicon nanowire
- Majorana mode wavefunctions
- Electron and hole component probability densities
- Electron-hole overlap calculations
- Localization length estimation

Key physical parameters can be modified in the notebook:
- Wire length
- Magnetic field strength
- Superconducting gap
- Spin-orbit coupling strength
- Number of discretization points

## Output

The simulation generates several plots:
1. Energy spectrum of the nanowire
2. Spatial distribution of Majorana modes
3. Probability density distributions for electron and hole components

## License

[Add your license information here]




