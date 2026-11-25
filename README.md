# Complementary task representations in hippocampus and prefrontal cortex for generalizing the structure of problems

Code for reproducing all figures in [Samborska, Butler, Walton, Behrens & Akam (2022) *Nature Neuroscience*](https://doi.org/10.1038/s41593-022-01149-8).

## Corrections

The analysis code has been corrected since publication. [CORRECTIONS.md](CORRECTIONS.md) lists each bug, the panels it affects and the results that change.

## Data

Download behavioural and neural data from [figshare](https://doi.org/10.6084/m9.figshare.19773334) and place it in a local folder.

## Setup

```bash
git clone <repo-url>
cd notebooks_paper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.9.

## Running the analyses

### Option 1: Standalone script (recommended)

Runs all analysis steps and saves figures as SVG + PNG:

```bash
python run_all_analyses.py --data-path /path/to/data_share/ --perms 5000 --output-dir figures
```

Use `--perms 5` or `--perms 100` for a quick check. The paper used 5000 permutations. The random seed is reset before
every step (`--seed`, default 0), so each figure is reproduced exactly; `--only fig6c` runs a single step.
With 5000 permutations the two GLM steps (`fig4b`, `ext5a`) hold every permutation of every neuron in memory, more than
60 GB at the peak; with less memory they run from swap, so do not run them in parallel.

### Option 2: Jupyter notebooks

Open the notebooks in Jupyter or VS Code and update the data path in the first cell.

| Notebook | Contents |
|----------|----------|
| `behaviour.ipynb` | Behavioural results — main (Fig 1) and extended data (Ext Fig 1–2) |
| `cells - main figures.ipynb` | Neural results — main figures (Fig 2, 4–6) |
| `cells - extended data.ipynb` | Neural results — extended data (Ext Fig 4–5, 7–9) |

## Repository structure

```
├── CORRECTIONS.md               # Bugs fixed since publication and the results that change
├── run_all_analyses.py          # Standalone script to run all analyses
├── behaviour.ipynb              # Behavioural analysis notebook
├── cells - main figures.ipynb   # Neural main figures notebook
├── cells - extended data.ipynb  # Neural extended data notebook
├── functions/
│   ├── helper_functions.py      # Shared utilities (CPD, EMA, task IDs, animal permutations)
│   ├── RSAs.py                  # Representational dissimilarity matrices
│   ├── data_import.py           # Import training behavioural data (pyControl)
│   ├── data_import_ephys.py     # Import recording behavioural data (pyControl)
│   ├── functions_behaviour.py   # Behavioural analyses for training (Fig 1, Ext Fig 1)
│   ├── functions_behaviour_rec.py # Behavioural analyses during recordings (Ext Fig 2)
│   ├── functions_cells_regress.py # GLM regression / CPD analysis (Fig 4B)
│   ├── functions_cells_rsa.py   # RSA across time (Fig 4C–E)
│   ├── functions_cells_decode.py # SVM decoding across tasks (Fig 4F–G)
│   ├── functions_cells_port.py  # Post hoc port versus role decoding (not in the paper)
│   ├── functions_cells_svd.py   # SVD population pattern analysis (Fig 5)
│   ├── functions_cells_policy.py # Neural correlates of behavioural policy (Fig 6)
│   ├── functions_cells_ext.py   # Extended data neural analyses (Ext Fig 4, 8)
│   └── plot_neurons.py          # Single neuron visualisation (Fig 2)
├── tests/
│   ├── test_validation.py       # Core functions
│   ├── test_behaviour.py        # Behavioural analyses (synthetic sessions)
│   ├── test_decode.py           # Decoder (synthetic sessions)
│   ├── test_port.py             # Port versus role decoding (synthetic sessions)
│   ├── test_neural_synthetic.py # RSA, SVD and policy known-answer tests
│   └── test_fixes.py            # One test per remaining fix in CORRECTIONS.md
├── figures/                     # Output of run_all_analyses.py (PNG; 5,000 permutations, seed 0)
└── requirements.txt
```

## Tests

```bash
pip install pytest
python -m pytest tests
```

The tests use synthetic data only and do not need the data download.

## Statistical tests

Permutation tests shuffle sessions between brain regions (HP vs PFC) to test for regional differences. Two types:
- **Session shuffle**: randomly reassign sessions to regions
- **Animal shuffle**: enumerate all possible splits of 7 animals into HP-sized (3) and PFC-sized (4) groups

The `--perms` argument (`perms` or `n_perms` in the notebooks) controls the number of session-shuffle permutations. Animal-shuffle permutations enumerate all C(7,3) = 35 splits and use exact rank-based P values (the smallest possible P is 1/35 = 0.029).

## Citation

```bibtex
@article{samborska2022complementary,
  title={Complementary task representations in hippocampus and prefrontal cortex for generalizing the structure of problems},
  author={Samborska, Veronika and Butler, James L and Walton, Mark E and Behrens, Timothy EJ and Akam, Thomas},
  journal={Nature Neuroscience},
  volume={25},
  pages={1314--1326},
  year={2022}
}
```
