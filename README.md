# GC-GP models for thermophysical properties of nitrogen-containing compounds

Code and data for predicting seven thermophysical properties of
nitrogen-containing organic compounds (enriched in amines) with Gaussian
processes built on group-contribution (GC) estimates. The GC-GP approach
follows Agbodekhe et al., *Mol. Syst. Des. Eng.* 2026, 11, 85–106.

Properties: normal boiling point (Tb), melting point (Tm), critical pressure
(Pc), critical volume (Vc), critical temperature (Tc), enthalpy of vaporization
(ΔHvap) and the octanol–water partition coefficient (log P).

## Repository layout

```
Input_Data/             every file the notebooks read (never written to)
  model_building/       one CSV per property: CAS, SMILES, MW, GC estimate, CRC value
  raw/                  cleaned JRGC dataset with topological indices
  reference_results/    pooled CV predictions from the earlier Colab run (for comparison)
Output_Data/            every table, prediction file and figure the notebooks write
src/gcgp/               shared code used by every notebook
scripts/                environment setup, batch-run scripts and VS Code settings (scripts/vscode/)
notebooks/              the analysis, numbered in run order
```

Inputs and outputs are kept strictly apart: the code reads only from
`Input_Data/` and writes only to `Output_Data/`. `Input_Data/README.md` lists
every input file and which notebook reads it; `Output_Data/README.md` lists what
each notebook writes. Each notebook checks that its inputs exist before it
starts and reports any missing file by name.

## Notebooks

| Notebook | What it does |
|:--|:--|
| `00_Data_Curation` | Builds the curated dataset of each property; the only place where molecules are removed |
| `01_JRGC_Baseline` | Linear GC baseline on an 80/20 split, plus raw GC errors |
| `02_Model01` … `02_Model12` | One GC-GP model form or kernel per notebook, scored on all properties with the same 10-fold cross-validation |
| `03_Kernel_Comparison` | Compares the twelve models on their cross-validated metrics, with the fold-to-fold spread |
| `04_Final_GCGP_CrossValidation` | 10-fold CV of the final model for each property |
| `05_Uncertainty_Coverage` | Share of molecules inside the 95% prediction interval |
| `06_ML_Baseline_Comparison` | GP versus SVR, random forest, XGBoost and a neural network on the same folds |
| `07_Mechanistic_Insights` | Molecule by molecule: where JRGC and GC-GP predictions differ, which molecules both models miss, and which chemical motifs they share |

Each notebook starts with a short summary of its purpose and outputs, and runs
from top to bottom on a fresh kernel. Run them in numerical order:

* `00_Data_Curation` runs first. It applies the curation rules in
  `src/gcgp/curation.py` and writes one dataset per property to
  `Output_Data/curated_data/`. Notebooks 02 to 07 read only these datasets and
  use one model setting per property, so every result of a property refers to
  the same molecules. `01_JRGC_Baseline` reads the input files directly.
* Notebook 03 reads the output of the `02_Model*` notebooks, notebook 05 reads
  the output of notebook 04 (or the reference results in
  `Input_Data/reference_results/`), and notebook 07 reads the output of
  notebook 04 (and of notebook 01 when `jrgcSource = "fitted"`).

### Results are written from the data

Each notebook ends with a "Summary of the results" cell. It builds plain
sentences from the numbers computed in that run (for example "pooled test
R² = 0.95, RMSE = 26.6 K, 94.1% of molecules inside the 95% interval"), shows
them in the notebook and saves them as `results_summary.md` in the matching
`Output_Data/` folder. Nothing in the notebooks quotes fixed numbers, so the
text always matches the dataset that was run. The intended workflow is: run the
notebooks, then use the saved tables, figures and summaries to write the article.

### Figures

Every figure is saved twice, as PNG and as PDF, at 600 dpi, next to the tables
it belongs to in `Output_Data/`.

### Checkpoints and resuming long runs

Each GP optimizer restart is saved in `Output_Data/checkpoints/` as soon as it
finishes. The checkpoint name is built from the training data, the GP settings,
the restart number and the optimizer limits, so a checkpoint is only reused for
exactly the same fit. If a run stops (kernel crash, closed laptop, time limit),
simply run the notebook again: finished restarts load in a fraction of a second
and the run continues where it stopped, giving the same numbers. Changing the
data or a setting automatically leads to a fresh fit.

* `GCGP_RESUME=0` (or `bash scripts/run_notebooks.sh --fresh`) ignores the saved
  checkpoints and refits everything.
* Deleting `Output_Data/checkpoints/` removes them for good.

### Faster GP fits (log P)

The exact GP costs grow with the cube of the number of training molecules, so
log P (about 2,700 per fold) dominates the run time. Three things help without
changing any result:

* **Checkpoints.** Every restart is saved as soon as it finishes, so a long run
  can be stopped and resumed. Notebook 06 uses the same folds and settings as
  notebook 04, so after notebook 04 its GP restarts are read from the
  checkpoints instead of being refitted.
* **Condition number once per fit.** The kernel condition number (a reported
  diagnostic) is computed only for the restart that is kept, instead of for
  every restart. It needs a full SVD, about 1.5 optimizer steps for log P.
* **Parallel restarts.** Set `GCGP_N_WORKERS` (default 1) to fit restarts in
  parallel worker processes, for example `bash scripts/run_notebooks.sh
  --workers 4`. Each worker uses TensorFlow's default thread settings, the same
  as a serial run, so the results are bit-for-bit identical; limiting the
  threads per worker would change the last digits. Whether it is faster
  depends on the number of cores, so measure it once:
  `.venv/bin/python scripts/benchmark_gp_workers.py` fits the same log P restarts
  with 1, 2 and 4 workers, prints the run times and checks that the results are
  identical.

### Numerical failures

When a GP fit fails numerically, for example when TensorFlow reports
"Cholesky decomposition was not successful", the code does not stop. In
cross-validation (notebooks 02, 04 and 06) the failed fold is dropped, a message is
printed, and the pooled metrics use the remaining folds. The dropped folds and
the reason are listed in the fold table and in the `Dropped_Folds` column of the
summary. In the kernel-study notebooks (02) a failed fold is dropped the same way, and a
property with fewer than five usable folds is reported and left out of the summary table.

## Setup

Python 3.10 to 3.12 is required (TensorFlow 2.16 has no wheels for 3.13).

### One-command setup

From the repository folder, run:

```bash
bash scripts/setup_env.sh
```

The script finds Python 3.10–3.12 (on macOS, `brew install python@3.11` if
none is installed), creates the virtual environment in `.venv/` with the
prompt name `gcgp-amines`, installs `requirements.txt`, registers a Jupyter
kernel called `gcgp-amines`, copies the VS Code settings from
`scripts/vscode/` into `.vscode/` (only if that folder does not exist yet), and
checks that every input file is present. Running it again reuses the existing
environment.

The same steps by hand:

```bash
python3.11 -m venv --prompt gcgp-amines .venv
source .venv/bin/activate          # the shell prompt now starts with (gcgp-amines)
pip install -r requirements.txt
python -m ipykernel install --user --name gcgp-amines --display-name "Python 3 (gcgp-amines)"
```

The environment files live in the hidden `.venv/` folder, which git ignores.
To remove the Jupyter kernel later, run `jupyter kernelspec uninstall gcgp-amines`.

### Apple Silicon Macs (M1, M2, M3, M4)

TensorFlow needs a **native arm64 Python** on these Macs. An Intel (x86_64)
Python, for example the Intel version of Anaconda in `~/anaconda3`, runs under
Rosetta 2, which has no AVX instructions, and TensorFlow then stops with
"The TensorFlow library was compiled to use AVX instructions" and
"Abort trap: 6". `setup_env.sh` skips Intel Pythons on these Macs and rebuilds an
existing `.venv` that was made with one. If it finds no arm64 Python, install
one with Homebrew (`brew install python@3.11`, which lands in `/opt/homebrew`)
and run the script again. Use the Apple Silicon build of VS Code as well.

### Running in VS Code

1. Open the `gcgp-amines` folder in VS Code (File > Open Folder).
2. Install the two recommended extensions when prompted (Python and Jupyter).
3. Open the VS Code terminal (Terminal > New Terminal) and run
   `bash scripts/setup_env.sh`. The first install downloads TensorFlow and
   takes a few minutes. Afterwards, reload the window (Command Palette >
   "Developer: Reload Window") so VS Code picks up the new settings.
4. Open a notebook in `notebooks/` and choose the kernel
   **Python 3 (gcgp-amines)** (top right of the notebook). Run the notebooks in
   numerical order with **Run All**.

The installed `.vscode/settings.json` points VS Code at `.venv/bin/python`, so
new terminals open with the environment active, and `.vscode/tasks.json` adds
the tasks **gcgp: run all notebooks (fast check)** and **gcgp: run all
notebooks (full)** under Terminal > Run Task.

### Running all notebooks at once

```bash
bash scripts/run_notebooks.sh --fast     # quick check, a few minutes per notebook
bash scripts/run_notebooks.sh            # full run with the final settings (hours; resumes from checkpoints)
bash scripts/run_notebooks.sh --fresh    # full run, ignoring saved checkpoints
bash scripts/run_notebooks.sh 01 04 05   # only the notebooks starting with 01, 04 and 05
```

The notebooks in `notebooks/` are left unchanged;
executed copies with all outputs are saved in `Output_Data/executed_notebooks/`,
together with a run log.

## Reproducibility

* All paths are relative to the repository. To read inputs from or write
  outputs to another location, set `GCGP_INPUT_DIR` or `GCGP_OUTPUT_DIR`.
* Every notebook calls `setGlobalSeed(42)`. The GP code also re-seeds NumPy and
  TensorFlow before each optimizer restart (seed = 100 × restart index), and all
  data splits use `random_state=42`.
* The refactored code reproduces the original results exactly where this could
  be checked (for example Hvap model 2 and model 4, Pc model 4, and the pooled
  CV coverage for Hvap and Tc).
* Small numerical differences can appear across operating systems and BLAS
  libraries, because the GP hyperparameters come from a numerical optimizer.

### Quick check

`scripts/run_notebooks.sh --fast` sets `GCGP_FAST_RUN=1`. Each property is then
subsampled to 120 molecules, the GP uses two restarts and cross-validation uses
three folds. The numbers from a fast run are **not** final results, and each
results summary says so at the top. Inside
VS Code you can get the same mode by adding `GCGP_FAST_RUN=1` to the
environment before starting the kernel.

## Data notes

* The column `Log P (JR)` holds the Wildman–Crippen estimate from RDKit, not a
  Joback–Reid value. The name is kept to match the original files.
* `Input_Data/raw/JRGC_Data_cleaned.csv` is the cleaned JRGC dataset without
  topological indices. It is not read by the notebooks but is kept for
  reference.

## Citation

Please cite the accompanying article (details to be added on publication) and
Agbodekhe et al. (2026) for the GC-GP method.
