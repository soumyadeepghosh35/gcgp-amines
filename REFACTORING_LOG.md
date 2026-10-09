# Refactoring log

This file records how the original Colab notebooks (`Notebooks/codes/` in the
project folder) map to this repository, what was changed, and which method
questions came up along the way. The modelling logic was kept as it was, so the
published numbers stay traceable; method issues are listed for a later decision
rather than fixed silently.

## 1. Notebook mapping

| Original notebook | New notebook | What changed |
|:--|:--|:--|
| `train_test_JRGC_data_SG_wFtrEngg.ipynb` | `01_JRGC_Baseline` | Same baseline (linear regression on MW + GC, plus four topological indices for Tb and Tm). Added the raw GC error on the same split. Removed the per-call correlation heatmaps, which overwrote each other. |
| `train_test_JRGC_data_SG.ipynb` | merged into `01_JRGC_Baseline` | Same code without the topological indices; superseded by the version above. Pc, Vc, Tc, ΔHvap and log P results are identical in both. |
| `Model1_…` to `Model11_…` | `02_Model01_…` to `02_Model11_…` | Each notebook is now a short run plan (properties and GP settings) that calls `gcgp.singleSplit`. Duplicate runs and runs that were overwritten in the original are removed (see the notes cell in each notebook). |
| `Model12_…_RBF_with_explain.ipynb` | `02_Model12_Method4_AnisotropicMatern_TrainedNoise` | The original file was a copy of Model 5. Rebuilt from discussion slide 19 and checked against the saved `model_4-opt8` results (see section 8). |
| `Hyperparameter_Tuning.ipynb` | `03_Kernel_Comparison` | Reads the metrics written by the `02_Model*` notebooks. Option to compare all-data or outlier-removed runs. |
| `GaussianProcess_wOptimizedModels.ipynb` | `04_Final_GCGP_CrossValidation` | Final settings moved to `gcgp/finalConfig.py`. The base CV function and the Tm variant are now one function with a `skipFailedFolds` switch. Code cells that had been turned into markdown to disable them are replaced by a `runAllVersions` switch. |
| `plot_95CI.ipynb` | `05_Uncertainty_Coverage` | Same coverage calculation and broken-axis figure. Added Wilson confidence intervals for test coverage and an optional continuous-axis figure. |
| `EnsembleModel_wOptimizedModels.ipynb` | `06_ML_Baseline_Comparison` | Same five models on the same folds. Renamed because no ensemble is built. Results are written once per run instead of being appended to an existing file. |

## 2. Shared code (`src/gcgp/`)

| Module | Contents | Replaces |
|:--|:--|:--|
| `settings.py` | Relative paths, `setGlobalSeed`, fast-run switch | hard-coded Google Drive paths |
| `properties.py` | Column names, labels and loader for the seven properties | per-notebook `pd.read_csv` blocks |
| `gpModels.py` | `gpConfigFromMethod`, `getGpData`, `normalize`, `buildBoundedModel`, `fitOneRestart`, `trainGp`, `gpPredict`, `fitAndPredict` | 15 near-identical copies of the GP helpers |
| `singleSplit.py` | `splitTrainTest`, `runSingleSplit`, `runSingleSplitPlan` | `process_single_property` (two versions per notebook) |
| `crossValidation.py` | `runGpCrossValidation` | `process_single_property_cv` and `process_single_property_cv_Tm` |
| `outliers.py` | `removeTargetOutliersIqr`, `filterByGcError` | `remove_outliers` and eight copies of `filter_outliers_comprehensive` |
| `finalConfig.py` | Final GP settings per property and version | settings scattered across notebook cells |
| `mlBaselines.py` | `getMlModels`, `runModelComparisonCv` | `run_cv_for_model`, `process_property` |
| `jrgcBaseline.py` | `prepareBaselineData`, `fitLinearBaseline`, `runBaseline` | `runCompleteTrainTestAnalysis` and helpers |
| `metrics.py` | `calculateMetrics`, `countOutside95`, `coverageBySet` | per-notebook copies |
| `plotting.py` | Parity, fold, coverage and comparison figures | per-notebook copies |
| `reporting.py` | `loadKernelComparison` | loader in `Hyperparameter_Tuning.ipynb` |

About 57,000 lines across 18 notebooks became about 1,800 lines of shared code
plus 17 short notebooks.

**Removed:** Colab `drive.mount` and `pip install` cells, duplicate imports,
cells that only displayed a DataFrame or its columns, commented-out code,
progress and debugging prints, unused helpers (`set_white_exp_95CI`,
`stratifyvector`, `discrepancy_to_property`, the text-table printers), and an
unfinished, never-run "two-stage Δ-learning / multi-fidelity" cell in the
Model 2 and Model 3 notebooks.

## 3. Changes to outputs (the method is unchanged)

* All-data and outlier-removed runs are written to separate folders
  (`<model>/` and `<model>_noOutlier/`). In the original code the second run
  overwrote the first.
* The nine `.txt` files per single-split run are replaced by one
  `<property>_predictions.csv` (row index, SMILES, set, experimental,
  predicted, 95% half-width, GC value).
* The `Improvement` column in the original `<property>_metrics.csv` files was
  shifted by one row (the R² gap sat in the MAE row, and so on), because the
  values were stored in a different order from the metric names. The new
  `Gap (overfit)` column is matched by name.
* Summary tables are overwritten on each run instead of appended, so they no
  longer collect duplicate rows.
* Baseline files are named by property code (the log P files were previously
  named `JRGC_train_(CRC).csv`).
* MAPE for the log P baseline skips molecules with log P = 0 instead of
  returning infinity, matching the GP code.
* Parity plots now have legends; the CV parity plot labels the two colours as
  in-fold and out-of-fold predictions.

## 4. Verification

* A fresh-kernel, top-to-bottom run of all 17 notebooks completes without
  errors in fast mode (`GCGP_FAST_RUN=1`).
* Full-setting spot checks match the original saved results to all printed
  digits: Hvap model 2 (outliers removed) test RMSE 1.917; Hvap model 4 test
  RMSE 6.065; Pc model 4 test RMSE 4.495; Tc final CV test RMSE 32.18 K with
  90.0% test coverage; Hvap final CV coverage 99.52% train and 92.86% test;
  baseline test RMSE for Tb 45.19 K, Tm 64.57 K, Pc 6.48 bar, Vc 11.51
  cm³/mol, Tc 63.53 K, ΔHvap 7.69 kJ/mol and log P 0.713.

## 5. Method issues found during the refactor

These are kept as they were, so the current numbers can be traced. Each needs a
decision before submission. Items marked † overlap with `manuscript_gaps.md`.

1. **The "JRGC" baseline is a fitted linear model, not the raw GC estimate.**
   On the same test molecules the raw GC estimate has a lower RMSE than the
   fitted baseline for Pc, Vc and Tc (Tc: 39.2 K raw versus 63.5 K fitted).
   The fitted baseline can also produce a negative Pc, which explains the point
   in Fig. 2e (†M15). Against the raw GC estimate, the GP gain for Tc is much
   smaller than the paper suggests. *Resolved (revision 3): the fitted linear
   model is the reference; the Methods section should describe it as such.*
2. **Fig. 4 mixes two kinds of runs** (†M7). Because of the overwrite above,
   the saved `model_4`, `model_4-opt6` and `model_4-opt8` results are on all
   data, while the other models are mostly on outlier-removed data. The
   `model_3` folders for Tb and Tc hold Matérn 3/2 results that the Model 3
   notebook cannot produce. *Resolved (revision 3): each Fig. 4 bar is
   intentionally the better of the two runs by test R²; notebook 03 now applies
   this rule explicitly and reports which run was kept.*
3. **Model 12 has no source notebook.** Fig. 4 reads "model_12" from
   `model_4-opt8` (anisotropic Matérn with a trained noise term, judging by the
   saved hyperparameters), but no notebook writes that folder. Rerunning that
   guess does not match the saved numbers. The file called Model 12 is a copy of
   Model 5 that wrote into `model_4`. *Resolved (revision 3): settings taken
   from discussion slide 19 and confirmed against the saved results.*
4. **Two different outlier rules** (†C6). The kernel study drops experimental
   values outside the 1.5 × IQR fences. The final models first keep a fixed
   range (for Tb 0–600 K, which removes high boilers) and then drop molecules
   whose GC error is outside the IQR fences. The second rule removes the cases
   where the GC estimate is worst, so the "no outlier" metrics describe an
   easier subset.
5. **Mixed versions in the final results.** Tb, Tm, Tc and ΔHvap are reported
   after outlier removal; Pc, Vc and log P on all data. For Tm the two
   versions also use different kernels (Matérn 1/2 versus RQ).
6. **Model selection and evaluation share data** (†M1). Kernels were chosen on
   one 80/20 split of the same molecules that are later used for 10-fold CV.
7. **Tm cross-validation drops non-converged folds**, so Tm metrics come from a
   subset of molecules. Report how many folds were used.
8. **The single-split "stratified" split is not stratified.**
   `iterative_train_test_split` is a multi-label method and is given the two
   continuous inputs as labels. The result is close to a random split. Kept
   for reproducibility.
9. **Method 3 mean.** The fixed mean is the standardized GC column, while the
   target is standardized with its own scaler, so the mean is not exactly y_GC
   in property units.
10. **Pooled training points appear nine times** in the CV parity plots
    (†M1). Consider plotting out-of-fold points only.
11. **Model 10 settings** differ between the all-data runs (Matérn 1/2 for all
    properties) and the outlier-removed runs (Matérn 3/2, Tm and Pc only).
    *Resolved (revision 3): intended, per discussion slide 19.*
12. **log P uses 5 restarts**, all other properties 10.
13. **`Data_woOutlier/*_df_woOutliers.csv`** were only read by commented-out
    code, and no notebook in the project creates them. They are not used here.
14. **The saved baseline metrics file** (`Output_files/JRGC/JRGC_statistical_metrics.csv`)
    holds the Tb and Tm results without topological indices, while the Fig. 2
    panels use the version with them.

## 6. Author answers

All four questions were answered (revision 3):

1. Model 12 (`model_4-opt8`): settings from discussion slide 19, see section 8.
2. Model 10: the mix of Matérn 1/2 and Matérn 3/2 with BFGS was intended.
3. Baseline: the fitted linear model is the reference for the paper.
4. Fig. 4: each bar is the better of the all-data and outlier-removed runs,
   judged by test R².

## 7. Revision 2: one input folder and one output folder

* `data/` is renamed to `Input_Data/` and holds every file the notebooks read.
  `results/` is replaced by `Output_Data/`, which holds every file they write.
  No code reads from `Output_Data/` except notebooks 03 and 05, which read
  results written earlier by notebooks 02 and 04.
* `gcgp/settings.py` now defines `INPUT_DIR`, `OUTPUT_DIR`, a list of every
  required input file (`REQUIRED_INPUTS`), `checkInputs()` and
  `outputFolder()`. The environment variables are now `GCGP_INPUT_DIR` and
  `GCGP_OUTPUT_DIR` (previously `GCGP_DATA_DIR` and `GCGP_RESULTS_DIR`).
* Each notebook calls `checkInputs()` in its setup cell, so a missing input is
  reported by name before any model is fitted.
* In the shared code, the `resultsDir` argument is renamed `outputDir`.
* `Input_Data/README.md` documents every input file; `Output_Data/README.md`
  documents what each notebook writes. Git ignores the contents of
  `Output_Data/` except its README.
* Not copied into `Input_Data/` because no notebook uses them:
  `Data_Model_Building/Amine_dataset_with_logP.csv` and the whole
  `Data_woOutlier/` folder (see issue 13 above).

## 8. Revision 3: author answers applied

* **Model 12.** `02_Model12_Method4_RBF_copy` is replaced by
  `02_Model12_Method4_AnisotropicMatern_TrainedNoise`, which writes to
  `model_4-opt8`. Its run plan follows slide 19 (Model 11 settings with the
  noise term trained). Matching it to the saved `model_4-opt8` results showed
  two further details, now part of the run plan: the test set is 15% of the
  molecules, and the outlier-removed runs for Tb and ΔHvap use |z| > 2.5
  instead of the IQR rule. Checked test R² (saved / rerun): all data Tb
  0.763 / 0.763, Pc 0.688 / 0.688, Vc 0.981 / 0.981, Tc 0.529 / 0.529, ΔHvap
  0.433 / 0.433, Tm 0.562 / 0.562; outliers removed Tb 0.704 / 0.704, Pc 0.750 / 0.750, Vc
  0.973 / 0.973, Tc 0.524 / 0.524, ΔHvap −0.016 / −0.016. The outlier-removed
  Tm run and both log P runs were not rechecked because of run time. To support this, `runSingleSplit` and the
  run plans accept `testSize`, `outlierMethod` and `outlierThreshold`.
* **Model 10 and Model 11.** Notes updated to match slide 19. Slide 19 lists
  L-BFGS-B for Model 11, but the original notebook used BFGS for Tm and Tc;
  the notebook settings are kept, because they produced the saved results
  (the Model 12 Tc check above also needed BFGS).
* **Fig. 4.** `loadKernelComparison` gains `versionSelection` (`"best"`,
  `"allData"`, `"noOutlier"`) and `bestBy` (`"R2"`, `"MAE"`, `"RMSE"`).
  Notebook 03 defaults to `"best"` by test R², as in the manuscript, and shows
  which run was kept for every bar.
* **Baseline.** Notebook 01 and `jrgcBaseline.py` now state that the fitted
  linear model is the paper's reference; the raw GC rows are kept for context.
* **Environment.** The README creates the virtual environment with
  `python3.11 -m venv --prompt gcgp-amines .venv` and registers a Jupyter
  kernel named `gcgp-amines`; all notebooks use that kernel. `ipykernel` was
  added to `requirements.txt`.

## 9. Revision 4: environment setup and VS Code

* `scripts/setup_env.sh` creates the `gcgp-amines` environment in `.venv/`,
  installs `requirements.txt`, registers the `gcgp-amines` Jupyter kernel and
  checks that all inputs are present. Tested from scratch with Python 3.11.
* `scripts/run_notebooks.sh` runs the notebooks in order on fresh kernels
  (`--fast` for the quick check, optional number prefixes to run a subset) and
  saves executed copies and a log in `Output_Data/executed_notebooks/`.
* `scripts/vscode/` holds the interpreter setting, the recommended Python and
  Jupyter extensions, and tasks for both scripts; `setup_env.sh` copies them
  into `.vscode/` the first time it runs.
* `setup_env.sh` now detects Apple Silicon Macs, skips Intel (x86_64)
  Pythons there (they run TensorFlow under Rosetta 2 and abort with an AVX
  error), and rebuilds an existing `.venv` that was made with one.

## 10. Revision 5: results from the data, checkpoints, 600 dpi figures, failed folds

* **Wording.** Labels and text that referred to the paper or manuscript (for
  example "Linear GC baseline (paper)", `PAPER_VERSION`, `InPaper`, figure
  numbers) were replaced with neutral names: "Linear GC baseline (fitted)",
  `SELECTED_VERSION` and `Selected`. `Input_Data/reference_results/` is now
  described as the earlier Colab run, kept for comparison.
* **Results written from the data.** The fixed interpretive "Summary" cells were
  replaced by code cells that call `gcgp.narrative.describe*`. They turn the
  result tables of the current run into sentences, show them in the notebook and
  save them as `results_summary.md`. A fast run is flagged at the top of each
  summary.
* **Checkpoints** (`gcgp.checkpoints`). Every GP optimizer restart is saved in
  `Output_Data/checkpoints/` under a key built from the data, the GP settings,
  the restart index and the optimizer limits. Writes are atomic, so an
  interrupted write never leaves a broken file. On a rerun, saved restarts are
  loaded and the best one is rebuilt from its stored parameters. Checked: Hvap
  model 4 (test RMSE 6.065) and Pc model 4 (4.495) are identical on the first and
  the resumed run, and the resumed run took 17 s instead of 66 s.
  `GCGP_RESUME=0` or `run_notebooks.sh --fresh` refits everything.
* **Figures.** `savePlot` writes every figure as PNG and PDF at 600 dpi.
* **Numerical failures.** A restart that raises an error or returns a
  non-finite loss counts as not converged. A cross-validation fold is dropped
  from the pooled metrics, with a printed reason, when no restart converged,
  when prediction raises (for example a failed Cholesky decomposition), or when
  the predictions or intervals contain NaN or infinite values. Dropping failed
  folds is now the default for every property (previously only Tm), and the
  summary gains `N_Failed_Folds` and `Dropped_Folds`. If fewer than five folds
  succeed, the property is reported as failed and the run continues. Checked by
  forcing a NaN fold and a Cholesky error in a test run: both folds were
  dropped and the pooled metrics used the remaining fold. The same handling is
  used for the GP and ML models in notebook 06. TensorFlow's own warning
  messages are hidden (`TF_CPP_MIN_LOG_LEVEL=2`).

## 11. Revision 6: notebook 07, molecule-level comparison

`mechanistic_insights.ipynb` (Colab) is replaced by
`07_Mechanistic_Insights.ipynb`, with the shared code in
`gcgp/moleculeComparison.py`. The original notebook is kept, unchanged, as
`Notebooks/codes/mechanistic_insights_colab.ipynb`. No other notebook was changed.

What changed compared with the original notebook:

* **One prediction per molecule.** The original took the top 20 errors from the
  full cross-validation file, in which each molecule appears once as a test
  point and nine times as a training point. The same molecule therefore filled
  several of the 20 places (for example `CCCCC(CC)CN` four times for ΔHvap),
  mostly through in-sample training predictions. Only the out-of-fold test rows
  are used now.
* **Correct molecule matching.** The original attached SMILES to the GP
  predictions by matching experimental values, which pairs the wrong molecules
  whenever two molecules share a value. Matching now uses the `RowIndex` column
  written by notebook 04, and every pair is checked by SMILES. On the full run,
  the GC-GP RMSE recomputed per molecule equals the pooled test RMSE of notebook
  04 for all seven properties.
* **Same molecules for both models.** The original compared the JRGC errors on
  all molecules with the GC-GP errors on the outlier-filtered subset. Both are
  now compared on the molecules used in notebook 04.
* **Direct comparison.** Besides the common outliers, the notebook lists the
  molecules where the two predictions differ most and labels every molecule as
  poorly predicted by both, by JRGC only (corrected by GC-GP), by GC-GP only, or
  neither. It also tabulates chemical motifs (SMARTS) per category.
* **JRGC source.** `jrgcSource = "raw"` (default, as in the original) uses the
  GC estimate from the input file; `"fitted"` uses the fitted linear baseline of
  notebook 01.
* **Figures.** Structure grids are drawn at 2700 × 1800 px and saved as PNG and
  PDF at 600 dpi. In the original, a 1250 × 250 px image was only tagged as
  600 dpi, which made it about 2 × 0.4 inches when printed. The deprecated Morgan
  fingerprint call was replaced with `rdFingerprintGenerator`.
* **Dependency.** `rdkit` was added to `requirements.txt`.

Open point: `topN = 20` follows the original. For Pc, Vc, Tc and ΔHvap (about 70
molecules) this marks almost a third of the molecules as poorly predicted, and
most of them end up in "both" (15 to 19 of 20 in the full run). A fractional
`topN` (for example 0.05) gives a comparable cut across properties.

### 11.1 One row per structure

Notebook 07 now identifies molecules by structure (standard InChIKey from
RDKit), not by the SMILES text. Entries with the same key are merged into one
row before ranking, counting, motif analysis and drawing; values are averaged
and CAS numbers listed. `duplicate_structures.csv` lists every merged entry.
`structureLevel = "connectivity"` also merges stereoisomers. Previously only the
structure grids removed repeats, and only by exact SMILES text.

Full-run data (molecules used in notebook 04):

| Property | Entries | Unique structures | Same compound, two spellings | Isomers with identical SMILES |
|:--|--:|--:|--:|--:|
| Tb | 886 | 883 | 0 | 3 |
| Tm | 1839 | 1829 | 1 | 8 |
| log P | 3463 | 2978 | 485 | 0 |
| Pc, Vc, Tc, ΔHvap | | no repeats | | |

* **Isomers (Tb, Tm).** Pairs such as cis/trans crotononitrile or fumaronitrile
  and maleonitrile have different CAS numbers and experimental values (up to
  64.5 K apart for Tm) but the same stereo-free SMILES, so the GP gets identical
  inputs for both and cannot separate them.
* **Duplicates in log P (issue for notebook 04, not changed).** 485 compounds
  appear twice, once as PubChem-style SMILES and once as aromatic lower-case
  SMILES (four of them as different tautomer spellings), with the same log P.
  In 431 of the 485 pairs the two copies fell in different folds, so one copy
  was in the training set when the other was predicted. The GC-GP MAE on these
  entries is 0.18, compared with 0.41 on the rest, while the JRGC MAE is similar
  for both groups (0.44 and 0.47). The pooled log P RMSE from notebook 04
  (0.593) rises to 0.675 when only structures that occur once are scored. Before
  the log P cross-validation is reported, the duplicates should be removed from
  `logP_prediction_data_fcl_with_N.csv`, or the folds should be grouped by
  structure, and notebook 04 rerun.

## 12. Revision 7: one curated dataset and one model per property

* **New notebook `00_Data_Curation`** with the rules in `src/gcgp/curation.py`.
  `SELECTED_VERSION` sets the rule per property (`noOutlier` for Tb, Tm, Tc and
  ΔHvap; `allData` for Pc, Vc and log P). The `noOutlier` rule is the one used
  for the final models before: accepted experimental range (Tb 0–600 K, Tm
  0–700 K, Tc 0–800 K, ΔHvap 10–70 kJ/mol), then 1.5 × IQR fences on the GC
  error. log P is reduced to one entry per structure (InChIKey). The curated
  datasets, the list of removed molecules with reasons, a summary, a log with
  file hashes and one figure per curated property are written to
  `Output_Data/curated_data/`.
* **Notebooks 02 to 07 read only the curated datasets** (`loadPropertyData`)
  and have one setting per property (`FINAL_SETTINGS` in `finalConfig.py`). All
  version switches, `_noOutlier` folders and outlier wording are gone from these
  notebooks, from `singleSplit`, `reporting`, `narrative` and
  `moleculeComparison`. `removeTargetOutliersIqr` and its detectors were removed
  because nothing uses them any more. Notebook 01 is unchanged and still reads
  the input files.
* **Kernel study (02, 03).** Each of the twelve models runs once per property on
  the curated data; the separate outlier-removal runs, the z-score rule of
  Model 12 and the Matérn 3/2 outlier runs of Model 10 are gone, and notebook 03
  no longer picks the better of two runs. These numbers differ from the earlier
  kernel figure.
* **Run names.** Results of every property are written under the property name
  (`Tb`, `Tm`, `Pc`, `Vc`, `Tc`, `Hvap`, `LogP`). The reference files in
  `Input_Data/reference_results/` were renamed the same way.
* **Consistency check.** For Tb, Tm, Pc, Vc, Tc and ΔHvap the curated data give
  bit-for-bit the same input arrays, in the same order, as the old filter
  inside notebook 04, and the same checkpoint keys, so the final results are
  unchanged and existing checkpoints are reused. A full-setting rerun in the
  cloud reproduces the original Colab predictions (ΔHvap test RMSE 1.937 kJ/mol,
  Tc 32.18 K). log P changes because 485 repeated entries were removed (481 are
  the same compound written twice, with no CAS number on one copy; 4 are
  isomer pairs with different CAS numbers, the same stereo-free SMILES and the
  same log P).
* **Platform note.** With identical data, settings and seeds, the GP optimizer
  can end at a slightly different optimum on different machines, because the
  numerical libraries (TensorFlow and BLAS builds) differ. Test RMSE of the
  final 10-fold CV:

  | Property | Colab (original) | Author's Mac | Cloud Linux |
  |:--|--:|--:|--:|
  | Pc (bar) | 5.482 | 5.550 | 5.520 |
  | Vc (cm³/mol) | 18.117 | 18.116 | 17.923 |
  | Tc (K) | 32.184 | 32.183 | 32.184 |
  | ΔHvap (kJ/mol) | 1.937 | 1.962 | 1.937 |

  This is not caused by the redesign: on one machine the curated data give
  exactly the same computation as before. Report the numbers from one complete
  run on one machine, and state the software versions.

## 13. Revision 8: faster GP fits with identical results

* **Condition number.** `fitOneRestart` no longer computes the kernel
  condition number (a full SVD) before every restart. `trainGp` computes it
  once, for the restart that is kept, from the same starting point
  (`startingValuesNoSeed` reproduces the starting values without touching the
  global random state). For log P one SVD costs about 1.5 optimizer steps
  (7.6 s against 5.1 s per loss-and-gradient evaluation on a 2-core machine),
  so this saves a few percent per restart.
* **Parallel restarts.** `GCGP_N_WORKERS` (default 1) runs the restarts that
  still have to be fitted in a pool of "spawn" worker processes, reused for the
  whole notebook. Each finished restart is checkpointed at once; a failed worker
  falls back to fitting in the notebook process. The best restart is chosen by
  the same rule, so the result does not depend on the number of workers.
  Workers keep TensorFlow's default thread settings: a test with one thread per
  worker changed the log P loss in the 15th digit, so thread limits were not
  used. `scripts/run_notebooks.sh --workers N` and
  `scripts/benchmark_gp_workers.py` were added.
* **Checkpoint reuse.** Notebook 06 fits its GP on the same folds and settings
  as notebook 04, so its restarts come from notebook 04's checkpoints.
* **Verification.** Notebooks 00, 02 (Model 4), 04 and 06 were run in fast mode
  with the old code and with the new code (1 and 2 workers). 102 of the 117 CSV
  files, including every GP prediction, metric and condition number, are
  byte-for-byte identical. The 15 that differ are all random-forest results
  (differences of about 1e-15); they also differ between two runs of the old
  code, because scikit-learn adds the trees' predictions in parallel threads.
* **Run-to-run note.** An earlier run of the old code, made while another heavy
  job shared the machine, differed from a later run of the same code in the
  last digits (up to about 1e-6 in a prediction). The new code reproduces the
  later run exactly. Such differences are far below the reported precision.
