# Input_Data

Everything the notebooks read lives here. The notebooks never write into this
folder; all results go to `../Output_Data/`.

Before any modelling starts, each notebook calls `checkInputs()`, which stops
with a list of any missing file.

## model_building/

One CSV per property. Read by notebook 00, which writes the curated dataset of
each property to `Output_Data/curated_data/`, and by notebook 01.

| File | Molecules (rows) | Columns |
|:--|--:|:--|
| `Tb_prediction_data_fcl_with_N.csv` | 1030 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `JR Tb /K`, `CRC Tb /K` |
| `Tm_prediction_data_fcl_with_N.csv` | 1939 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `JR Tm /K`, `CRC Tm /K` |
| `Pc_prediction_data_fcl_with_N.csv` | 75 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `JR Pc /Bars`, `CRC Pc /Bars` |
| `Vc_prediction_data_fcl_with_N.csv` | 69 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `JR Vc /(cm3 mol-1)`, `CRC Vc /(cm3 mol-1)` |
| `Tc_prediction_data_fcl_with_N.csv` | 75 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `JR Tc /K`, `CRC Tc /K` |
| `Hvap_prediction_data_fcl_with_N.csv` | 75 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. Wt.`, `JR Hvap /(kJ mol-1)`, `CRC Hvap /(kJ mol-1)` |
| `logP_prediction_data_fcl_with_N.csv` | 3536 | `CAS Reg. No.`, `PubChem SMILES`, `Mol. wt.`, `Log P (JR)`, `logP (CRC)` |

`JR` columns are Joback–Reid group-contribution estimates, except for log P,
where the column holds the Wildman–Crippen estimate from RDKit. `CRC` columns
are experimental values from the CRC Handbook.

## raw/

* `feature_engineered_raw_data.csv`: the cleaned JRGC dataset with RDKit and
  topological descriptors (Wiener, Zagreb M1, Zagreb M2, Randić). Read by
  notebook 01 to choose the baseline molecules and to add the topological
  indices for Tb and Tm.
* `JRGC_Data_cleaned.csv`: the same dataset without the descriptors. It is not
  read by any notebook and is kept for reference only.

## reference_results/

Pooled 10-fold cross-validation predictions of the final models from the
earlier Colab run, one file per property (`<run>_all_cv_predictions.csv`). Read
by notebook 05 only when `predictionSource = "reference"`, for comparison with
a new run. The log P file predates the removal of duplicate structures, so its
numbers differ from a new run.
