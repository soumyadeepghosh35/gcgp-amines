"""Collect saved results into tables for the summary notebooks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kernelCv import cvFolder

# Folder name used by each model notebook, and the label used in the comparison.
MODEL_FOLDERS = {
    "model_1": "model_1",
    "model_2": "model_2",
    "model_3": "model_3",
    "model_4": "model_4",
    "model_4-opt1": "model_5",
    "model_4-opt2": "model_6",
    "model_4-opt3": "model_7",
    "model_4-opt4": "model_8",
    "model_4-opt5": "model_9",
    "model_4-opt6": "model_10",
    "model_4-opt7": "model_11",
    "model_4-opt8": "model_12",
}

COMPARISON_COLUMNS = ["Property", "Model", "Folder", "R2", "R2_Mean", "R2_Std", "MAE", "RMSE",
                      "CI_Outside", "N_Folds", "N_Molecules"]


def loadKernelComparison(outputDir, propertyCodes, modelFolders=MODEL_FOLDERS):
    """Read the cross-validation summaries written by the Model 1 to Model 12 notebooks.

    Returns ``(table, missing)``. The table has one row per property and
    model. ``R2``, ``MAE`` and ``RMSE`` pool the out-of-fold predictions;
    ``R2_Mean`` and ``R2_Std`` are the mean and spread of the per-fold R².
    ``missing`` lists (property, folder) pairs with no summary file.
    """
    rows, missing = [], []
    for code in propertyCodes:
        for folder, label in modelFolders.items():
            path = cvFolder(outputDir, folder) / code / f"{code}_cv_summary.json"
            if not path.exists():
                missing.append((code, folder))
                continue
            summary = json.loads(path.read_text())
            rows.append({
                "Property": code, "Model": label, "Folder": folder,
                "R2": summary["Overall_Test_R2"], "R2_Mean": summary["Mean_Test_R2"],
                "R2_Std": summary["Std_Test_R2"], "MAE": summary["Overall_Test_MAE"],
                "RMSE": summary["Overall_Test_RMSE"], "CI_Outside": summary["CI_Test_Outside_Frac"],
                "N_Folds": summary["N_Successful_Folds"], "N_Molecules": summary["N_Molecules"],
            })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS), missing


def loadFoldR2(outputDir, propertyCodes, modelFolders=MODEL_FOLDERS):
    """Test R² of every successful fold: one row per property and model, one column per fold."""
    rows = []
    for code in propertyCodes:
        for folder, label in modelFolders.items():
            path = cvFolder(outputDir, folder) / code / f"{code}_fold_metrics.csv"
            if not path.exists():
                continue
            folds = pd.read_csv(path)
            folds = folds[folds["Status"] == "SUCCESS"].set_index("Fold")["Test_R2"]
            rows.append(pd.Series(folds, name=(code, label)))
    table = pd.DataFrame(rows)
    if not table.empty:
        table.index = pd.MultiIndex.from_tuples(table.index, names=["Property", "Model"])
        table.columns = [f"Fold_{int(c)}" for c in table.columns]
    return table


def bestWithinOneStdError(metricsDf):
    """Models whose mean fold R² is within one standard error of the best model of the same property.

    This is the "one-standard-error rule": models inside that band cannot be
    told apart from the best one with ten folds. The standard error is
    ``R2_Std / sqrt(N_Folds)`` of the best model.
    """
    rows = []
    for code, group in metricsDf.groupby("Property", sort=False):
        top = group.loc[group["R2_Mean"].idxmax()]
        threshold = top["R2_Mean"] - top["R2_Std"] / np.sqrt(top["N_Folds"])
        tied = group[group["R2_Mean"] >= threshold]
        rows.append({"Property": code, "Best": top["Model"], "Best_R2_Mean": top["R2_Mean"],
                     "Threshold": threshold, "Indistinguishable": ", ".join(tied["Model"])})
    return pd.DataFrame(rows).set_index("Property")
