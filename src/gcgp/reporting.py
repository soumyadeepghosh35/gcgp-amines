"""Collect saved results into tables for the summary notebooks."""

from pathlib import Path

import pandas as pd

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


def readTestMetrics(path):
    """Test-set R2, MAE and RMSE from one ``<property>_metrics.csv`` file."""
    table = pd.read_csv(path).set_index("Metric")["Test"]
    return {"R2": table["R2"], "MAE": table["MAE"], "RMSE": table["RMSE"]}


def loadKernelComparison(outputDir, propertyCodes, modelFolders=MODEL_FOLDERS):
    """Read the test-set metrics written by the Model 1 to Model 12 notebooks.

    Returns ``(table, missing)``. The table has one row per property and
    model; ``missing`` lists (property, folder) pairs with no metrics file.
    """
    rows, missing = [], []
    for code in propertyCodes:
        for folder, label in modelFolders.items():
            path = Path(outputDir) / code / folder / f"{code}_metrics.csv"
            if path.exists():
                rows.append(dict(readTestMetrics(path), Property=code, Model=label, Folder=folder))
            else:
                missing.append((code, folder))
    columns = ["Property", "Model", "Folder", "R2", "MAE", "RMSE"]
    return pd.DataFrame(rows, columns=columns), missing
