"""Group-contribution baseline ("JRGC" in the figures).

Reproduces ``train_test_JRGC_data_SG_wFtrEngg.ipynb``.

The baseline is a linear regression fitted on the
training split, with inputs ``[molecular weight, GC prediction]``. For Tb and
Tm four topological indices (Wiener, Zagreb M1, Zagreb M2, Randic) are added
as extra inputs. In other words, it is a recalibrated GC model rather than the
raw Joback-Reid (or Crippen) number. For context, :func:`runBaseline` also
reports the metrics of the raw GC column on the same split.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .metrics import calculateMetrics
from .plotting import plotParity
from .properties import PROPERTIES, loadRawPropertyFile, subsampleForFastRun
from .settings import GLOBAL_SEED, RAW_DATA_DIR, ensureDir

DESCRIPTOR_COLS = ["Wiener", "Zagreb_M1", "Zagreb_M2", "Randic"]

# Inputs of the linear baseline for each property.
BASELINE_FEATURES = {
    "Tb": ["Mol. wt."] + DESCRIPTOR_COLS + ["JR Tb /K"],
    "Tm": ["Mol. wt."] + DESCRIPTOR_COLS + ["JR Tm /K"],
    "Pc": ["Mol. wt.", "JR Pc /Bars"],
    "Vc": ["Mol. wt.", "JR Vc /(cm3 mol-1)"],
    "Tc": ["Mol. wt.", "JR Tc /K"],
    "Hvap": ["Mol. Wt.", "JR Hvap /(kJ mol-1)"],
    "logP": ["Mol. wt.", "Log P (JR)"],
}

# The original notebook shuffled these two tables before splitting. The
# shuffle changes which molecules land in the test set, so it is kept.
SHUFFLE_BEFORE_SPLIT = {"Pc", "Hvap"}

TEST_FRACTION = 0.2


def loadDescriptorTable(rawDir=RAW_DATA_DIR):
    """SMILES and topological indices from the cleaned JRGC dataset."""
    table = pd.read_csv(rawDir / "feature_engineered_raw_data.csv")
    return table[["SMILES"] + DESCRIPTOR_COLS]


def prepareBaselineData(code, descriptorTable, seed=GLOBAL_SEED):
    """Property table restricted to molecules present in the JRGC dataset.

    For Tb and Tm the topological indices are merged in by SMILES; rows with a
    missing or infinite index are dropped for Tm, as in the original code.
    """
    info = PROPERTIES[code]
    df = loadRawPropertyFile(code)
    df = df[df["PubChem SMILES"].isin(descriptorTable["SMILES"])].copy()

    if code in ("Tb", "Tm"):
        df = df.merge(descriptorTable, left_on="PubChem SMILES", right_on="SMILES", how="left")
        df = df.drop(columns=["SMILES"])
        if code == "Tm":
            df["Wiener"] = pd.to_numeric(df["Wiener"], errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=BASELINE_FEATURES[code] + [info["targetCol"]])

    if code in SHUFFLE_BEFORE_SPLIT:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return subsampleForFastRun(df.reset_index(drop=True), seed)


def fitLinearBaseline(train, test, featureCols, targetCol):
    """Standardize inputs, fit ordinary least squares, predict train and test.

    Missing inputs are filled with the training mean (only the Tb and Tm
    descriptors can be missing).
    """
    fillValues = train[featureCols].mean()
    XTrain = train[featureCols].fillna(fillValues)
    XTest = test[featureCols].fillna(fillValues)

    scaler = StandardScaler().fit(XTrain)
    model = LinearRegression().fit(scaler.transform(XTrain), train[targetCol])
    coefficients = pd.DataFrame({"Feature": featureCols, "Coefficient (standardized input)": model.coef_})
    coefficients.loc[len(coefficients)] = ["Intercept", model.intercept_]
    return model.predict(scaler.transform(XTrain)), model.predict(scaler.transform(XTest)), coefficients


def runBaseline(code, descriptorTable, outputDir, seed=GLOBAL_SEED, showPlot=True):
    """Fit the linear GC baseline for one property and save its outputs.

    Returns a list of two metric rows: the fitted baseline
    and the raw GC column on the same test molecules.
    """
    info = PROPERTIES[code]
    featureCols = BASELINE_FEATURES[code]
    targetCol = info["targetCol"]
    gcCol = info["featureCols"][1]

    df = prepareBaselineData(code, descriptorTable, seed)
    valid = df[df[targetCol].notna()].copy()
    train, test = train_test_split(valid, test_size=TEST_FRACTION, random_state=seed)

    predTrain, predTest, coefficients = fitLinearBaseline(train, test, featureCols, targetCol)
    outDir = ensureDir(outputDir)
    train.assign(Predicted=predTrain).to_csv(outDir / f"JRGC_train_{code}.csv", index=False)
    test.assign(Predicted=predTest).to_csv(outDir / f"JRGC_test_{code}.csv", index=False)
    coefficients.to_csv(outDir / f"JRGC_coefficients_{code}.csv", index=False)

    title = f"{code} comparison (JRGC" + (" - Selected Features)" if code in ("Tb", "Tm") else ")")
    trainStats, testStats = plotParity(
        train[targetCol], predTrain, test[targetCol], predTest,
        info["xLabel"], info["yLabel"], title, figsize=(5, 5),
        savePath=outDir / f"parity_plot_{code}_combined.png", show=showPlot)

    rawTrain = calculateMetrics(train[targetCol], train[gcCol])
    rawTest = calculateMetrics(test[targetCol], test[gcCol])
    base = {"Property": code, "Unit": info["unit"], "N_Train": len(train), "N_Test": len(test),
            "N_Features": len(featureCols), "Features": ", ".join(featureCols)}
    rows = []
    for label, trS, teS in [("Linear GC baseline (fitted)", trainStats, testStats),
                            ("Raw GC prediction", rawTrain, rawTest)]:
        row = dict(base, Model=label)
        for m in ["R2", "MAE", "RMSE", "MAPE"]:
            row[f"{m}_Train"] = trS[m]
            row[f"{m}_Test"] = teS[m]
        rows.append(row)
    return rows, coefficients
