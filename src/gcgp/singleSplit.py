"""One 80/20 train/test split per property (used for the kernel comparison).

This follows ``process_single_property`` from the original Model 1 to Model 12
notebooks. Each property is read from its curated dataset and the results of
one model go to ``<outputDir>/<property>/<saveName>/``.
"""

import json
import time

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from .gpModels import fitAndPredict, gpConfigFromMethod, hyperparameterText
from .metrics import calculateMetrics, calculateTrainTestGap, countOutside95, metricsTable, METRIC_NAMES
from .plotting import plotSingleSplitResults
from .properties import PROPERTIES, loadPropertyData
from .settings import GLOBAL_SEED, ensureDir

TEST_FRACTION = 0.2


def splitTrainTest(X, seed=GLOBAL_SEED, testSize=TEST_FRACTION):
    """Row indices for the 80/20 split used in the original notebooks.

    The original code calls scikit-multilearn's ``iterative_train_test_split``
    with the row index (plus two constant columns) as "features" and the two
    input columns as "labels". The inputs are continuous, so this is not a
    true stratified split, but it is kept so the earlier numbers can be
    reproduced. NumPy is re-seeded first because the function uses the global
    NumPy random state.
    """
    nRows = X.shape[0]
    indexBlock = np.column_stack((np.arange(nRows), np.ones((nRows, 2), dtype=int)))
    np.random.seed(seed)
    trainBlock, _, testBlock, _ = iterative_train_test_split(indexBlock, X, test_size=testSize)
    return trainBlock[:, 0].astype(int), testBlock[:, 0].astype(int)


def runSingleSplit(code, gpConfig, retrainGp, outputDir, seed=GLOBAL_SEED, showPlot=True,
                   testSize=TEST_FRACTION):
    """Fit and evaluate one GP on one property with an 80/20 split.

    Returns a one-row summary dict, or None if the GP could not be fitted.
    """
    np.random.seed(seed)
    info = PROPERTIES[code]
    df = loadPropertyData(code)

    X = df[info["featureCols"]].to_numpy(dtype=float)
    y = df[info["targetCol"]].to_numpy(dtype=float).reshape(-1, 1)
    trainIdx, testIdx = splitTrainTest(X, seed, testSize)

    fit = fitAndPredict(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], gpConfig, retrainGp)
    if fit.get("failed"):
        print(f"{code}: the GP fit could not be used ({fit['reason']}), so this run is skipped.")
        return None

    outDir = ensureDir(outputDir / code / gpConfig["saveName"])

    predictions = pd.DataFrame({
        "RowIndex": np.concatenate([trainIdx, testIdx]),
        "SMILES": df["PubChem SMILES"].to_numpy()[np.concatenate([trainIdx, testIdx])],
        "Set": ["Train"] * len(trainIdx) + ["Test"] * len(testIdx),
        "Experimental": np.concatenate([fit["yTrainTrue"], fit["yTestTrue"]]).ravel(),
        "Predicted": np.concatenate([fit["yTrainPred"], fit["yTestPred"]]).ravel(),
        "CI95": np.concatenate([fit["yTrainCi"], fit["yTestCi"]]).ravel(),
        "GC": np.concatenate([fit["yGcTrain"], fit["yGcTest"]]).ravel(),
    })
    predictions.to_csv(outDir / f"{code}_predictions.csv", index=False)
    (outDir / "model_summary.txt").write_text(hyperparameterText(fit))

    trainMetrics = calculateMetrics(fit["yTrainTrue"], fit["yTrainPred"])
    testMetrics = calculateMetrics(fit["yTestTrue"], fit["yTestPred"])
    metricsTable(trainMetrics, testMetrics).to_csv(outDir / f"{code}_metrics.csv", index=False)

    nOutTrain, fracOutTrain = countOutside95(fit["yTrainTrue"], fit["yTrainPred"], fit["yTrainCi"])
    nOutTest, fracOutTest = countOutside95(fit["yTestTrue"], fit["yTestPred"], fit["yTestCi"])

    _, _, combinedMetrics = plotSingleSplitResults(
        fit["yTrainTrue"], fit["yTrainPred"], fit["yTestTrue"], fit["yTestPred"],
        fit["yTrainCi"], fit["yTestCi"], code, savePath=outDir / f"{code}_GP_results.png", show=showPlot)

    gap = calculateTrainTestGap(trainMetrics, testMetrics)
    summary = {
        "Property": code,
        "Model": gpConfig["saveName"],
        "Method": gpConfig["methodNumber"],
        "Kernel": gpConfig["kernel"],
        "Anisotropic": gpConfig["anisotropic"],
        "Optimizer": gpConfig["optMethod"],
        "TrainLikelihood": gpConfig["trainLikelihood"],
        "N_Molecules": len(df),
        "N_Train": len(trainIdx),
        "N_Test": len(testIdx),
        "LML": fit["lml"],
        "Condition_Number": fit["conditionNumber"],
        "Fit_Success": fit["fitSuccess"],
        "CI_Train_Outside": fracOutTrain,
        "CI_Test_Outside": fracOutTest,
    }
    for name in METRIC_NAMES:
        summary[f"Train_{name}"] = trainMetrics[name]
        summary[f"Test_{name}"] = testMetrics[name]
        summary[f"Combined_{name}"] = combinedMetrics[name]
        summary[f"Gap_{name}"] = gap[name]

    with open(outDir / f"{code}_summary.json", "w") as handle:
        json.dump(summary, handle, indent=4, default=str)
    return summary


def runSingleSplitPlan(runPlan, methodNumber, saveName, outputDir, seed=GLOBAL_SEED, showPlot=True):
    """Run every entry of a notebook's run plan and collect the summaries.

    Each entry of ``runPlan`` is a dict with keys ``properties``, ``kernel``,
    ``anisotropic``, ``optMethod`` and ``trainLikelihood``, ``retrainGp`` and
    optionally ``testSize``. Different properties may use different settings,
    as in the original Model 11 and Model 12 notebooks.
    """
    summaries = []
    for step in runPlan:
        gpConfig = gpConfigFromMethod(
            methodNumber, kernel=step["kernel"], anisotropic=step["anisotropic"],
            useWhiteKernel=step.get("useWhiteKernel", True), trainLikelihood=step["trainLikelihood"],
            optMethod=step["optMethod"], saveName=saveName)
        for code in step["properties"]:
            start = time.time()
            print(f"{code}: kernel={step['kernel']}, anisotropic={step['anisotropic']}, "
                  f"optimizer={step['optMethod']}")
            result = runSingleSplit(code, gpConfig, step["retrainGp"], outputDir, seed=seed,
                                    showPlot=showPlot, testSize=step.get("testSize", TEST_FRACTION))
            if result is not None:
                result["Seconds"] = round(time.time() - start, 1)
                summaries.append(result)
    return pd.DataFrame(summaries)


def compactSummary(summaryDf):
    """Key columns of a run-plan summary, rounded for display."""
    columns = ["Property", "Kernel", "N_Train", "N_Test",
               "Train_R2", "Test_R2", "Test_RMSE", "Test_MAE", "CI_Test_Outside"]
    return summaryDf[columns].round(3)
