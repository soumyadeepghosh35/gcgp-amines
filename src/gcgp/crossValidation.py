"""K-fold cross-validation of the GC-GP models (final models, Fig. 3 and Fig. 6).

Based on ``process_single_property_cv`` and its Tm variant from
``GaussianProcess_wOptimizedModels.ipynb``.

A fold whose GP cannot be used is dropped and never stops the run. This
covers folds where no optimizer restart converged, where TensorFlow reports a
failed Cholesky decomposition (the predictions then contain NaNs), or where
any other numerical error occurs. The dropped folds and their reasons are
listed in the fold table, and pooled metrics use the remaining folds only, as
long as at least ``minSuccessfulFolds`` succeed. ``skipFailedFolds=False``
restores the original behaviour of skipping the whole property instead.

Pooled "test" metrics are computed from the out-of-fold predictions, so each
molecule appears once as a test point. Pooled "train" metrics stack the
in-fold predictions of all folds, so each molecule appears K-1 times.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .gpModels import fitAndPredict, hyperparameterText
from .metrics import calculateMetrics, countOutside95
from .plotting import plotFoldR2, plotParity
from .settings import GLOBAL_SEED, ensureDir, getRunSettings


def resolveFolds(nFolds):
    """Number of folds after applying the fast-run override."""
    override = getRunSettings()["nFolds"]
    return int(override) if override is not None else int(nFolds)


def runGpCrossValidation(runName, df, featureCols, targetCol, gpConfig, retrainGp, outputDir,
                         xLabel, yLabel, plotTitle, nFolds=10, seed=GLOBAL_SEED,
                         skipFailedFolds=True, minSuccessfulFolds=5, showPlots=True):
    """K-fold CV for one property. Files go to ``<outputDir>/<runName>/``.

    Returns a dict with the summary row, the per-fold table and the pooled
    prediction table, or None if too few folds succeeded.
    """
    np.random.seed(seed)
    nFolds = resolveFolds(nFolds)
    data = df.dropna(subset=featureCols + [targetCol])
    X = data[featureCols].to_numpy(dtype=float)
    y = data[targetCol].to_numpy(dtype=float).reshape(-1, 1)
    outDir = ensureDir(outputDir / runName)

    folds = KFold(n_splits=nFolds, shuffle=True, random_state=seed)
    foldRows, predictionParts, failedFolds = [], [], []

    for foldIndex, (trainIdx, testIdx) in enumerate(folds.split(X), start=1):
        try:
            fit = fitAndPredict(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], gpConfig, retrainGp)
            problem = checkFold(fit)
        except Exception as error:
            fit, problem = None, f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
        if problem:
            if not skipFailedFolds:
                print(f"{runName}, fold {foldIndex}: {problem}. The property is skipped "
                      "(skipFailedFolds=False).")
                return None
            failedFolds.append({"Fold": foldIndex, "Reason": problem})
            print(f"Fold {foldIndex}/{nFolds} dropped from the pooled results: {problem}")
            continue

        trainStats = calculateMetrics(fit["yTrainTrue"], fit["yTrainPred"])
        testStats = calculateMetrics(fit["yTestTrue"], fit["yTestPred"])
        _, outTrain = countOutside95(fit["yTrainTrue"], fit["yTrainPred"], fit["yTrainCi"])
        _, outTest = countOutside95(fit["yTestTrue"], fit["yTestPred"], fit["yTestCi"])
        foldRows.append({
            "Fold": foldIndex, "Status": "SUCCESS", "N_Train": len(trainIdx), "N_Test": len(testIdx),
            "LML": fit["lml"], "Cond_Num": fit["conditionNumber"], "Fit_Success": fit["fitSuccess"],
            "Train_R2": trainStats["R2"], "Train_MAE": trainStats["MAE"],
            "Train_RMSE": trainStats["RMSE"], "Train_MAPE": trainStats["MAPE"],
            "Test_R2": testStats["R2"], "Test_MAE": testStats["MAE"],
            "Test_RMSE": testStats["RMSE"], "Test_MAPE": testStats["MAPE"],
            "CI_Train_Outside_Frac": outTrain, "CI_Test_Outside_Frac": outTest,
        })
        predictionParts.append(foldPredictionTable(fit, data, trainIdx, testIdx, foldIndex, xLabel, yLabel))
        (outDir / f"fold_{foldIndex}_model_summary.txt").write_text(hyperparameterText(fit))
        print(f"Fold {foldIndex}/{nFolds}: train R² = {trainStats['R2']:.3f}, "
              f"test R² = {testStats['R2']:.3f}, test RMSE = {testStats['RMSE']:.2f}")

    if len(foldRows) == 0 or len(foldRows) < min(minSuccessfulFolds, nFolds):
        print(f"{runName}: only {len(foldRows)} of {nFolds} folds succeeded; no results saved.")
        with open(outDir / f"{runName}_cv_FAILED.json", "w") as handle:
            json.dump({"Run": runName, "FailedFolds": failedFolds}, handle, indent=4)
        return None

    foldDf = pd.DataFrame(foldRows)
    for failure in failedFolds:
        foldDf = pd.concat([foldDf, pd.DataFrame([{"Fold": failure["Fold"], "Status": "FAILED",
                                                   "Failure_Reason": failure["Reason"]}])])
    foldDf = foldDf.sort_values("Fold").reset_index(drop=True)
    successDf = foldDf[foldDf["Status"] == "SUCCESS"]

    predictions = pd.concat(predictionParts, ignore_index=True)
    # All training rows first, then all test rows (same row order as the original files).
    predictions = pd.concat([predictions[predictions["Set"] == "Train"],
                             predictions[predictions["Set"] == "Test"]], ignore_index=True)
    trainPart = predictions[predictions["Set"] == "Train"]
    testPart = predictions[predictions["Set"] == "Test"]
    trainOverall = calculateMetrics(trainPart[xLabel], trainPart[yLabel])
    testOverall = calculateMetrics(testPart[xLabel], testPart[yLabel])
    _, outTrainAll = countOutside95(trainPart[xLabel], trainPart[yLabel], trainPart["95% CI"])
    _, outTestAll = countOutside95(testPart[xLabel], testPart[yLabel], testPart["95% CI"])

    foldDf.to_csv(outDir / f"{runName}_fold_metrics.csv", index=False)
    predictions.to_csv(outDir / f"{runName}_all_cv_predictions.csv", index=False)
    statisticsTable(trainOverall, testOverall, outTrainAll, outTestAll, successDf).to_csv(
        outDir / f"{runName}_cv_statistics.csv", index=False)

    nUsed = len(successDf)
    titleSuffix = f" ({nUsed}/{nFolds} folds)" if failedFolds else ""
    plotParity(trainPart[xLabel], trainPart[yLabel], testPart[xLabel], testPart[yLabel],
               xLabel, yLabel, plotTitle + titleSuffix, trainOverall, testOverall,
               savePath=outDir / f"{runName}_GP_CV_parity.png",
               trainLabel="Train (in-fold)", testLabel="Test (out-of-fold)", show=showPlots)
    plotFoldR2(successDf, trainOverall, testOverall, plotTitle,
               savePath=outDir / f"{runName}_GP_CV_fold_r2.png", show=showPlots)

    summary = {
        "Property": runName, "Method": gpConfig["methodNumber"], "Kernel": gpConfig["kernel"],
        "Anisotropic": gpConfig["anisotropic"], "Optimizer": gpConfig["optMethod"],
        "TrainLikelihood": gpConfig["trainLikelihood"], "N_Molecules": len(data),
        "N_Folds": nFolds, "N_Successful_Folds": nUsed, "N_Failed_Folds": len(failedFolds),
        "Dropped_Folds": "; ".join(f"fold {f['Fold']}: {f['Reason']}" for f in failedFolds),
        "Overall_Train_R2": trainOverall["R2"], "Overall_Test_R2": testOverall["R2"],
        "Overfitting_Delta_R2": trainOverall["R2"] - testOverall["R2"],
        "Overall_Test_MAE": testOverall["MAE"], "Overall_Test_RMSE": testOverall["RMSE"],
        "Overall_Test_MAPE": testOverall["MAPE"], "Overall_Test_MPE": testOverall["MPE"],
        "Mean_Test_R2": successDf["Test_R2"].mean(), "Std_Test_R2": successDf["Test_R2"].std(),
        "Mean_Test_MAE": successDf["Test_MAE"].mean(), "Std_Test_MAE": successDf["Test_MAE"].std(),
        "Mean_Test_RMSE": successDf["Test_RMSE"].mean(), "Std_Test_RMSE": successDf["Test_RMSE"].std(),
        "CI_Train_Outside_Frac": outTrainAll, "CI_Test_Outside_Frac": outTestAll,
    }
    with open(outDir / f"{runName}_cv_summary.json", "w") as handle:
        json.dump(summary, handle, indent=4, default=str)
    return {"summary": summary, "folds": foldDf, "predictions": predictions}


def checkFold(fit):
    """Return a short reason if a fold result cannot be used, else an empty string."""
    if fit is None or fit.get("failed"):
        return (fit or {}).get("reason") or "no GP restart converged"
    if not np.isfinite(fit["lml"]):
        return f"invalid log marginal likelihood ({fit['lml']})"
    for name in ("yTrainPred", "yTrainCi", "yTestPred", "yTestCi"):
        if not np.all(np.isfinite(fit[name])):
            return "NaN or infinite predictions (numerical failure, e.g. Cholesky)"
    return ""


def foldPredictionTable(fit, data, trainIdx, testIdx, foldIndex, xLabel, yLabel):
    """Long table of train and test predictions for one fold."""
    smiles = data["PubChem SMILES"].to_numpy()
    parts = []
    for setName, idx, yTrue, yPred, ci in [
        ("Train", trainIdx, fit["yTrainTrue"], fit["yTrainPred"], fit["yTrainCi"]),
        ("Test", testIdx, fit["yTestTrue"], fit["yTestPred"], fit["yTestCi"]),
    ]:
        parts.append(pd.DataFrame({
            xLabel: np.ravel(yTrue), yLabel: np.ravel(yPred), "95% CI": np.ravel(ci),
            "Fold": foldIndex, "Set": setName, "RowIndex": idx, "SMILES": smiles[idx],
        }))
    return pd.concat(parts, ignore_index=True)


def statisticsTable(trainOverall, testOverall, outTrain, outTest, successDf):
    """Pooled train/test metrics plus the mean and spread across folds."""
    return pd.DataFrame({
        "Metric": ["R2", "MAE", "RMSE", "MAPE", "MPE", "Outside_95CI_Frac"],
        "Train_Overall": [trainOverall[m] for m in ["R2", "MAE", "RMSE", "MAPE", "MPE"]] + [outTrain],
        "Test_Overall": [testOverall[m] for m in ["R2", "MAE", "RMSE", "MAPE", "MPE"]] + [outTest],
        "Test_Mean": [successDf[f"Test_{m}"].mean() for m in ["R2", "MAE", "RMSE", "MAPE"]]
                     + [np.nan, successDf["CI_Test_Outside_Frac"].mean()],
        "Test_Std": [successDf[f"Test_{m}"].std() for m in ["R2", "MAE", "RMSE", "MAPE"]]
                    + [np.nan, successDf["CI_Test_Outside_Frac"].std()],
    })
