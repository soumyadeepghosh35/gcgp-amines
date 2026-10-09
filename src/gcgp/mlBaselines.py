"""Standard machine-learning baselines on the same CV folds as the GP.

Reproduces ``EnsembleModel_wOptimizedModels.ipynb``. Despite the original
name, no ensemble is built: each model (GP, SVR, random forest, XGBoost and a
small neural network) is trained and scored separately on identical folds,
using the same two inputs as the GP. The ML models give no uncertainty, so
their interval columns are zero.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .crossValidation import resolveFolds
from .gpModels import fitAndPredict, getGpData, normalize, normalizationFor
from .metrics import calculateMetrics, countOutside95
from .plotting import plotFoldR2, plotParity
from .settings import GLOBAL_SEED, ensureDir

MODEL_NAMES = ["GP", "SVR", "RandomForest", "XGBoost", "NeuralNetwork"]


def getMlModels(seed=GLOBAL_SEED):
    """Fresh, unfitted copies of the four ML baselines."""
    return {
        "SVR": Pipeline([("scaler", StandardScaler()),
                         ("model", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))]),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.8, random_state=seed, verbosity=0),
        "NeuralNetwork": Pipeline([("scaler", StandardScaler()),
                                   ("model", MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                                          max_iter=2000, random_state=seed,
                                                          early_stopping=True, validation_fraction=0.1))]),
    }


def fitAndPredictMl(modelName, XTrain, yTrain, XTest, yTest, methodNumber):
    """Train one ML baseline on one fold and return predictions in property units.

    The ML models see the same inputs as the GP for the chosen method (for
    method 2: molecular weight only, learning the GC error).
    """
    XTr, yTr, gcTr = getGpData(XTrain, yTrain, methodNumber)
    XTe, yTe, gcTe = getGpData(XTest, yTest, methodNumber)
    normMethod = normalizationFor(methodNumber)
    XTrN, scalerX = normalize(XTr, method=normMethod)
    XTeN, _ = normalize(XTe, scaler=scalerX, method=normMethod)

    estimator = getMlModels()[modelName]
    estimator.fit(XTrN, yTr.ravel())
    predTr = estimator.predict(XTrN).reshape(-1, 1)
    predTe = estimator.predict(XTeN).reshape(-1, 1)

    if methodNumber == 2:
        yTr, yTe, predTr, predTe = yTr + gcTr, yTe + gcTe, predTr + gcTr, predTe + gcTe
    return {"failed": False, "yTrainTrue": yTr, "yTrainPred": predTr, "yTrainCi": np.zeros_like(predTr),
            "yTestTrue": yTe, "yTestPred": predTe, "yTestCi": np.zeros_like(predTe),
            "lml": np.nan, "conditionNumber": np.nan}


def runModelComparisonCv(runName, df, featureCols, targetCol, gpConfig, retrainGp, outputDir,
                         xLabel, yLabel, plotTitle, nFolds=10, seed=GLOBAL_SEED,
                         modelNames=MODEL_NAMES, showPlots=True):
    """Cross-validate every model on the same folds. Returns one summary row per model."""
    np.random.seed(seed)
    nFolds = resolveFolds(nFolds)
    data = df.dropna(subset=featureCols + [targetCol])
    X = data[featureCols].to_numpy(dtype=float)
    y = data[targetCol].to_numpy(dtype=float).reshape(-1, 1)
    splits = list(KFold(n_splits=nFolds, shuffle=True, random_state=seed).split(X))
    methodNumber = gpConfig["methodNumber"]

    summaries, droppedFolds = [], []
    for modelName in modelNames:
        foldRows, predictionParts = [], []
        for foldIndex, (trainIdx, testIdx) in enumerate(splits, start=1):
            try:
                if modelName == "GP":
                    fit = fitAndPredict(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], gpConfig, retrainGp)
                else:
                    fit = fitAndPredictMl(modelName, X[trainIdx], y[trainIdx], X[testIdx], y[testIdx],
                                          methodNumber)
                reason = fit.get("reason", "") if fit.get("failed") else ""
                if not reason and not all(np.all(np.isfinite(fit[k])) for k in ("yTrainPred", "yTestPred")):
                    reason = "NaN or infinite predictions"
            except Exception as error:
                reason = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
            if reason:
                print(f"{runName} {modelName} fold {foldIndex} dropped from the pooled results: {reason}")
                droppedFolds.append({"Model": modelName, "Fold": foldIndex, "Reason": reason})
                continue

            trainStats = calculateMetrics(fit["yTrainTrue"], fit["yTrainPred"])
            testStats = calculateMetrics(fit["yTestTrue"], fit["yTestPred"])
            _, outTrain = countOutside95(fit["yTrainTrue"], fit["yTrainPred"], fit["yTrainCi"])
            _, outTest = countOutside95(fit["yTestTrue"], fit["yTestPred"], fit["yTestCi"])
            foldRows.append({
                "Fold": foldIndex, "Model": modelName, "N_Train": len(trainIdx), "N_Test": len(testIdx),
                "LML": fit["lml"], "Cond_Num": fit["conditionNumber"],
                "Train_R2": trainStats["R2"], "Train_MAE": trainStats["MAE"],
                "Train_RMSE": trainStats["RMSE"], "Train_MAPE": trainStats["MAPE"],
                "Test_R2": testStats["R2"], "Test_MAE": testStats["MAE"],
                "Test_RMSE": testStats["RMSE"], "Test_MAPE": testStats["MAPE"],
                "CI_Train_Outside_Frac": outTrain, "CI_Test_Outside_Frac": outTest,
            })
            for setName, yTrue, yPred, ci in [("Train", fit["yTrainTrue"], fit["yTrainPred"], fit["yTrainCi"]),
                                              ("Test", fit["yTestTrue"], fit["yTestPred"], fit["yTestCi"])]:
                predictionParts.append(pd.DataFrame({xLabel: np.ravel(yTrue), yLabel: np.ravel(yPred),
                                                     "95CI": np.ravel(ci), "Fold": foldIndex, "Set": setName}))

        if not foldRows:
            print(f"{modelName}: no successful folds, skipped.")
            continue

        foldDf = pd.DataFrame(foldRows)
        predictions = pd.concat(predictionParts, ignore_index=True)
        trainPart = predictions[predictions["Set"] == "Train"]
        testPart = predictions[predictions["Set"] == "Test"]
        trainOverall = calculateMetrics(trainPart[xLabel], trainPart[yLabel])
        testOverall = calculateMetrics(testPart[xLabel], testPart[yLabel])
        _, outTrainAll = countOutside95(trainPart[xLabel], trainPart[yLabel], trainPart["95CI"])
        _, outTestAll = countOutside95(testPart[xLabel], testPart[yLabel], testPart["95CI"])

        modelDir = ensureDir(outputDir / runName / modelName)
        foldDf.to_csv(modelDir / f"{runName}_{modelName}_fold_metrics.csv", index=False)
        predictions.to_csv(modelDir / f"{runName}_{modelName}_predictions.csv", index=False)
        plotParity(trainPart[xLabel], trainPart[yLabel], testPart[xLabel], testPart[yLabel],
                   xLabel, yLabel, f"{plotTitle} [{modelName}]", trainOverall, testOverall,
                   savePath=modelDir / f"{runName}_{modelName}_parity.png",
                   trainLabel="Train (in-fold)", testLabel="Test (out-of-fold)", show=showPlots)
        plotFoldR2(foldDf, trainOverall, testOverall, f"{plotTitle} [{modelName}]",
                   savePath=modelDir / f"{runName}_{modelName}_fold_r2.png", show=showPlots)

        summaries.append({
            "Property": runName, "Model": modelName, "N_Folds_Used": len(foldDf),
            "Dropped_Folds": "; ".join(f"fold {d['Fold']}: {d['Reason']}" for d in droppedFolds
                                       if d["Model"] == modelName),
            "Overall_Train_R2": trainOverall["R2"], "Overall_Train_MAE": trainOverall["MAE"],
            "Overall_Train_RMSE": trainOverall["RMSE"], "Overall_Train_MAPE": trainOverall["MAPE"],
            "Overall_Test_R2": testOverall["R2"], "Overall_Test_MAE": testOverall["MAE"],
            "Overall_Test_RMSE": testOverall["RMSE"], "Overall_Test_MAPE": testOverall["MAPE"],
            "Mean_Test_R2": foldDf["Test_R2"].mean(), "Std_Test_R2": foldDf["Test_R2"].std(),
            "Mean_Test_MAE": foldDf["Test_MAE"].mean(), "Std_Test_MAE": foldDf["Test_MAE"].std(),
            "Mean_Test_RMSE": foldDf["Test_RMSE"].mean(), "Std_Test_RMSE": foldDf["Test_RMSE"].std(),
            "Overfitting_dR2": trainOverall["R2"] - testOverall["R2"],
            "CI_Train_Out_Frac": outTrainAll, "CI_Test_Out_Frac": outTestAll,
        })
    return pd.DataFrame(summaries)
