"""10-fold cross-validation of one model of the kernel study (02_Model* notebooks).

Every model of the study is scored with the same cross-validation routine as
the final models (``crossValidation.runGpCrossValidation``). The folds depend
only on the data and the seed, so all twelve models are tested on exactly the
same molecules in each fold and their results can be compared fold by fold.

Files of one model go to ``<outputDir>/cv/<saveName>/<property>/``.
"""

import time
from pathlib import Path

import pandas as pd

from .crossValidation import runGpCrossValidation
from .gpModels import gpConfigFromMethod
from .properties import PROPERTIES, loadPropertyData
from .settings import GLOBAL_SEED, ensureDir

N_FOLDS = 10


def cvFolder(outputDir, saveName):
    """Folder that holds the cross-validation files of one model."""
    return Path(outputDir) / "cv" / saveName


def runCvPlan(runPlan, methodNumber, saveName, outputDir, nFolds=N_FOLDS, seed=GLOBAL_SEED, showPlots=False):
    """Cross-validate every entry of a notebook's run plan and collect the summaries.

    Each entry of ``runPlan`` is a dict with keys ``properties``, ``kernel``,
    ``anisotropic``, ``optMethod``, ``trainLikelihood`` and ``retrainGp``.
    A property whose GP fails in too many folds is reported and left out.
    """
    modelDir = ensureDir(cvFolder(outputDir, saveName))
    summaries = []
    for step in runPlan:
        gpConfig = gpConfigFromMethod(
            methodNumber, kernel=step["kernel"], anisotropic=step["anisotropic"],
            useWhiteKernel=step.get("useWhiteKernel", True), trainLikelihood=step["trainLikelihood"],
            optMethod=step["optMethod"], saveName=saveName)
        for code in step["properties"]:
            info = PROPERTIES[code]
            start = time.time()
            print(f"\n{code}: kernel={step['kernel']}, anisotropic={step['anisotropic']}, "
                  f"optimizer={step['optMethod']}")
            result = runGpCrossValidation(
                code, loadPropertyData(code), info["featureCols"], info["targetCol"], gpConfig,
                step["retrainGp"], modelDir, info["xLabel"], info["yLabel"], plotTitle=f"{code}, {saveName}",
                nFolds=nFolds, seed=seed, showPlots=showPlots)
            if result is None:
                print(f"{code}: too few folds could be fitted, so this run is skipped.")
                continue
            row = dict(result["summary"], Property=code, Model=saveName, Seconds=round(time.time() - start, 1))
            summaries.append(row)
    return pd.DataFrame(summaries)


def compactCvSummary(summaryDf):
    """Key columns of a cross-validation summary, rounded for display."""
    columns = ["Property", "Kernel", "N_Molecules", "N_Successful_Folds", "Overall_Test_R2",
               "Mean_Test_R2", "Std_Test_R2", "Overall_Test_RMSE", "Overall_Test_MAE", "CI_Test_Outside_Frac"]
    return summaryDf[columns].round(3)


def cvTestPredictions(outputDir, saveName, code):
    """Out-of-fold predictions of one model in the format of ``plotTestParityGrid``."""
    info = PROPERTIES[code]
    table = pd.read_csv(cvFolder(outputDir, saveName) / code / f"{code}_all_cv_predictions.csv")
    table = table[table["Set"] == "Test"]
    return pd.DataFrame({
        "Set": table["Set"], "Experimental": table[info["xLabel"]],
        "Predicted": table[info["yLabel"]], "CI95": table["95% CI"],
    })
