"""Error metrics and confidence-interval coverage."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_NAMES = ["MAE", "RMSE", "R2", "MAPE", "MPE"]


def calculateMetrics(yTrue, yPred):
    """Return MAE, RMSE, R², MAPE (%) and MPE (%) for two arrays.

    MAPE and MPE skip points where the true value is zero, so log P values of
    exactly 0 do not cause a division by zero.
    """
    yTrue = np.asarray(yTrue, dtype=float).flatten()
    yPred = np.asarray(yPred, dtype=float).flatten()

    mae = mean_absolute_error(yTrue, yPred)
    rmse = np.sqrt(mean_squared_error(yTrue, yPred))
    r2 = r2_score(yTrue, yPred)

    nonZero = yTrue != 0
    if nonZero.sum() > 0:
        relErr = (yTrue[nonZero] - yPred[nonZero]) / yTrue[nonZero]
        mape = np.mean(np.abs(relErr)) * 100
        mpe = np.mean(relErr) * 100
    else:
        mape = np.inf
        mpe = np.inf
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "MPE": mpe}


def calculateTrainTestGap(trainMetrics, testMetrics):
    """Train-test gap for each metric. Positive values point to overfitting.

    For R² the gap is train minus test; for the error metrics it is test minus
    train. (The original notebooks called this column "Improvement".)
    """
    return {
        "R2": trainMetrics["R2"] - testMetrics["R2"],
        "MAE": testMetrics["MAE"] - trainMetrics["MAE"],
        "RMSE": testMetrics["RMSE"] - trainMetrics["RMSE"],
        "MAPE": testMetrics["MAPE"] - trainMetrics["MAPE"],
        "MPE": testMetrics["MPE"] - trainMetrics["MPE"],
    }


def countOutside95(yTrue, yPred, ci95):
    """Count points whose error is larger than their 95% half-width.

    ``ci95`` is the half-width of the interval (1.96 x predictive std).
    Returns ``(number outside, fraction outside)``.
    """
    yTrue = np.asarray(yTrue, dtype=float).flatten()
    yPred = np.asarray(yPred, dtype=float).flatten()
    ci95 = np.asarray(ci95, dtype=float).flatten()
    nOut = int(np.sum(np.abs(yTrue - yPred) > ci95))
    return nOut, nOut / len(yTrue)


def coverageBySet(predictionsDf, trueCol=None, predCol=None, ciCol="95% CI", setCol="Set"):
    """Percent of points inside their 95% interval, split into Train and Test.

    ``predictionsDf`` is a pooled CV prediction table as written by
    :func:`gcgp.crossValidation.runGpCrossValidation`. By default the first
    column is the true value and the second is the prediction.
    """
    trueCol = trueCol or predictionsDf.columns[0]
    predCol = predCol or predictionsDf.columns[1]
    inside = (predictionsDf[trueCol] - predictionsDf[predCol]).abs() <= predictionsDf[ciCol]
    setName = predictionsDf[setCol].str.lower()
    return {
        "Train": 100 * inside[setName == "train"].mean(),
        "Test": 100 * inside[setName == "test"].mean(),
        "nTrain": int((setName == "train").sum()),
        "nTest": int((setName == "test").sum()),
    }


def metricsTable(trainMetrics, testMetrics):
    """Small Train / Test / Gap table for display in a notebook."""
    gap = calculateTrainTestGap(trainMetrics, testMetrics)
    return pd.DataFrame({
        "Metric": METRIC_NAMES,
        "Training": [trainMetrics[m] for m in METRIC_NAMES],
        "Test": [testMetrics[m] for m in METRIC_NAMES],
        "Gap (overfit)": [gap[m] for m in METRIC_NAMES],
    })
