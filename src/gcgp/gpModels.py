"""Gaussian process models built on group-contribution (GC) predictions.

Inputs are always ``X = [molecular weight, GC prediction]`` and the target is
the experimental value. Four model forms ("methods") are available; they
follow Agbodekhe et al. (Mol. Syst. Des. Eng., 2026):

* Method 1, ``y = GP(0, K(MW, y_GC))``: the GP learns the property directly
  with a zero mean.
* Method 2, ``y = y_GC + GP(0, K(MW))``: the GP learns the GC error from
  molecular weight only.
* Method 3, ``y = GP(y_GC, K(MW, y_GC))``: the GP learns the property with
  the GC column as a fixed mean (see note below).
* Method 4, ``y = GP(a*MW + b*y_GC + c, K(MW, y_GC))``: the GP learns the
  property with a trainable linear mean on both inputs.

Note on method 3: the mean is the *standardized* GC column, while the target
is standardized with its own scaler. The mean is therefore close to, but not
exactly, y_GC in property units. This matches the original code.

All hyperparameters are bounded with a sigmoid transform between 1e-5 and
100 (RQ alpha up to 5000). Each fit is restarted ``retrainGp`` times from
different starting points and the restart with the lowest negative log
marginal likelihood (among those that converged) is kept.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gpflow
import tensorflow as tf
from sklearn import preprocessing
from tensorflow_probability import bijectors as tfb

from .checkpoints import loadCheckpoint, makeKey, saveCheckpoint
from .settings import getRunSettings, restartWorkers

KERNELS = {
    "RQ": gpflow.kernels.RationalQuadratic,
    "RBF": gpflow.kernels.RBF,
    "Matern12": gpflow.kernels.Matern12,
    "Matern32": gpflow.kernels.Matern32,
    "Matern52": gpflow.kernels.Matern52,
}

METHODS = {
    1: {"meanFunction": "Zero", "name": "y_exp = GP(0, K(x1,x2))", "saveName": "model_1"},
    2: {"meanFunction": "Zero", "name": "y_exp = y_GC + GP(0, K(x1))", "saveName": "model_2"},
    3: {"meanFunction": "Constant", "name": "y_exp = GP(y_GC, K(x1,x2))", "saveName": "model_3"},
    4: {"meanFunction": "Linear", "name": "y_exp = GP(B@X, K(x1,x2))", "saveName": "model_4"},
}

HYPER_LOW = 1e-5
HYPER_HIGH = 100.0
ALPHA_HIGH = 5000.0
INITIAL_NOISE = 1e-5
CI_Z = 1.96


def gpConfigFromMethod(methodNumber, kernel="RBF", anisotropic=False, useWhiteKernel=True,
                       trainLikelihood=True, optMethod="L-BFGS-B", saveName=None):
    """Collect all GP settings in one dictionary.

    ``saveName`` sets the results subfolder; by default it is ``model_<method>``.
    """
    if methodNumber not in METHODS:
        raise ValueError(f"Unknown method number {methodNumber}; use 1, 2, 3 or 4.")
    method = METHODS[methodNumber]
    return {
        "methodNumber": methodNumber,
        "kernel": kernel,
        "anisotropic": anisotropic,
        "useWhiteKernel": useWhiteKernel,
        "trainLikelihood": trainLikelihood,
        "optMethod": optMethod,
        "meanFunction": method["meanFunction"],
        "name": method["name"],
        "saveName": saveName or method["saveName"],
    }


def getGpData(X, y, methodNumber):
    """Turn raw inputs into what the GP is trained on.

    For method 2 the GP sees molecular weight only and learns the GC error
    (``y - y_GC``). For the other methods it sees both inputs and learns
    ``y`` itself. Returns ``(Xgp, ygp, yGc)``, all 2-D.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yGc = X[:, 1].reshape(-1, 1)
    if methodNumber == 2:
        return X[:, 0].reshape(-1, 1), (y - X[:, 1]).reshape(-1, 1), yGc
    return X, y.reshape(-1, 1), yGc


def normalizationFor(methodNumber):
    """Method 2 works in raw units; all other methods standardize X and y."""
    return "None" if methodNumber == 2 else "Standardization"


def normalize(values, scaler=None, method="Standardization", reverse=False):
    """Scale (or unscale) a 2-D array column by column.

    ``method`` is one of ``Standardization``, ``MinMax``, ``LogStand``,
    ``Log+bStand`` or ``None``. When ``scaler`` is None a new scaler is fitted
    on ``values``. Returns ``(scaledValues, scaler)``.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if method == "None":
        return values, None

    def forwardTransform(arr):
        if method in ("Standardization", "MinMax"):
            return arr
        if method == "LogStand":
            return np.log(arr)
        if method == "Log+bStand":
            return np.log(arr + 1e-3)
        raise ValueError(f"Unknown normalization method: {method}")

    if scaler is None:
        base = preprocessing.MinMaxScaler() if method == "MinMax" else preprocessing.StandardScaler()
        scaler = base.fit(forwardTransform(values))

    if reverse:
        out = scaler.inverse_transform(values)
        if method == "LogStand":
            out = np.exp(out)
        elif method == "Log+bStand":
            out = np.exp(out) - 1e-3
        return out, scaler
    return scaler.transform(forwardTransform(values)), scaler


def buildBoundedModel(X, y, gpConfig, lengthscaleInit, lengthscaleInit2, varianceInit):
    """Create an (untrained) gpflow GPR model with bounded hyperparameters."""
    low = tf.cast(HYPER_LOW, tf.float64)
    high = tf.cast(HYPER_HIGH, tf.float64)
    alphaHigh = tf.cast(ALPHA_HIGH, tf.float64)
    ls1 = tf.cast(lengthscaleInit, tf.float64)
    ls2 = tf.cast(lengthscaleInit2, tf.float64)
    var0 = tf.cast(varianceInit, tf.float64)

    if gpConfig["anisotropic"]:
        lengthscales = gpflow.Parameter([ls1, ls2], transform=tfb.Sigmoid(low, high), dtype=tf.float64)
    else:
        lengthscales = gpflow.Parameter(ls1, transform=tfb.Sigmoid(low, high), dtype=tf.float64)
    alpha = gpflow.Parameter(ls1, transform=tfb.Sigmoid(low, alphaHigh), dtype=tf.float64)
    variance = gpflow.Parameter(var0, transform=tfb.Sigmoid(low, high), dtype=tf.float64)

    baseKernel = KERNELS[gpConfig["kernel"]]()
    if gpConfig["kernel"] == "RQ":
        baseKernel.alpha = alpha
    baseKernel.lengthscales = lengthscales
    baseKernel.variance = variance
    kernel = baseKernel + gpflow.kernels.White(variance=1.0) if gpConfig["useWhiteKernel"] else baseKernel

    meanType = gpConfig["meanFunction"]
    if meanType == "Zero":
        meanFunction = None
    elif meanType == "Constant":
        meanFunction = gpflow.functions.Linear(np.array([0, 1]).reshape(-1, 1))
    elif meanType == "Linear":
        meanFunction = gpflow.functions.Linear(np.ones((X.shape[1], 1)))
    else:
        raise ValueError(f"Unknown mean function: {meanType}")

    model = gpflow.models.GPR(data=(X, y), kernel=kernel, mean_function=meanFunction,
                              noise_variance=INITIAL_NOISE)
    if meanType == "Constant":
        gpflow.set_trainable(model.mean_function.A, False)
        gpflow.set_trainable(model.mean_function.b, False)
    gpflow.set_trainable(model.likelihood.variance, gpConfig["trainLikelihood"])
    return model


def startingValues(restartIndex):
    """Starting length scales and kernel variance for one restart.

    Restart 0 starts every hyperparameter at 1. Later restarts draw length
    scales from U(0, 100) and the kernel variance from LogNormal(0, 1), using
    seed ``100 * restartIndex`` so every restart is reproducible.
    """
    seed = int(restartIndex) * 100
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if restartIndex == 0:
        return 1, 1, 1
    ls1 = np.array(np.random.uniform(0, 100))
    ls2 = np.array(np.random.uniform(0, 100))
    var0 = np.array(np.random.lognormal(0, 1.0))
    return ls1, ls2, var0


def startingValuesNoSeed(restartIndex):
    """Same values as :func:`startingValues`, without touching the global seeds.

    ``np.random.seed(s)`` followed by draws from the global generator gives the
    same numbers as a ``RandomState(s)``, so this reproduces the starting point
    of any restart exactly while leaving the global random state alone.
    """
    if restartIndex == 0:
        return 1, 1, 1
    rng = np.random.RandomState(int(restartIndex) * 100)
    ls1 = np.array(rng.uniform(0, 100))
    ls2 = np.array(rng.uniform(0, 100))
    var0 = np.array(rng.lognormal(0, 1.0))
    return ls1, ls2, var0


def initialConditionNumber(X, y, gpConfig, restartIndex):
    """Condition number of the kernel matrix at the starting point of one restart.

    This is a diagnostic only; it is never used to choose between restarts.
    It needs a full SVD of the kernel matrix, which costs several Cholesky
    factorisations, so it is computed once for the chosen restart instead of
    for every restart. The value is identical to computing it before fitting.
    """
    ls1, ls2, var0 = startingValuesNoSeed(restartIndex)
    model = buildBoundedModel(X, y, gpConfig, ls1, ls2, var0)
    return float(np.linalg.cond(model.kernel(X)))


def parameterState(model):
    """Unconstrained values of every model parameter, in a fixed order."""
    return [p.unconstrained_variable.numpy().copy() for p in model.parameters]


def restoreParameterState(model, state):
    """Put saved unconstrained values back into a freshly built model."""
    for parameter, value in zip(model.parameters, state):
        parameter.unconstrained_variable.assign(value)


def fitOneRestart(X, y, gpConfig, restartIndex):
    """Fit one GP from one starting point.

    Numerical failures are caught instead of stopping the run. A common one is
    a failed Cholesky decomposition (TensorFlow then warns "Cholesky
    decomposition was not successful" and fills the matrix with NaNs). Any
    restart that raises an error, does not converge, or ends with a non-finite
    loss is marked as failed and is never chosen as the best fit.

    Returns a small dict that can be saved as a checkpoint. The condition
    number is filled in later, only for the restart that is kept (see
    :func:`initialConditionNumber`).
    """
    ls1, ls2, var0 = startingValues(restartIndex)
    record = {"restart": restartIndex, "loss": float("inf"), "success": False,
              "conditionNumber": float("nan"), "state": None, "error": ""}
    try:
        model = buildBoundedModel(X, y, gpConfig, ls1, ls2, var0)
        result = gpflow.optimizers.Scipy().minimize(
            model.training_loss,
            model.trainable_variables,
            options={"maxiter": getRunSettings()["maxIter"]},
            method=gpConfig["optMethod"],
        )
        loss = float(model.training_loss())
        state = parameterState(model)
        finite = np.isfinite(loss) and all(np.all(np.isfinite(v)) for v in state)
        record.update(loss=loss if np.isfinite(loss) else float("inf"),
                      success=bool(result.success) and finite, state=state)
        if not finite:
            record["error"] = "non-finite loss or parameters (numerical failure, e.g. Cholesky)"
        elif not result.success:
            record["error"] = f"optimizer did not converge: {result.message}"
    except Exception as error:  # numerical errors from TensorFlow / SciPy
        record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return record


def restartKey(X, y, gpConfig, restartIndex):
    """Checkpoint key for one restart: depends on the data, settings and restart index."""
    settings = {k: gpConfig[k] for k in ("methodNumber", "kernel", "anisotropic", "useWhiteKernel",
                                         "trainLikelihood", "optMethod", "meanFunction")}
    return makeKey(X=np.asarray(X, dtype=float), y=np.asarray(y, dtype=float), settings=settings,
                   restart=int(restartIndex), maxIter=getRunSettings()["maxIter"],
                   bounds=[HYPER_LOW, HYPER_HIGH, ALPHA_HIGH, INITIAL_NOISE])


_POOL = {"executor": None, "workers": 0}


def _workerInit():
    """Run once in each worker process."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def restartPool(nWorkers):
    """A pool of worker processes, created once and reused for every fit.

    Workers are started with "spawn" (a fresh Python process), which is the
    safe choice with TensorFlow and the default on macOS. Each worker uses
    TensorFlow's default thread settings, the same as a serial run, so a
    restart gives bit-for-bit the same result in a worker as in the notebook.
    """
    if _POOL["executor"] is None or _POOL["workers"] != nWorkers:
        shutdownRestartPool()
        import multiprocessing

        _POOL["executor"] = ProcessPoolExecutor(max_workers=nWorkers, initializer=_workerInit,
                                                mp_context=multiprocessing.get_context("spawn"))
        _POOL["workers"] = nWorkers
    return _POOL["executor"]


def shutdownRestartPool():
    """Stop the worker processes (they are also stopped when Python exits)."""
    if _POOL["executor"] is not None:
        _POOL["executor"].shutdown(wait=True, cancel_futures=True)
    _POOL["executor"], _POOL["workers"] = None, 0


def runRestarts(X, y, gpConfig, restarts):
    """Fit the given restarts, in parallel when ``GCGP_N_WORKERS`` > 1.

    Every finished restart is saved as a checkpoint straight away. If a worker
    process fails (for example because it ran out of memory), its restart is
    refitted in the notebook process instead, so the run never stops.
    """
    records = {}
    nWorkers = min(restartWorkers(), len(restarts))
    if nWorkers > 1:
        try:
            pool = restartPool(nWorkers)
            futures = {pool.submit(fitOneRestart, X, y, gpConfig, r): r for r in restarts}
            for future in as_completed(futures):
                restart = futures[future]
                try:
                    records[restart] = future.result()
                except Exception:
                    continue
                saveCheckpoint("gp_restarts", restartKey(X, y, gpConfig, restart), records[restart])
        except Exception:
            shutdownRestartPool()
    for restart in restarts:
        if restart not in records:
            records[restart] = fitOneRestart(X, y, gpConfig, restart)
            saveCheckpoint("gp_restarts", restartKey(X, y, gpConfig, restart), records[restart])
    return records


def trainGp(X, y, gpConfig, retrainGp):
    """Fit the GP ``retrainGp`` times and keep the best converged fit.

    Each restart is saved as a checkpoint when it finishes and is read back
    on the next run (see :mod:`gcgp.checkpoints`), so an interrupted run
    resumes where it stopped. Restarts that still have to be fitted run in
    parallel when ``GCGP_N_WORKERS`` is above 1. The best restart is chosen
    by the same rule either way (lowest loss, ties broken by restart order),
    so the result does not depend on the number of workers.

    Returns a dict with the model, its log marginal likelihood (LML), whether
    the optimizer converged, the kernel condition number at the start of the
    best restart, the fitted hyperparameters, and the list of restart records.
    ``model`` is None when no restart converged.
    """
    override = getRunSettings()["retrainGp"]
    nRestarts = int(override if override is not None else retrainGp)

    loaded = {r: loadCheckpoint("gp_restarts", restartKey(X, y, gpConfig, r)) for r in range(nRestarts)}
    missing = [r for r, record in loaded.items() if record is None]
    loaded.update(runRestarts(X, y, gpConfig, missing) if missing else {})
    records = [loaded[r] for r in range(nRestarts)]

    good = [r for r in records if r["success"]]
    if not good:
        reasons = sorted({r["error"] for r in records if r["error"]})
        return {"model": None, "loss": float("inf"), "fitSuccess": False, "lml": np.nan,
                "conditionNumber": float("nan"), "hyperparameters": {}, "restarts": records,
                "failureReason": "; ".join(reasons) or "no restart converged"}

    best = min(good, key=lambda r: r["loss"])
    if not np.isfinite(best.get("conditionNumber", float("nan"))):
        best["conditionNumber"] = initialConditionNumber(X, y, gpConfig, best["restart"])
        saveCheckpoint("gp_restarts", restartKey(X, y, gpConfig, best["restart"]), best)
    model = buildBoundedModel(X, y, gpConfig, 1, 1, 1)
    restoreParameterState(model, best["state"])
    return {"model": model, "loss": best["loss"], "fitSuccess": True, "lml": -best["loss"],
            "conditionNumber": best["conditionNumber"],
            "hyperparameters": gpflow.utilities.read_values(model), "restarts": records,
            "failureReason": ""}


def gpPredict(model, X):
    """Predictive mean and variance of the latent GP at ``X`` (numpy arrays)."""
    mean, var = model.predict_f(X)
    return mean.numpy(), var.numpy()


def fitAndPredict(XTrain, yTrain, XTest, yTest, gpConfig, retrainGp):
    """Full train-and-predict step for one train/test split.

    Handles the method-specific inputs, scaling, GP fitting, unscaling and
    (for method 2) adding the GC value back. All returned values are in
    property units, and ``*Ci`` is the 95% half-width (1.96 x std).
    When the fit cannot be used (no restart converged, or the predictions
    contain NaN or clearly negative variances) it returns a dict with
    ``failed=True`` and a ``reason`` instead of raising an error.
    """
    methodNumber = gpConfig["methodNumber"]
    XTr, yTr, gcTr = getGpData(XTrain, yTrain, methodNumber)
    XTe, yTe, gcTe = getGpData(XTest, yTest, methodNumber)

    normMethod = normalizationFor(methodNumber)
    XTrN, scalerX = normalize(XTr, method=normMethod)
    XTeN, _ = normalize(XTe, scaler=scalerX, method=normMethod)
    yTrN, scalerY = normalize(yTr, method=normMethod)

    fit = trainGp(XTrN, yTrN, gpConfig, retrainGp)
    if fit["model"] is None:
        return {"failed": True, "reason": fit["failureReason"], "restarts": fit["restarts"]}

    try:
        predTrN, varTrN = gpPredict(fit["model"], XTrN)
        predTeN, varTeN = gpPredict(fit["model"], XTeN)
    except Exception as error:
        return {"failed": True, "reason": f"prediction failed: {type(error).__name__}",
                "restarts": fit["restarts"]}
    allValues = np.concatenate([predTrN.ravel(), varTrN.ravel(), predTeN.ravel(), varTeN.ravel()])
    if not np.all(np.isfinite(allValues)):
        return {"failed": True, "reason": "NaN or infinite predictions (numerical failure, e.g. Cholesky)",
                "restarts": fit["restarts"]}
    if min(varTrN.min(), varTeN.min()) < -1e-6:
        return {"failed": True, "reason": "negative predictive variance", "restarts": fit["restarts"]}
    # Round-off can give variances like -1e-12; treat them as zero.
    varTrN, varTeN = np.clip(varTrN, 0, None), np.clip(varTeN, 0, None)

    if scalerY is not None:
        predTr, _ = normalize(predTrN, scaler=scalerY, method=normMethod, reverse=True)
        predTe, _ = normalize(predTeN, scaler=scalerY, method=normMethod, reverse=True)
        varTr = (scalerY.scale_ ** 2) * varTrN
        varTe = (scalerY.scale_ ** 2) * varTeN
    else:
        predTr, predTe, varTr, varTe = predTrN, predTeN, varTrN, varTeN

    if methodNumber == 2:
        yTrTrue, yTeTrue = yTr + gcTr, yTe + gcTe
        predTr, predTe = predTr + gcTr, predTe + gcTe
    else:
        yTrTrue, yTeTrue = yTr, yTe

    return {
        "failed": False,
        "reason": "",
        "yTrainTrue": yTrTrue,
        "yTrainPred": predTr,
        "yTrainCi": CI_Z * np.sqrt(varTr),
        "yTestTrue": yTeTrue,
        "yTestPred": predTe,
        "yTestCi": CI_Z * np.sqrt(varTe),
        "yGcTrain": gcTr,
        "yGcTest": gcTe,
        "gpTrainData": (XTr, yTr),
        "gpTestData": (XTe, yTe),
        "lml": fit["lml"],
        "fitSuccess": fit["fitSuccess"],
        "conditionNumber": fit["conditionNumber"],
        "hyperparameters": fit["hyperparameters"],
        "model": fit["model"],
        "restarts": fit["restarts"],
    }


def hyperparameterText(fit):
    """Readable summary of a fitted model, written next to the results."""
    lines = [f"{name}: {np.array2string(np.asarray(value), precision=6)}"
             for name, value in fit["hyperparameters"].items()]
    lines += [
        f"Condition number: {fit['conditionNumber']}",
        f"Fit success: {fit['fitSuccess']}",
        f"Log marginal likelihood: {fit['lml']}",
    ]
    return "\n".join(lines)
