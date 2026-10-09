"""Settings of the final GP model for each property.

There is one setting per property, applied to that property's curated
dataset (see ``00_Data_Curation``). The settings are the final-model settings
from the original ``GaussianProcess_wOptimizedModels.ipynb``.
"""

from .gpModels import gpConfigFromMethod

FINAL_SETTINGS = {
    "Tb": {"method": 4, "kernel": "Matern32", "anisotropic": True, "optMethod": "L-BFGS-B",
           "trainLikelihood": True, "retrainGp": 10},
    "Tm": {"method": 4, "kernel": "RQ", "anisotropic": False, "optMethod": "L-BFGS-B",
           "trainLikelihood": False, "retrainGp": 10},
    "Pc": {"method": 2, "kernel": "RQ", "anisotropic": False, "optMethod": "L-BFGS-B",
           "trainLikelihood": False, "retrainGp": 10},
    "Vc": {"method": 4, "kernel": "Matern12", "anisotropic": False, "optMethod": "BFGS",
           "trainLikelihood": False, "retrainGp": 10},
    "Tc": {"method": 4, "kernel": "Matern12", "anisotropic": False, "optMethod": "L-BFGS-B",
           "trainLikelihood": False, "retrainGp": 10},
    "Hvap": {"method": 2, "kernel": "RQ", "anisotropic": False, "optMethod": "L-BFGS-B",
             "trainLikelihood": False, "retrainGp": 10},
    "logP": {"method": 4, "kernel": "Matern32", "anisotropic": True, "optMethod": "L-BFGS-B",
             "trainLikelihood": True, "retrainGp": 5},
}

# Folder and file prefix used for each property's results.
RUN_NAMES = {"Tb": "Tb", "Tm": "Tm", "Pc": "Pc", "Vc": "Vc", "Tc": "Tc", "Hvap": "Hvap", "logP": "LogP"}


def getSettings(code):
    """Settings dict of the final model for one property."""
    return FINAL_SETTINGS[code]


def getGpConfig(code):
    """gpConfig dict of the final model for one property."""
    s = FINAL_SETTINGS[code]
    return gpConfigFromMethod(s["method"], kernel=s["kernel"], anisotropic=s["anisotropic"],
                              trainLikelihood=s["trainLikelihood"], optMethod=s["optMethod"])


def settingsTable():
    """All final settings as a list of rows, handy for a notebook table."""
    return [{"Property": code, "Method": s["method"], "Kernel": s["kernel"], "Anisotropic": s["anisotropic"],
             "Optimizer": s["optMethod"], "TrainLikelihood": s["trainLikelihood"], "Restarts": s["retrainGp"]}
            for code, s in FINAL_SETTINGS.items()]
