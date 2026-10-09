"""Project paths, random seeds and run settings.

The repository uses exactly two data folders:

* ``Input_Data/``: everything the notebooks read. Nothing in it is ever
  written to.
* ``Output_Data/``: everything the notebooks write (tables, predictions and
  figures). It can be deleted at any time and rebuilt by rerunning the
  notebooks.

The modelling notebooks (02 to 07) read the datasets written by
``00_Data_Curation`` to ``Output_Data/curated_data/``, so that notebook runs
first.

Both are found relative to the repository root, so the notebooks run the same
way on any machine. Two environment variables can point them elsewhere:

* ``GCGP_INPUT_DIR``: replaces ``Input_Data/``.
* ``GCGP_OUTPUT_DIR``: replaces ``Output_Data/``.

Set ``GCGP_FAST_RUN=1`` to run every notebook on a small subset of the data with
fewer optimizer restarts. This is only meant for checking that the code runs
from top to bottom; its numbers are not final results.
"""

import os
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = Path(os.environ.get("GCGP_INPUT_DIR", REPO_ROOT / "Input_Data"))
MODEL_DATA_DIR = INPUT_DIR / "model_building"
RAW_DATA_DIR = INPUT_DIR / "raw"
REFERENCE_RESULTS_DIR = INPUT_DIR / "reference_results"

OUTPUT_DIR = Path(os.environ.get("GCGP_OUTPUT_DIR", REPO_ROOT / "Output_Data"))
CURATED_DATA_DIR = OUTPUT_DIR / "curated_data"

# Every input file the notebooks need, grouped by the notebooks that read it.
# ``checkInputs`` uses this list to report anything missing before a run starts.
REQUIRED_INPUTS = {
    "model_building": [
        "Tb_prediction_data_fcl_with_N.csv",
        "Tm_prediction_data_fcl_with_N.csv",
        "Pc_prediction_data_fcl_with_N.csv",
        "Vc_prediction_data_fcl_with_N.csv",
        "Tc_prediction_data_fcl_with_N.csv",
        "Hvap_prediction_data_fcl_with_N.csv",
        "logP_prediction_data_fcl_with_N.csv",
    ],
    "raw": [
        "feature_engineered_raw_data.csv",
    ],
    "reference_results": [
        "Tb_all_cv_predictions.csv",
        "Tm_all_cv_predictions.csv",
        "Pc_all_cv_predictions.csv",
        "Vc_all_cv_predictions.csv",
        "Tc_all_cv_predictions.csv",
        "Hvap_all_cv_predictions.csv",
        "LogP_all_cv_predictions.csv",
    ],
}

GLOBAL_SEED = 42


def isFastRun():
    """Return True when the quick smoke-test mode is switched on."""
    return os.environ.get("GCGP_FAST_RUN", "0").strip().lower() in {"1", "true", "yes"}


def getRunSettings():
    """Return the settings that change between a full run and a fast run.

    In a full run nothing is overridden, so every notebook uses the values it
    declares. In a fast run the data are subsampled, the GP uses only two
    restarts and cross-validation uses three folds, which keeps each notebook
    to a few minutes on a laptop.
    """
    if isFastRun():
        return {
            "fastRun": True,
            "maxRows": 120,
            "retrainGp": 2,
            "nFolds": 3,
            "maxIter": 10**9,
        }
    return {
        "fastRun": False,
        "maxRows": None,
        "retrainGp": None,
        "nFolds": None,
        "maxIter": 10**9,
    }


def restartWorkers():
    """Number of worker processes used to fit GP restarts in parallel.

    Set with the environment variable ``GCGP_N_WORKERS`` (default 1, which
    fits the restarts one after another in the notebook process). The results
    are the same for any value; only the run time changes.
    """
    value = os.environ.get("GCGP_N_WORKERS", "1").strip()
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def setGlobalSeed(seed=GLOBAL_SEED):
    """Seed Python, NumPy and TensorFlow so a fresh run gives the same numbers.

    The GP code also re-seeds NumPy and TensorFlow before every optimizer
    restart (seed = 100 x restart index), exactly as in the original notebooks.
    This global seed covers everything else, such as data shuffling.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensureDir(path):
    """Create a folder (and its parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkInputs(groups=("model_building", "raw")):
    """Make sure every required input file is present in ``Input_Data/``.

    ``groups`` selects which parts of :data:`REQUIRED_INPUTS` the calling
    notebook needs. Raises ``FileNotFoundError`` listing every missing file,
    so a run never stops halfway because of a missing input.
    """
    missing = [INPUT_DIR / group / name
               for group in groups for name in REQUIRED_INPUTS[group]
               if not (INPUT_DIR / group / name).is_file()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"These input files are missing from {INPUT_DIR}:\n{listing}")
    return True


def checkCuratedData(propertyCodes=None):
    """Make sure ``00_Data_Curation`` has written the curated datasets.

    Raises ``FileNotFoundError`` naming the missing files, so a modelling
    notebook stops before fitting anything.
    """
    from .properties import PROPERTY_ORDER, curatedFilePath

    missing = [curatedFilePath(code) for code in (propertyCodes or PROPERTY_ORDER)
               if not curatedFilePath(code).is_file()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Curated datasets not found:\n{listing}\n"
                                "Run notebooks/00_Data_Curation.ipynb first.")
    return True


def outputFolder(*parts):
    """Return a folder inside ``Output_Data/``, creating it if needed."""
    return ensureDir(OUTPUT_DIR.joinpath(*parts))


def describeSetup():
    """Short text summary of where data are read from and results written to."""
    mode = "FAST (quick check on small subsets; numbers are not final results)" if isFastRun() else "FULL"
    return (
        f"Run mode      : {mode}\n"
        f"Input folder  : {INPUT_DIR}\n"
        f"Output folder : {OUTPUT_DIR}\n"
        f"Curated data  : {CURATED_DATA_DIR}\n"
        f"GP workers    : {restartWorkers()} (set GCGP_N_WORKERS to fit restarts in parallel)\n"
        f"Global seed   : {GLOBAL_SEED}"
    )
