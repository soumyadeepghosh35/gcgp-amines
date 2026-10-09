"""Checkpoint files so that long runs can stop and resume.

Every GP optimizer restart is saved to ``Output_Data/checkpoints/gp_restarts/``
as soon as it finishes. The file name is a hash of everything that decides the
result: the training data, the GP settings, the restart index and the
optimizer limit. When a notebook is run again, each restart whose file already
exists is read back instead of being refitted, so an interrupted run carries
on from the last finished restart. If the data or any setting changes, the
hash changes too and the restart is fitted again, so a checkpoint can never be
reused for the wrong model.

Set ``GCGP_RESUME=0`` to ignore existing checkpoints and refit everything
(new checkpoints are still written). Delete ``Output_Data/checkpoints/`` to
remove them all.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from .settings import OUTPUT_DIR

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_VERSION = 1


def isResumeEnabled():
    """True unless the environment variable GCGP_RESUME is set to 0/false/no."""
    return os.environ.get("GCGP_RESUME", "1").strip().lower() not in {"0", "false", "no"}


def hashArray(array):
    """Stable short hash of a NumPy array (shape, dtype and values)."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha1()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def makeKey(**parts):
    """Hash of the given parts. Arrays are hashed by value, other values by text."""
    normalized = {}
    for name, value in parts.items():
        if isinstance(value, np.ndarray):
            normalized[name] = "array:" + hashArray(value)
        else:
            normalized[name] = value
    text = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha1(f"v{CHECKPOINT_VERSION}:{text}".encode()).hexdigest()


def checkpointPath(category, key):
    """File path for one checkpoint; files are spread over 256 subfolders."""
    return CHECKPOINT_DIR / category / key[:2] / f"{key}.pkl"


def loadCheckpoint(category, key):
    """Return the saved object, or None if there is none (or resume is off)."""
    if not isResumeEnabled():
        return None
    path = checkpointPath(category, key)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def saveCheckpoint(category, key, obj):
    """Write a checkpoint atomically, so a crash never leaves a half-written file."""
    path = checkpointPath(category, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpPath = path.with_suffix(".tmp")
    with open(tmpPath, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmpPath, path)


def checkpointStatus():
    """Short text describing whether checkpoints are used and where they live."""
    count = sum(1 for _ in CHECKPOINT_DIR.rglob("*.pkl")) if CHECKPOINT_DIR.exists() else 0
    mode = "resume from saved restarts" if isResumeEnabled() else "refit everything (GCGP_RESUME=0)"
    return f"Checkpoints   : {mode}; {count} saved in {CHECKPOINT_DIR}"
