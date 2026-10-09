"""Data curation: build the dataset used for each property (notebook 00).

Every modelling notebook reads the curated datasets written here, so the
curation rules live in this one place.

``SELECTED_VERSION`` says how each property is curated:

* ``"allData"``: every usable molecule of the input file is kept.
* ``"noOutlier"``: molecules are removed with :func:`gcgp.outliers.filterByGcError`
  using the rule in ``CURATION_RULES``. First the experimental value must lie
  inside ``targetRange``; then, among the remaining molecules, the GC error
  (experimental minus GC estimate) must lie inside the 1.5 x IQR fences.

Properties listed in ``REMOVE_DUPLICATE_STRUCTURES`` are also reduced to one
entry per structure (standard InChIKey). The first entry of each structure is
kept and every later copy is listed as removed.

The curated file keeps the columns and row order of the input file, and the
values are written so they read back bit for bit. A model fitted on the curated
data therefore sees exactly the molecules, values and order it saw when the
same filter was applied inside the modelling notebooks.
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .outliers import filterByGcError
from .properties import PROPERTIES, PROPERTY_ORDER, curatedFilePath, loadInputData, loadRawPropertyFile
from .settings import CURATED_DATA_DIR, MODEL_DATA_DIR, ensureDir
from .structures import structureKeys

SELECTED_VERSION = {
    "Tb": "noOutlier",
    "Tm": "noOutlier",
    "Pc": "allData",
    "Vc": "allData",
    "Tc": "noOutlier",
    "Hvap": "noOutlier",
    "logP": "allData",
}

IQR_MULTIPLIER = 1.5

CURATION_RULES = {
    "Tb": {"targetRange": (0, 600)},
    "Tm": {"targetRange": (0, 700)},
    "Tc": {"targetRange": (0, 800)},
    "Hvap": {"targetRange": (10, 70)},
}

REMOVE_DUPLICATE_STRUCTURES = {"logP"}

CURATION_LOG = "curation_log.json"

REASON_RANGE = "Experimental value outside the accepted range"
REASON_GC_ERROR = "GC error outside the 1.5 x IQR fences"
REASON_DUPLICATE = "Duplicate structure"


def checkCurationConfig():
    """Every property must have a version, and every "noOutlier" one a rule."""
    for code in PROPERTY_ORDER:
        version = SELECTED_VERSION.get(code)
        if version not in ("allData", "noOutlier"):
            raise ValueError(f"SELECTED_VERSION['{code}'] must be 'allData' or 'noOutlier'.")
        if version == "noOutlier" and code not in CURATION_RULES:
            raise ValueError(f"{code} is set to 'noOutlier' but has no entry in CURATION_RULES.")


def fileHash(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def removedRows(df, mask, code, reason, detail=""):
    """Table of the rows where ``mask`` is True, with the reason they were removed."""
    info = PROPERTIES[code]
    part = df.loc[mask]
    return pd.DataFrame({
        "Property": code, "Input_Row": part.index.to_numpy(),
        "CAS Reg. No.": part["CAS Reg. No."].to_numpy(), "PubChem SMILES": part["PubChem SMILES"].to_numpy(),
        "Experimental": part[info["targetCol"]].to_numpy(), "GC_Estimate": part[info["featureCols"][1]].to_numpy(),
        "GC_Error": (part[info["targetCol"]] - part[info["featureCols"][1]]).to_numpy(),
        "Reason": reason, "Detail": detail,
    })


def curateProperty(code):
    """Curate one property.

    Returns ``(curatedDf, removedDf, summaryRow)``. ``removedDf`` lists every
    removed molecule with the reason; ``Input_Row`` is its position among the
    usable rows of the input file.
    """
    checkCurationConfig()
    info = PROPERTIES[code]
    nFileRows = len(loadRawPropertyFile(code))
    df = loadInputData(code)
    targetCol, gcCol = info["targetCol"], info["featureCols"][1]
    removed = []
    summary = {"Property": code, "Version": SELECTED_VERSION[code], "Rows_in_file": nFileRows,
               "Usable_rows": len(df), "Removed_range": 0, "Removed_GC_error": 0,
               "Removed_duplicates": 0, "Range": "", "GC_error_fences": ""}

    keep = pd.Series(True, index=df.index)
    if SELECTED_VERSION[code] == "noOutlier":
        rule = CURATION_RULES[code]
        low, high = rule["targetRange"]
        _, keep, _ = filterByGcError(df, targetCol, gcCol, targetRange=(low, high),
                                     iqrMultiplier=IQR_MULTIPLIER)
        inRange = (df[targetCol] >= low) & (df[targetCol] <= high)
        gcError = df[targetCol] - df[gcCol]
        q1, q3 = gcError[inRange].quantile(0.25), gcError[inRange].quantile(0.75)
        fenceLow, fenceHigh = q1 - IQR_MULTIPLIER * (q3 - q1), q3 + IQR_MULTIPLIER * (q3 - q1)
        removed.append(removedRows(df, ~inRange, code, REASON_RANGE, f"range [{low}, {high}]"))
        removed.append(removedRows(df, inRange & ~keep, code, REASON_GC_ERROR,
                                   f"fences [{fenceLow:.3f}, {fenceHigh:.3f}]"))
        summary.update(Removed_range=int((~inRange).sum()), Removed_GC_error=int((inRange & ~keep).sum()),
                       Range=f"[{low}, {high}]", GC_error_fences=f"[{fenceLow:.3f}, {fenceHigh:.3f}]")

    if code in REMOVE_DUPLICATE_STRUCTURES:
        keys, _ = structureKeys(df["PubChem SMILES"].tolist())
        keys = pd.Series(keys, index=df.index)
        firstRow = {}
        duplicate = pd.Series(False, index=df.index)
        details = {}
        for row in df.index[keep.to_numpy()]:
            key = keys[row]
            if key in firstRow:
                duplicate[row] = True
                kept = firstRow[key]
                details[row] = (f"same structure as input row {kept} ({df.at[kept, 'PubChem SMILES']}); "
                                f"experimental values differ by {abs(df.at[row, targetCol] - df.at[kept, targetCol]):.3g}")
            else:
                firstRow[key] = row
        part = removedRows(df, duplicate, code, REASON_DUPLICATE)
        part["Detail"] = [details[row] for row in part["Input_Row"]]
        removed.append(part)
        keep &= ~duplicate
        summary["Removed_duplicates"] = int(duplicate.sum())

    curated = df.loc[keep.to_numpy()].reset_index(drop=True)
    summary["Curated_rows"] = len(curated)
    removedDf = pd.concat(removed, ignore_index=True) if removed else removedRows(df, ~keep, code, "")
    return curated, removedDf, summary


def runCuration(propertyCodes=PROPERTY_ORDER, outDir=CURATED_DATA_DIR):
    """Curate every property and write the results to ``Output_Data/curated_data/``.

    Writes ``<property>_curated.csv``, ``removed_molecules.csv``,
    ``curation_summary.csv`` and ``curation_log.json`` (rules, row counts and
    file hashes, so a later run can tell whether the inputs changed).
    """
    outDir = ensureDir(outDir)
    summaries, removedTables, log = [], [], {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                            "iqrMultiplier": IQR_MULTIPLIER, "properties": {}}
    for code in propertyCodes:
        curated, removedDf, summary = curateProperty(code)
        path = curatedFilePath(code)
        curated.to_csv(path, index=False)
        check = pd.read_csv(path, float_precision="round_trip")
        numeric = curated.select_dtypes("number").columns
        if not np.array_equal(check[numeric].to_numpy(), curated[numeric].to_numpy(), equal_nan=True):
            raise RuntimeError(f"{path} does not read back exactly.")
        summaries.append(summary)
        removedTables.append(removedDf)
        inputPath = MODEL_DATA_DIR / PROPERTIES[code]["file"]
        log["properties"][code] = {
            "version": SELECTED_VERSION[code], "rule": CURATION_RULES.get(code),
            "removeDuplicateStructures": code in REMOVE_DUPLICATE_STRUCTURES,
            "inputFile": inputPath.name, "inputSha1": fileHash(inputPath),
            "curatedFile": path.name, "curatedSha1": fileHash(path), "rows": len(curated),
        }
    summaryDf = pd.DataFrame(summaries)
    removedDf = pd.concat(removedTables, ignore_index=True)
    summaryDf.to_csv(outDir / "curation_summary.csv", index=False)
    removedDf.to_csv(outDir / "removed_molecules.csv", index=False)
    (outDir / CURATION_LOG).write_text(json.dumps(log, indent=2, default=str))
    return summaryDf, removedDf


def plotCuration(code, savePath=None, show=True):
    """GC error of every usable molecule, with the accepted range and fences.

    Grey points are kept, coloured points removed. Only meaningful for a
    property curated with ``"noOutlier"``.
    """
    import matplotlib.pyplot as plt

    from .plotting import finishFigure, savePlot

    info = PROPERTIES[code]
    df = loadInputData(code)
    curated, removedDf, summary = curateProperty(code)
    gcError = df[info["targetCol"]] - df[info["featureCols"][1]]
    removedRange = removedDf[removedDf["Reason"] == REASON_RANGE]["Input_Row"]
    removedGc = removedDf[removedDf["Reason"] == REASON_GC_ERROR]["Input_Row"]
    keptRows = df.index.difference(pd.Index(removedDf["Input_Row"]))
    unit = info["unit"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[keptRows, info["targetCol"]], gcError[keptRows], s=12, color="#9e9e9e", alpha=0.6,
               edgecolor="none", label=f"Kept (n={len(keptRows)})")
    ax.scatter(df.loc[removedGc, info["targetCol"]], gcError[removedGc], s=16, color="#b2182b",
               edgecolor="none", label=f"GC error outside fences (n={len(removedGc)})")
    ax.scatter(df.loc[removedRange, info["targetCol"]], gcError[removedRange], s=16, color="#2166ac",
               marker="s", edgecolor="none", label=f"Outside range (n={len(removedRange)})")
    if summary["GC_error_fences"]:
        low, high = (float(v) for v in summary["GC_error_fences"].strip("[]").split(","))
        for value in (low, high):
            ax.axhline(value, color="#b2182b", linestyle="--", linewidth=1)
        rangeLow, rangeHigh = CURATION_RULES[code]["targetRange"]
        for value in (rangeLow, rangeHigh):
            if df[info["targetCol"]].min() <= value <= df[info["targetCol"]].max():
                ax.axvline(value, color="#2166ac", linestyle=":", linewidth=1)
    unitText = f" ({unit})" if unit else ""
    ax.set_xlabel(f"Experimental {info['label']}{unitText}")
    ax.set_ylabel(f"Experimental minus GC estimate{unitText}")
    ax.set_title(f"{info['label']}: {summary['Curated_rows']} of {summary['Usable_rows']} molecules kept")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)
