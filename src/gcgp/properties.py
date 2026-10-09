"""Property definitions and data loading.

Each model-building file has the same five columns:
CAS number, PubChem SMILES, molecular weight, group-contribution (GC)
prediction, and the experimental value from the CRC Handbook. The GP models
use two inputs, ``[molecular weight, GC prediction]``, and predict the
experimental value.

For log P the "GC" column holds the Wildman-Crippen atom-contribution value
from RDKit, not a Joback-Reid estimate. The column is still called
``Log P (JR)`` in the data files, so the name is kept here.
"""

import pandas as pd

from .settings import CURATED_DATA_DIR, GLOBAL_SEED, MODEL_DATA_DIR, getRunSettings

PROPERTY_ORDER = ["Tb", "Tm", "Pc", "Vc", "Tc", "Hvap", "logP"]

PROPERTIES = {
    "Tb": {
        "file": "Tb_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "JR Tb /K"],
        "targetCol": "CRC Tb /K",
        "label": "Tb",
        "unit": "K",
        "xLabel": "Experimental Tb (K)",
        "yLabel": "Predicted Tb (K)",
    },
    "Tm": {
        "file": "Tm_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "JR Tm /K"],
        "targetCol": "CRC Tm /K",
        "label": "Tm",
        "unit": "K",
        "xLabel": "Experimental Tm (K)",
        "yLabel": "Predicted Tm (K)",
    },
    "Pc": {
        "file": "Pc_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "JR Pc /Bars"],
        "targetCol": "CRC Pc /Bars",
        "label": "Pc",
        "unit": "bar",
        "xLabel": "Experimental Pc (bar)",
        "yLabel": "Predicted Pc (bar)",
    },
    "Vc": {
        "file": "Vc_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "JR Vc /(cm3 mol-1)"],
        "targetCol": "CRC Vc /(cm3 mol-1)",
        "label": "Vc",
        "unit": "cm³/mol",
        "xLabel": "Experimental Vc (cm³/mol)",
        "yLabel": "Predicted Vc (cm³/mol)",
    },
    "Tc": {
        "file": "Tc_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "JR Tc /K"],
        "targetCol": "CRC Tc /K",
        "label": "Tc",
        "unit": "K",
        "xLabel": "Experimental Tc (K)",
        "yLabel": "Predicted Tc (K)",
    },
    "Hvap": {
        "file": "Hvap_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. Wt.", "JR Hvap /(kJ mol-1)"],
        "targetCol": "CRC Hvap /(kJ mol-1)",
        "label": "ΔHvap",
        "unit": "kJ/mol",
        "xLabel": "Experimental Hvap (kJ/mol)",
        "yLabel": "Predicted Hvap (kJ/mol)",
    },
    "logP": {
        "file": "logP_prediction_data_fcl_with_N.csv",
        "featureCols": ["Mol. wt.", "Log P (JR)"],
        "targetCol": "logP (CRC)",
        "label": "log P",
        "unit": "",
        "xLabel": "Experimental logP",
        "yLabel": "Predicted logP",
    },
}


def loadRawPropertyFile(code, dataDir=MODEL_DATA_DIR):
    """Read the model-building CSV for one property, without any filtering."""
    return pd.read_csv(dataDir / PROPERTIES[code]["file"])


def subsampleForFastRun(df, seed=GLOBAL_SEED):
    """In fast-run mode, keep a random subset of rows; otherwise return df as is."""
    maxRows = getRunSettings()["maxRows"]
    if maxRows is None or len(df) <= maxRows:
        return df
    return df.sample(n=maxRows, random_state=seed).sort_index().reset_index(drop=True)


def loadInputData(code, dataDir=MODEL_DATA_DIR):
    """Input table for one property with rows missing an input or target removed.

    Rows are kept in file order. Only ``00_Data_Curation`` reads the input files
    this way; the modelling notebooks read the curated datasets with
    :func:`loadPropertyData`.
    """
    info = PROPERTIES[code]
    df = loadRawPropertyFile(code, dataDir)
    return df.dropna(subset=info["featureCols"] + [info["targetCol"]]).reset_index(drop=True)


def curatedFilePath(code):
    """Location of the curated dataset of one property."""
    return CURATED_DATA_DIR / f"{code}_curated.csv"


def loadPropertyData(code):
    """Load the curated dataset of one property (written by ``00_Data_Curation``).

    Rows are kept in file order, which matters because the train/test split
    and the K folds are drawn from row positions. Values are read with
    ``float_precision="round_trip"``, so every number is bit-for-bit the value
    that was written. In a fast run a random subset of rows is returned.
    """
    path = curatedFilePath(code)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run notebooks/00_Data_Curation.ipynb first.")
    df = pd.read_csv(path, float_precision="round_trip")
    return subsampleForFastRun(df)


def datasetSizeTable(dataDir=MODEL_DATA_DIR):
    """Number of usable molecules per property (useful for a 'Table 1')."""
    rows = []
    for code in PROPERTY_ORDER:
        info = PROPERTIES[code]
        raw = loadRawPropertyFile(code, dataDir)
        clean = raw.dropna(subset=info["featureCols"] + [info["targetCol"]])
        rows.append({
            "Property": code,
            "Rows in file": len(raw),
            "Usable rows": len(clean),
            "Unique SMILES": clean["PubChem SMILES"].nunique(),
        })
    return pd.DataFrame(rows)
