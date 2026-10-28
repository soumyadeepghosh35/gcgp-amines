"""Molecule-level comparison of the JRGC and GC-GP predictions (notebook 07).

For every molecule used in the final cross-validation (notebook 04) the GC-GP
prediction is its out-of-fold prediction, so it comes from a model that never
saw that molecule. The JRGC prediction is either the raw group-contribution
estimate from the input file (``jrgcSource="raw"``) or the fitted linear GC
baseline from notebook 01 (``jrgcSource="fitted"``).

Molecules are aligned with the cross-validation predictions through
``RowIndex``, which is the row position in the curated dataset used by
notebook 04. The SMILES of every row are checked, so duplicate SMILES or tied
experimental values can never pair a prediction with the wrong molecule.

Every molecule is identified by its structure, not by how its SMILES happens to
be written. The input files sometimes list one structure more than once: the
same compound written as two different SMILES strings, or stereoisomers and
cis/trans isomers whose SMILES carry no stereochemistry, so they cannot be
told apart. With ``uniqueStructures=True``
(the default) all entries with the same structure key are merged into one row
before ranking, so each structure is listed, counted and drawn once.
:func:`duplicateStructures` lists every merged entry.
"""

import numpy as np
import pandas as pd

from .finalConfig import RUN_NAMES
from .properties import PROPERTIES, PROPERTY_ORDER, loadPropertyData
from .settings import OUTPUT_DIR
from .structures import STRUCTURE_LEVELS, structureKeys

CAS_COL = "CAS Reg. No."
SMILES_COL = "PubChem SMILES"
KEY_COL = "StructureKey"
CANONICAL_COL = "Canonical_SMILES"

CATEGORY_BOTH = "Poorly predicted by both"
CATEGORY_JRGC_ONLY = "Poor JRGC, corrected by GC-GP"
CATEGORY_GCGP_ONLY = "Poor GC-GP only"
CATEGORY_NONE = "Not among the worst for either"
CATEGORY_ORDER = [CATEGORY_BOTH, CATEGORY_JRGC_ONLY, CATEGORY_GCGP_ONLY, CATEGORY_NONE]

# Chemical motifs used to describe the molecules in each category (SMARTS).
CHEMICAL_MOTIFS = {
    "Primary amine": "[NX3;H2;!$(NC=O);!$(N-a)][CX4]",
    "Secondary amine": "[NX3;H1;!$(NC=O)]([CX4])[CX4]",
    "Tertiary amine": "[NX3;H0;!$(NC=O)]([CX4])([CX4])[CX4]",
    "Aniline-type N": "[NX3;!$(NC=O)]-a",
    "Aromatic N (pyridine-like)": "[nX2]",
    "Aromatic NH (pyrrole-like)": "[nH]",
    "Amide": "[NX3][CX3]=O",
    "Nitrile": "C#N",
    "Nitro / nitroso": "[$([NX3+](=O)[O-]),$(N=O)]",
    "Hydroxyl": "[OX2H][#6]",
    "Carboxylic acid": "C(=O)[OH]",
    "Fluorinated": "[F]",
    "Other halogen": "[Cl,Br,I]",
    "Sulfur": "[#16]",
}


def loadCvRunData(code):
    """The data table notebook 04 used for one property.

    The curated dataset followed by the ``dropna`` in
    :func:`gcgp.crossValidation.runGpCrossValidation`, so row positions match
    the ``RowIndex`` column of the saved predictions.
    """
    info = PROPERTIES[code]
    df = loadPropertyData(code)
    return df.dropna(subset=info["featureCols"] + [info["targetCol"]]).reset_index(drop=True)


def loadOutOfFoldPredictions(code, cvDir=None):
    """Out-of-fold GC-GP predictions for one property, one row per entry."""
    runName = RUN_NAMES[code]
    cvDir = cvDir or OUTPUT_DIR / "final_gcgp_cv"
    path = cvDir / runName / f"{runName}_all_cv_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 04_Final_GCGP_CrossValidation.ipynb first.")
    table = pd.read_csv(path)
    if "RowIndex" not in table.columns:
        raise ValueError(f"{path} has no RowIndex column. Rerun notebook 04 with the current code.")
    return table[table["Set"] == "Test"].copy(), runName


def loadFittedBaseline(code, baselineDir=None):
    """Fitted linear GC baseline predictions from notebook 01 (train and test molecules)."""
    baselineDir = baselineDir or OUTPUT_DIR / "jrgc_baseline"
    parts = []
    for setName in ["train", "test"]:
        path = baselineDir / f"JRGC_{setName}_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 01_JRGC_Baseline.ipynb first, "
                                    "or set jrgcSource = 'raw'.")
        part = pd.read_csv(path)[[CAS_COL, SMILES_COL, "Predicted"]]
        part["JRGC_Set"] = setName.capitalize()
        parts.append(part)
    fitted = pd.concat(parts, ignore_index=True)
    return fitted.drop_duplicates(subset=[CAS_COL, SMILES_COL], keep="first")


def addErrorColumns(table):
    """Errors, absolute errors and the difference between the two predictions."""
    table["JRGC_Error"] = table["JRGC"] - table["Experimental"]
    table["GCGP_Error"] = table["GCGP"] - table["Experimental"]
    table["JRGC_AbsError"] = table["JRGC_Error"].abs()
    table["GCGP_AbsError"] = table["GCGP_Error"].abs()
    table["Prediction_Difference"] = table["GCGP"] - table["JRGC"]
    table["Abs_Prediction_Difference"] = table["Prediction_Difference"].abs()
    table["Error_Reduction"] = table["JRGC_AbsError"] - table["GCGP_AbsError"]
    table["GCGP_Inside_95CI"] = table["GCGP_AbsError"] <= table["GCGP_95CI"]
    return table


def joinUnique(values):
    seen = []
    for value in values:
        text = str(value)
        if pd.notna(value) and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def collapseDuplicates(table):
    """Merge rows that share a structure key into one row per structure.

    Numeric values (experimental value, predictions, 95% half-width, molecular
    weight) are averaged over the entries; CAS numbers, SMILES as written and
    folds are kept as lists. ``N_Entries`` counts the merged rows and
    ``Experimental_Spread`` is the range of their experimental values. A spread
    above zero usually means isomers that the SMILES do not distinguish, so the
    entries are different compounds with the same model inputs.
    """
    numeric = ["Molecular_Weight", "Experimental", "JRGC_Raw", "JRGC", "GCGP", "GCGP_95CI"]
    aggregations = {column: "mean" for column in numeric if column in table.columns}
    aggregations.update({"Property": "first", "Run": "first", CANONICAL_COL: "first",
                         CAS_COL: joinUnique, SMILES_COL: joinUnique, "Fold": joinUnique})
    if "JRGC_Set" in table.columns:
        aggregations["JRGC_Set"] = joinUnique
    grouped = table.groupby(KEY_COL, sort=False)
    merged = grouped.agg(aggregations)
    merged["N_Entries"] = grouped.size()
    merged["Experimental_Spread"] = grouped["Experimental"].max() - grouped["Experimental"].min()
    merged = merged.reset_index()
    first = ["Property", "Run", KEY_COL, CANONICAL_COL, CAS_COL, SMILES_COL, "N_Entries",
             "Experimental_Spread", "Fold"]
    return merged[first + [c for c in merged.columns if c not in first]]


def compareMolecules(code, jrgcSource="raw", cvDir=None, baselineDir=None,
                     uniqueStructures=True, structureLevel="inchikey"):
    """Per-molecule table with the experimental value and both predictions.

    Columns ending in ``_AbsError`` are absolute errors in property units.
    ``Prediction_Difference`` is GC-GP minus JRGC, and ``Error_Reduction`` is the
    JRGC absolute error minus the GC-GP absolute error (positive means GC-GP is
    closer to experiment). With ``uniqueStructures=True`` there is exactly one
    row per structure (see :func:`collapseDuplicates`).
    """
    info = PROPERTIES[code]
    data = loadCvRunData(code)
    predictions, runName = loadOutOfFoldPredictions(code, cvDir)

    rows = data.iloc[predictions["RowIndex"].to_numpy()].reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    mismatch = (rows[SMILES_COL].astype(str).to_numpy() != predictions["SMILES"].astype(str).to_numpy())
    if mismatch.any():
        raise ValueError(f"{runName}: {int(mismatch.sum())} rows do not match the notebook 04 data. "
                         "The curated data changed after notebook 04 was run; rerun it.")

    table = pd.DataFrame({
        "Property": code, "Run": runName, "Fold": predictions["Fold"].to_numpy(),
        CAS_COL: rows[CAS_COL].to_numpy(), SMILES_COL: rows[SMILES_COL].to_numpy(),
        "Molecular_Weight": rows[info["featureCols"][0]].to_numpy(),
        "Experimental": predictions[info["xLabel"]].to_numpy(),
        "JRGC_Raw": rows[info["featureCols"][1]].to_numpy(),
        "GCGP": predictions[info["yLabel"]].to_numpy(),
        "GCGP_95CI": predictions["95% CI"].to_numpy(),
    })
    table[KEY_COL], table[CANONICAL_COL] = structureKeys(table[SMILES_COL].tolist(), structureLevel)

    if jrgcSource == "fitted":
        fitted = loadFittedBaseline(code, baselineDir)
        table = table.merge(fitted, on=[CAS_COL, SMILES_COL], how="left")
        table = table.rename(columns={"Predicted": "JRGC"})
        missing = int(table["JRGC"].isna().sum())
        if missing:
            print(f"{code}: {missing} of {len(table)} molecules are not in the fitted-baseline set and are left out.")
        table = table.dropna(subset=["JRGC"]).reset_index(drop=True)
    elif jrgcSource == "raw":
        table["JRGC"] = table["JRGC_Raw"]
    else:
        raise ValueError("jrgcSource must be 'raw' or 'fitted'.")

    if uniqueStructures:
        table = collapseDuplicates(table)
    else:
        table["N_Entries"] = 1
        table["Experimental_Spread"] = 0.0
    return addErrorColumns(table)


def duplicateStructures(code, structureLevel="inchikey", cvDir=None):
    """Every entry of a notebook 04 run whose structure occurs more than once.

    One row per entry, with its own experimental value and out-of-fold GC-GP
    prediction. ``Different_Folds`` is True when copies of the structure were in
    different folds, i.e. the structure was in the training set of the fold
    that predicted its copy.
    """
    info = PROPERTIES[code]
    data = loadCvRunData(code)
    predictions, runName = loadOutOfFoldPredictions(code, cvDir)
    rows = data.iloc[predictions["RowIndex"].to_numpy()].reset_index(drop=True)
    table = pd.DataFrame({
        "Property": code, "Run": runName, "RowIndex": predictions["RowIndex"].to_numpy(),
        "Fold": predictions["Fold"].to_numpy(), CAS_COL: rows[CAS_COL].to_numpy(),
        SMILES_COL: rows[SMILES_COL].to_numpy(), "Experimental": predictions[info["xLabel"]].to_numpy(),
        "JRGC_Raw": rows[info["featureCols"][1]].to_numpy(), "GCGP": predictions[info["yLabel"]].to_numpy(),
    })
    table[KEY_COL], table[CANONICAL_COL] = structureKeys(table[SMILES_COL].tolist(), structureLevel)
    table = table[table[KEY_COL].duplicated(keep=False)].copy()
    if table.empty:
        return table
    grouped = table.groupby(KEY_COL)
    table["N_Entries"] = grouped[KEY_COL].transform("size")
    table["Experimental_Spread"] = grouped["Experimental"].transform(lambda s: s.max() - s.min())
    table["Different_Folds"] = grouped["Fold"].transform("nunique") > 1
    table["Same_SMILES_Text"] = grouped[SMILES_COL].transform("nunique") == 1
    return table.sort_values([KEY_COL, "RowIndex"]).reset_index(drop=True)


def resolveTopN(topN, nMolecules):
    """Number of molecules that count as "poorly predicted".

    An integer is used as it is; a value between 0 and 1 is read as a fraction
    of the molecules of that property (for example 0.05 for the worst 5%).
    """
    if 0 < topN < 1:
        topN = int(np.ceil(topN * nMolecules))
    return int(min(topN, nMolecules))


def assignCategories(table, topN=20):
    """Rank the molecules by each model's error and label them.

    A molecule is "poorly predicted" by a model when it is among the ``topN``
    largest absolute errors of that model for this property. The ranks are
    added as ``JRGC_Error_Rank`` and ``GCGP_Error_Rank`` (1 = largest error).
    """
    table = table.copy()
    topN = resolveTopN(topN, len(table))
    table["JRGC_Error_Rank"] = table["JRGC_AbsError"].rank(ascending=False, method="first").astype(int)
    table["GCGP_Error_Rank"] = table["GCGP_AbsError"].rank(ascending=False, method="first").astype(int)
    table["Difference_Rank"] = table["Abs_Prediction_Difference"].rank(ascending=False, method="first").astype(int)
    poorJrgc = table["JRGC_Error_Rank"] <= topN
    poorGcgp = table["GCGP_Error_Rank"] <= topN
    table["Category"] = np.select(
        [poorJrgc & poorGcgp, poorJrgc & ~poorGcgp, ~poorJrgc & poorGcgp],
        [CATEGORY_BOTH, CATEGORY_JRGC_ONLY, CATEGORY_GCGP_ONLY], default=CATEGORY_NONE)
    return table


def compareAllProperties(propertyCodes=PROPERTY_ORDER, jrgcSource="raw", topN=20, cvDir=None,
                         baselineDir=None, uniqueStructures=True, structureLevel="inchikey"):
    """Run :func:`compareMolecules` and :func:`assignCategories` for every property.

    A property whose notebook 04 output is missing is reported and skipped.
    """
    tables = []
    for code in propertyCodes:
        try:
            table = compareMolecules(code, jrgcSource=jrgcSource, cvDir=cvDir, baselineDir=baselineDir,
                                     uniqueStructures=uniqueStructures, structureLevel=structureLevel)
        except (FileNotFoundError, ValueError) as error:
            print(f"{code}: skipped. {error}")
            continue
        tables.append(assignCategories(table, topN))
    if not tables:
        raise FileNotFoundError("No cross-validation predictions found. Run notebook 04 first.")
    return pd.concat(tables, ignore_index=True)


def summarizeComparison(comparisonDf):
    """One row per property: errors of both models and the category counts."""
    rows = []
    for code, group in comparisonDf.groupby("Property", sort=False):
        row = {
            "Property": code, "Run": group["Run"].iloc[0], "N_Entries": int(group["N_Entries"].sum()),
            "N_Molecules": len(group), "N_Merged_Structures": int((group["N_Entries"] > 1).sum()),
            "N_Isomer_Groups": int((group["Experimental_Spread"] > 0).sum()),
            "JRGC_MAE": group["JRGC_AbsError"].mean(), "GCGP_MAE": group["GCGP_AbsError"].mean(),
            "JRGC_RMSE": float(np.sqrt((group["JRGC_Error"] ** 2).mean())),
            "GCGP_RMSE": float(np.sqrt((group["GCGP_Error"] ** 2).mean())),
            "Frac_GCGP_Closer": float((group["Error_Reduction"] > 0).mean()),
            "Median_Error_Reduction": group["Error_Reduction"].median(),
            "Median_Abs_Prediction_Difference": group["Abs_Prediction_Difference"].median(),
            "Max_Abs_Prediction_Difference": group["Abs_Prediction_Difference"].max(),
        }
        counts = group["Category"].value_counts()
        for category in CATEGORY_ORDER[:3]:
            row[f"N_{category}"] = int(counts.get(category, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def largestDifferences(comparisonDf, nPerProperty=10):
    """The molecules where GC-GP and JRGC disagree the most, per property."""
    columns = ["Property", "Difference_Rank", CAS_COL, CANONICAL_COL, "N_Entries", "Experimental", "JRGC", "GCGP",
               "GCGP_95CI", "Prediction_Difference", "JRGC_AbsError", "GCGP_AbsError", "Error_Reduction",
               "Category"]
    return (comparisonDf[comparisonDf["Difference_Rank"] <= nPerProperty]
            .sort_values(["Property", "Difference_Rank"], key=lambda s: s.map(PROPERTY_ORDER.index)
                         if s.name == "Property" else s)[columns]
            .reset_index(drop=True))


def motifTable(smilesList):
    """Boolean table of chemical motifs (one row per SMILES). Needs RDKit."""
    from rdkit import Chem

    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in CHEMICAL_MOTIFS.items()}
    rows = []
    for smiles in smilesList:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        rows.append({name: (bool(mol.HasSubstructMatch(p)) if mol is not None else np.nan)
                     for name, p in patterns.items()})
    return pd.DataFrame(rows)


def motifEnrichment(comparisonDf):
    """Share of molecules with each motif, overall and in every category.

    The first row of each property gives the number of molecules in each
    column; the other rows are percentages, so a motif that is much more common
    among the poorly predicted molecules than in the whole set stands out.
    """
    motifs = motifTable(comparisonDf[CANONICAL_COL].tolist())
    work = pd.concat([comparisonDf[["Property", "Category"]].reset_index(drop=True), motifs], axis=1)
    motifNames = list(CHEMICAL_MOTIFS)
    parts = []
    for code, group in work.groupby("Property", sort=False):
        subsets = {"All molecules": group}
        subsets.update({category: group[group["Category"] == category] for category in CATEGORY_ORDER[:3]})
        shares = {name: subset[motifNames].astype(float).mean() * 100 if len(subset)
                  else pd.Series(np.nan, index=motifNames) for name, subset in subsets.items()}
        part = pd.DataFrame(shares).rename_axis("Motif").reset_index()
        counts = pd.DataFrame([{"Motif": "Number of molecules", **{n: len(s) for n, s in subsets.items()}}])
        part = pd.concat([counts, part], ignore_index=True)
        part.insert(0, "Property", code)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


CATEGORY_COLORS = {
    CATEGORY_BOTH: "#b2182b",
    CATEGORY_JRGC_ONLY: "#2166ac",
    CATEGORY_GCGP_ONLY: "#e08214",
    CATEGORY_NONE: "#bdbdbd",
}


def plotErrorComparison(table, code, savePath=None, nLabels=10, show=True):
    """JRGC absolute error against GC-GP absolute error for one property.

    Points below the dashed line are molecules where GC-GP is closer to the
    experimental value. The ``nLabels`` molecules with the largest difference
    between the two predictions are numbered by their ``Difference_Rank``, so
    they can be looked up in the table of largest differences.
    """
    import matplotlib.pyplot as plt

    from .plotting import finishFigure, savePlot

    unit = PROPERTIES[code]["unit"]
    unitText = f" ({unit})" if unit else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for category in reversed(CATEGORY_ORDER):
        part = table[table["Category"] == category]
        if len(part):
            ax.scatter(part["JRGC_AbsError"], part["GCGP_AbsError"], s=22, alpha=0.75,
                       color=CATEGORY_COLORS[category], label=f"{category} (n={len(part)})",
                       edgecolor="none")
    high = float(max(table["JRGC_AbsError"].max(), table["GCGP_AbsError"].max())) * 1.05
    ax.plot([0, high], [0, high], "k--", linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)
    ax.set_xlabel(f"JRGC absolute error{unitText}")
    ax.set_ylabel(f"GC-GP absolute error{unitText}")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(table["JRGC"], table["GCGP"], s=18, alpha=0.6, color="#4d4d4d", edgecolor="none")
    low = float(min(table["JRGC"].min(), table["GCGP"].min()))
    top = float(max(table["JRGC"].max(), table["GCGP"].max()))
    ax.plot([low, top], [low, top], "k--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(f"JRGC prediction{unitText}")
    ax.set_ylabel(f"GC-GP prediction{unitText}")
    ax.grid(True, alpha=0.3)

    labelled = table[table["Difference_Rank"] <= nLabels]
    for _, row in labelled.iterrows():
        rank = int(row["Difference_Rank"])
        axes[0].annotate(str(rank), (row["JRGC_AbsError"], row["GCGP_AbsError"]), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
        axes[1].annotate(str(rank), (row["JRGC"], row["GCGP"]), fontsize=7, color="#b2182b",
                         xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)


def pickDiverse(smilesList, nPick=6, seed=42):
    """Positions of up to ``nPick`` structurally diverse molecules (MaxMin on Morgan fingerprints)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    valid = [(i, Chem.MolFromSmiles(s)) for i, s in enumerate(smilesList) if isinstance(s, str)]
    valid = [(i, m) for i, m in valid if m is not None]
    if len(valid) <= nPick:
        return [i for i, _ in valid]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [generator.GetFingerprint(m) for _, m in valid]

    def distance(a, b):
        return 1 - DataStructs.TanimotoSimilarity(fingerprints[a], fingerprints[b])

    picked = MaxMinPicker().LazyPick(distance, len(fingerprints), nPick, seed=seed)
    return sorted(valid[p][0] for p in picked)


def drawStructureGrid(table, code, savePath, nStructures=6, diverse=True, molsPerRow=3,
                      subImageSize=900, show=True):
    """Draw molecules with their experimental, JRGC and GC-GP values.

    ``table`` should already be sorted by importance (for example by the
    difference between the two predictions). With ``diverse=True`` a
    structurally diverse subset is drawn instead of the first rows, so one
    family of near-identical molecules does not fill the whole grid. The grid is
    saved as PNG and PDF at 600 dpi. Returns the rows that were drawn.
    """
    from pathlib import Path

    from rdkit import Chem
    from rdkit.Chem import Draw

    if table.empty:
        return table
    key = KEY_COL if KEY_COL in table.columns else SMILES_COL
    table = table.drop_duplicates(subset=[key]).reset_index(drop=True)
    smilesCol = CANONICAL_COL if CANONICAL_COL in table.columns else SMILES_COL
    positions = pickDiverse(table[smilesCol].tolist(), nStructures) if diverse \
        else list(range(min(nStructures, len(table))))
    chosen = table.iloc[positions].reset_index(drop=True)
    unit = PROPERTIES[code]["unit"]
    mols, legends = [], []
    for _, row in chosen.iterrows():
        mol = Chem.MolFromSmiles(row[smilesCol])
        if mol is None:
            continue
        mols.append(mol)
        unitText = f" {unit}" if unit else ""
        casText = str(row[CAS_COL]).split("; ")[0] if pd.notna(row[CAS_COL]) and str(row[CAS_COL]) else "n/a"
        entries = int(row.get("N_Entries", 1))
        if entries > 1:
            casText += f" (+{entries - 1} entr{'y' if entries == 2 else 'ies'})"
        legends.append(f"CAS {casText}   Exp. {row['Experimental']:.4g}{unitText}\n"
                       f"JRGC {row['JRGC']:.4g}   GC-GP {row['GCGP']:.4g} +/- {row['GCGP_95CI']:.2g}")
    if not mols:
        return chosen
    options = Draw.MolDrawOptions()
    options.legendFontSize = 34
    options.legendFraction = 0.18
    image = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(subImageSize, subImageSize),
                                 legends=legends, drawOptions=options, returnPNG=False)
    savePath = Path(savePath)
    savePath.parent.mkdir(parents=True, exist_ok=True)
    image.save(savePath.with_suffix(".png"), dpi=(600, 600))
    image.convert("RGB").save(savePath.with_suffix(".pdf"), resolution=600)
    if show:
        try:
            from IPython.display import display

            display(image)
        except ImportError:
            pass
    return chosen
