"""Plain-language result summaries written from the computed numbers.

Each notebook ends by calling one of the ``describe*`` functions. They turn
the result tables of that run into short sentences, so the text always matches
the data that was just analysed. The text is shown in the notebook and saved
as ``results_summary.md`` next to the other outputs, ready to be used as a
starting point when writing the results section.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .properties import PROPERTIES, PROPERTY_ORDER
from .settings import isFastRun


def formatNumber(value):
    """Round a metric to a sensible number of digits for text."""
    if value is None or not np.isfinite(value):
        return "n/a"
    size = abs(value)
    if size >= 100:
        return f"{value:.1f}"
    if size >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def withUnit(value, code):
    """Number followed by the property unit, e.g. '26.57 K'."""
    unit = PROPERTIES[code]["unit"] if code in PROPERTIES else ""
    return f"{formatNumber(value)} {unit}".strip()


def codeFromRun(runName):
    """Property code from a run name such as 'Tb' or 'LogP'."""
    return "logP" if runName.lower() == "logp" else runName


def label(code):
    return PROPERTIES[code]["label"] if code in PROPERTIES else code


def header(title):
    note = ("\n\n> These numbers come from a FAST run (small subsets, fewer restarts and folds). "
            "Rerun without GCGP_FAST_RUN before using them." if isFastRun() else "")
    return f"## {title}{note}\n"


def showAndSave(text, path):
    """Display the summary as Markdown in the notebook and save it as a .md file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)
    return path


def describeBaseline(metricsDf):
    """Sentences for notebook 01 (fitted linear GC baseline and raw GC)."""
    lines = [header("Group-contribution baseline: results")]
    fitted = metricsDf[metricsDf["Model"].str.startswith("Linear")].set_index("Property")
    raw = metricsDf[metricsDf["Model"].str.startswith("Raw")].set_index("Property")
    for code in [c for c in PROPERTY_ORDER if c in fitted.index]:
        f, r = fitted.loc[code], raw.loc[code] if code in raw.index else None
        sentence = (f"- **{label(code)}** ({int(f['N_Train'])} training / {int(f['N_Test'])} test molecules): "
                    f"the fitted baseline gives test R² = {formatNumber(f['R2_Test'])}, "
                    f"MAE = {withUnit(f['MAE_Test'], code)} and RMSE = {withUnit(f['RMSE_Test'], code)}.")
        if r is not None:
            better = "lower" if r["RMSE_Test"] < f["RMSE_Test"] else "higher"
            sentence += (f" The raw GC estimate on the same test molecules has RMSE = "
                         f"{withUnit(r['RMSE_Test'], code)} ({better} than the fitted baseline).")
        lines.append(sentence)
    rawWins = [c for c in fitted.index if c in raw.index and raw.loc[c, "RMSE_Test"] < fitted.loc[c, "RMSE_Test"]]
    if rawWins:
        lines.append(f"\nFor {', '.join(label(c) for c in rawWins)}, the raw GC estimate has a lower test RMSE "
                     "than the fitted baseline.")
    return "\n".join(lines) + "\n"


def describeSingleSplit(summaryDf, modelTitle):
    """Sentences for one 02_Model notebook."""
    lines = [header(f"{modelTitle}: results")]
    if summaryDf.empty:
        lines.append("No run produced a usable fit.")
        return "\n".join(lines) + "\n"
    for _, row in summaryDf.iterrows():
        code = row["Property"]
        lines.append(f"- {label(code)} ({int(row['N_Train'])} train / {int(row['N_Test'])} test): "
                     f"test R² = {formatNumber(row['Test_R2'])}, RMSE = {withUnit(row['Test_RMSE'], code)}, "
                     f"MAE = {withUnit(row['Test_MAE'], code)}; "
                     f"{100 * (1 - row['CI_Test_Outside']):.1f}% of test molecules inside the 95% interval.")
    return "\n".join(lines) + "\n"


def describeKernelComparison(metricsDf, bestDf):
    """Sentences for notebook 03 (best model per property and spread across models)."""
    lines = [header("Comparison of the twelve GC-GP models: results")]
    for code in [c for c in PROPERTY_ORDER if c in bestDf.index]:
        row = bestDf.loc[code]
        allR2 = metricsDf.loc[metricsDf["Property"] == code, "R2"]
        lines.append(f"- **{label(code)}**: best is {row['Model']} with test "
                     f"R² = {formatNumber(row['R2'])} and RMSE = {withUnit(row['RMSE'], code)}. Across the "
                     f"{len(allR2)} models, test R² ranges from {formatNumber(allR2.min())} to "
                     f"{formatNumber(allR2.max())}.")
    lines.append("\nEach value comes from a single train/test split, so small differences between "
                 "models should not be over-interpreted.")
    return "\n".join(lines) + "\n"


def describeCrossValidation(summaryDf, baselineDf=None):
    """Sentences for notebook 04 (pooled 10-fold CV of the final models)."""
    lines = [header("Final GC-GP models, cross-validation: results")]
    fitted = None
    if baselineDf is not None and not baselineDf.empty:
        fitted = baselineDf[baselineDf["Model"].str.startswith("Linear")].set_index("Property")
    for _, row in summaryDf.iterrows():
        code = codeFromRun(row["Property"])
        sentence = (f"- **{label(code)}** ({int(row['N_Molecules'])} molecules, "
                    f"{int(row['N_Successful_Folds'])} of {int(row['N_Folds'])} folds used): pooled test "
                    f"R² = {formatNumber(row['Overall_Test_R2'])} (fold mean {formatNumber(row['Mean_Test_R2'])} "
                    f"± {formatNumber(row['Std_Test_R2'])}), MAE = {withUnit(row['Overall_Test_MAE'], code)}, "
                    f"RMSE = {withUnit(row['Overall_Test_RMSE'], code)}; "
                    f"{100 * (1 - row['CI_Test_Outside_Frac']):.1f}% of molecules inside the 95% interval.")
        if fitted is not None and code in fitted.index:
            sentence += (f" For comparison, the fitted GC baseline has test RMSE = "
                         f"{withUnit(fitted.loc[code, 'RMSE_Test'], code)} on its own 80/20 split of the JRGC set.")
        if isinstance(row.get("Dropped_Folds"), str) and row["Dropped_Folds"]:
            sentence += f" Dropped folds: {row['Dropped_Folds']}."
        lines.append(sentence)
    return "\n".join(lines) + "\n"


def describeCoverage(coverageDf):
    """Sentences for notebook 05 (coverage of the 95% intervals)."""
    lines = [header("Coverage of the 95% prediction intervals: results")]
    under, over = [], []
    for code, row in coverageDf.iterrows():
        low, high = row["Test 95% CI low"], row["Test 95% CI high"]
        verdict = "consistent with 95%"
        if high < 95:
            verdict, _ = "below 95% (intervals too narrow)", under.append(code)
        elif low > 95:
            verdict, _ = "above 95% (intervals wider than needed)", over.append(code)
        lines.append(f"- **{label(code)}**: {row['Test']:.1f}% of {int(row['nTestMolecules'])} test molecules "
                     f"inside the interval (95% CI {low:.1f}–{high:.1f}%), {verdict}; training coverage "
                     f"{row['Train']:.1f}%.")
    if under:
        lines.append(f"\nTest coverage is significantly below 95% for {', '.join(label(c) for c in under)}.")
    if over:
        lines.append(f"\nTest coverage is significantly above 95% for {', '.join(label(c) for c in over)}.")
    if not under and not over:
        lines.append("\nFor every property the test coverage is statistically consistent with 95%.")
    return "\n".join(lines) + "\n"


def describeMlComparison(resultsDf):
    """Sentences for notebook 06 (GP versus standard ML models)."""
    lines = [header("GP versus standard machine-learning models: results")]
    for runName, group in resultsDf.groupby("Property", sort=False):
        code = codeFromRun(runName)
        group = group.set_index("Model")
        best = group["Overall_Test_R2"].idxmax()
        gpText = ""
        if "GP" in group.index:
            gp = group.loc["GP"]
            gpText = (f"GP test R² = {formatNumber(gp['Overall_Test_R2'])}, "
                      f"RMSE = {withUnit(gp['Overall_Test_RMSE'], code)}; ")
        lines.append(f"- **{label(code)}**: {gpText}best model is {best} "
                     f"(test R² = {formatNumber(group.loc[best, 'Overall_Test_R2'])}, "
                     f"RMSE = {withUnit(group.loc[best, 'Overall_Test_RMSE'], code)}).")
    gpBest = sum(1 for _, g in resultsDf.groupby("Property")
                 if g.set_index("Model")["Overall_Test_R2"].idxmax() == "GP")
    lines.append(f"\nThe GP has the highest test R² for {gpBest} of {resultsDf['Property'].nunique()} properties.")
    return "\n".join(lines) + "\n"


def describeMoleculeComparison(summaryDf, comparisonDf, jrgcSource="raw", topN=20, nExamples=3):
    """Sentences for notebook 07 (molecule-level JRGC versus GC-GP comparison)."""
    source = "raw group-contribution estimate" if jrgcSource == "raw" else "fitted linear GC baseline"
    topText = f"{100 * topN:g}%" if 0 < topN < 1 else str(int(topN))
    lines = [header("Molecules where the JRGC and GC-GP predictions differ: results"),
             f"JRGC here is the {source}. GC-GP values are out-of-fold predictions from the 10-fold "
             f"cross-validation, so each molecule is predicted by a model that did not see it. "
             f"\"Poorly predicted\" means among the {topText} largest absolute errors of that model "
             f"for the property. Each structure is counted once: entries with the same structure "
             f"(InChIKey) are merged and their values averaged.\n"]
    for _, row in summaryDf.iterrows():
        code = row["Property"]
        group = comparisonDf[comparisonDf["Property"] == code].sort_values("Difference_Rank")
        examples = ", ".join(
            f"{r['Canonical_SMILES']} (exp {formatNumber(r['Experimental'])}, JRGC {formatNumber(r['JRGC'])}, "
            f"GC-GP {formatNumber(r['GCGP'])})" for _, r in group.head(nExamples).iterrows())
        merged = ""
        if row.get("N_Merged_Structures", 0):
            merged = (f", from {int(row['N_Entries'])} entries; {int(row['N_Merged_Structures'])} structures "
                      f"appeared more than once and were merged, {int(row['N_Isomer_Groups'])} of them with "
                      f"different experimental values")
        lines.append(
            f"- **{label(code)}** ({int(row['N_Molecules'])} unique structures{merged}): "
            f"MAE {withUnit(row['JRGC_MAE'], code)} "
            f"for JRGC and {withUnit(row['GCGP_MAE'], code)} for GC-GP; GC-GP is closer to experiment for "
            f"{100 * row['Frac_GCGP_Closer']:.1f}% of the molecules. The two predictions differ by "
            f"{withUnit(row['Median_Abs_Prediction_Difference'], code)} at the median and by up to "
            f"{withUnit(row['Max_Abs_Prediction_Difference'], code)}. "
            f"{int(row['N_Poorly predicted by both'])} molecules are poorly predicted by both models, "
            f"{int(row['N_Poor JRGC, corrected by GC-GP'])} are poor for JRGC but not for GC-GP, and "
            f"{int(row['N_Poor GC-GP only'])} are poor for GC-GP only. Largest differences: {examples}.")
    return "\n".join(lines) + "\n"


def describeCuration(summaryDf):
    """Sentences for notebook 00 (data curation)."""
    # The curation always uses the complete input files, so no fast-run note is needed here.
    lines = ["## Data curation: results\n"]
    for _, row in summaryDf.iterrows():
        code = row["Property"]
        parts = []
        if row["Removed_range"]:
            parts.append(f"{int(row['Removed_range'])} with an experimental value outside {row['Range']} "
                         f"{PROPERTIES[code]['unit']}".rstrip())
        if row["Removed_GC_error"]:
            parts.append(f"{int(row['Removed_GC_error'])} with a GC error outside the 1.5 × IQR fences "
                         f"{row['GC_error_fences']}")
        if row["Removed_duplicates"]:
            parts.append(f"{int(row['Removed_duplicates'])} repeated entries of a structure already in the set")
        removedText = "; removed: " + "; ".join(parts) if parts else "; all usable molecules kept"
        lines.append(f"- **{label(code)}**: {int(row['Usable_rows'])} usable entries of {int(row['Rows_in_file'])} "
                     f"in the input file{removedText}. Curated dataset: {int(row['Curated_rows'])} molecules.")
    return "\n".join(lines) + "\n"
