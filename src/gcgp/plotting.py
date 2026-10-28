"""Figures used across the notebooks.

Colours follow the original figures: blue for training points, red for test
(held-out) points, dashed black line for perfect agreement.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import calculateMetrics

TRAIN_COLOR = "blue"
TEST_COLOR = "red"

plt.rcParams["legend.frameon"] = False


FIGURE_FORMATS = ("png", "pdf")
FIGURE_DPI = 600


def savePlot(fig, savePath, formats=None, dpi=FIGURE_DPI):
    """Save a figure as PNG and PDF at 600 dpi, plus any extra formats asked for.

    The PDF keeps lines and text as vector graphics, which is usually the best
    choice for a journal; the 600 dpi PNG is for documents that need a bitmap.
    """
    if savePath is None:
        return
    savePath = Path(savePath)
    savePath.parent.mkdir(parents=True, exist_ok=True)
    extensions = list(FIGURE_FORMATS) + [f for f in (formats or ()) if f not in FIGURE_FORMATS]
    for ext in extensions:
        fig.savefig(savePath.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")


def finishFigure(fig, show=True):
    """Show the figure in the notebook (optional) and free its memory."""
    if show:
        plt.show()
    plt.close(fig)


def addMetricsText(ax, stats, setName, color, corner, fontsize=12):
    """Write R², MAE and RMSE for one data set in a corner of the axes."""
    text = (f"$R^2_{{\\mathrm{{{setName}}}}} = {stats['R2']:.2f}$\n"
            f"$\\mathrm{{MAE}}_{{\\mathrm{{{setName}}}}} = {stats['MAE']:.2f}$\n"
            f"$\\mathrm{{RMSE}}_{{\\mathrm{{{setName}}}}} = {stats['RMSE']:.2f}$")
    if corner == "upper left":
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize,
                va="top", ha="left", color=color)
    else:
        ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=fontsize,
                va="bottom", ha="right", color=color)


def addStatsBox(ax, header, stats, fontsize=10):
    """Write a header line and R², RMSE, MAE and MAPE in the upper left of the axes."""
    text = (f"{header}\nR² = {stats['R2']:.3f}, RMSE = {stats['RMSE']:.2f}\n"
            f"MAE = {stats['MAE']:.2f}, MAPE = {stats['MAPE']:.1f}%")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize, va="top", ha="left")


def plotSingleSplitResults(yTrainTrue, yTrainPred, yTestTrue, yTestPred,
                           yTrainCi, yTestCi, propertyName, savePath=None, show=True):
    """Three parity panels (train, test, both) with 95% error bars.

    Used by the Model 1 to Model 12 notebooks, which fit one 80/20 split.
    """
    yTrainTrue, yTrainPred, yTrainCi = (np.ravel(a) for a in (yTrainTrue, yTrainPred, yTrainCi))
    yTestTrue, yTestPred, yTestCi = (np.ravel(a) for a in (yTestTrue, yTestPred, yTestCi))
    yAllTrue = np.concatenate([yTrainTrue, yTestTrue])
    yAllPred = np.concatenate([yTrainPred, yTestPred])
    yAllCi = np.concatenate([yTrainCi, yTestCi])

    trainStats = calculateMetrics(yTrainTrue, yTrainPred)
    testStats = calculateMetrics(yTestTrue, yTestPred)
    allStats = calculateMetrics(yAllTrue, yAllPred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], "Training set", yTrainTrue, yTrainPred, yTrainCi, TRAIN_COLOR, trainStats),
        (axes[1], "Test set", yTestTrue, yTestPred, yTestCi, TEST_COLOR, testStats),
    ]
    for ax, title, yt, yp, ci, color, stats in panels:
        ax.scatter(yt, yp, alpha=0.6, s=30, color=color, label=title)
        ax.errorbar(yt, yp, yerr=ci, fmt="none", alpha=0.3, color=color, label="95% interval")
        low, high = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([low, high], [low, high], "k--", alpha=0.7, linewidth=2, label="Parity")
        addStatsBox(ax, f"{title} (n={len(yt)})", stats)
        ax.legend(loc="lower right", fontsize=9)

    ax = axes[2]
    ax.scatter(yTrainTrue, yTrainPred, alpha=0.5, s=25, color=TRAIN_COLOR, label="Training")
    ax.scatter(yTestTrue, yTestPred, alpha=0.5, s=25, color=TEST_COLOR, label="Test")
    ax.errorbar(yAllTrue, yAllPred, yerr=yAllCi, fmt="none", alpha=0.2, color="gray", label="95% interval")
    low, high = min(yAllTrue.min(), yAllPred.min()), max(yAllTrue.max(), yAllPred.max())
    ax.plot([low, high], [low, high], "k--", alpha=0.7, linewidth=2, label="Parity")
    addStatsBox(ax, f"Combined (n={len(yAllTrue)})", allStats)
    ax.legend(loc="lower right", fontsize=9)

    for ax in axes:
        ax.set_xlabel(f"Experimental {propertyName}")
        ax.set_ylabel(f"Predicted {propertyName}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)
    return trainStats, testStats, allStats


def plotParity(yTrainTrue, yTrainPred, yTestTrue, yTestPred, xLabel, yLabel, title,
               trainStats=None, testStats=None, savePath=None, figsize=(8, 7),
               trainLabel="Train", testLabel="Test", fontsize=12, formats=None, show=True):
    """Single parity plot with train (blue) and test (red) points.

    Used for the cross-validation results and the JRGC baseline.
    """
    yTrainTrue, yTrainPred = np.ravel(yTrainTrue), np.ravel(yTrainPred)
    yTestTrue, yTestPred = np.ravel(yTestTrue), np.ravel(yTestPred)
    trainStats = trainStats or calculateMetrics(yTrainTrue, yTrainPred)
    testStats = testStats or calculateMetrics(yTestTrue, yTestPred)

    allValues = np.concatenate([yTrainTrue, yTrainPred, yTestTrue, yTestPred])
    low, high = allValues.min(), allValues.max()
    margin = 0.05 * (high - low)
    low, high = low - margin, high + margin

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(yTrainTrue, yTrainPred, s=20, color=TRAIN_COLOR, zorder=2, label=trainLabel)
    ax.scatter(yTestTrue, yTestPred, s=30, color=TEST_COLOR, zorder=3, label=testLabel)
    ax.plot([low, high], [low, high], "k--", linewidth=2, zorder=1, label="Parity")

    addMetricsText(ax, trainStats, "Train", TRAIN_COLOR, "upper left", fontsize)
    addMetricsText(ax, testStats, "Test", TEST_COLOR, "lower right", fontsize)

    ax.set_xlabel(xLabel, fontsize=fontsize, fontweight="bold")
    ax.set_ylabel(yLabel, fontsize=fontsize, fontweight="bold")
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62), fontsize=fontsize - 2, frameon=False)
    fig.tight_layout()
    savePlot(fig, savePath, formats=formats)
    finishFigure(fig, show)
    return trainStats, testStats


def plotFoldR2(foldMetricsDf, trainOverall, testOverall, title, savePath=None, show=True):
    """Bar chart of train and test R² for each CV fold."""
    foldIds = foldMetricsDf["Fold"].to_numpy()
    positions = np.arange(len(foldIds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions - width / 2, foldMetricsDf["Train_R2"], width, color=TRAIN_COLOR, label="Train R²")
    ax.bar(positions + width / 2, foldMetricsDf["Test_R2"], width, color=TEST_COLOR, label="Test R²")
    ax.axhline(trainOverall["R2"], color=TRAIN_COLOR, linestyle="--",
               label=f"Pooled train R² = {trainOverall['R2']:.3f}")
    ax.axhline(testOverall["R2"], color=TEST_COLOR, linestyle="--",
               label=f"Pooled test R² = {testOverall['R2']:.3f}")
    ax.set_xticks(positions)
    ax.set_xticklabels(foldIds)
    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("R²", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)


def plotCoverageBars(coverageDf, labels, savePath=None, brokenAxis=True, show=True):
    """Grouped bars of the percent of points inside the 95% interval.

    ``coverageDf`` needs ``Train`` and ``Test`` columns (percent) in plotting
    order. With ``brokenAxis=True`` the figure uses a broken y axis (upper
    panel 80-108%, lower panel 0-10%), as in the original figure. Set it to False for one
    continuous axis, which is easier to read.
    """
    x = np.arange(len(labels))
    width = 0.28
    trainVals = coverageDf["Train"].to_numpy()
    testVals = coverageDf["Test"].to_numpy()

    if not brokenAxis:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        trainBars = ax.bar(x - width / 2, trainVals, width, color=TRAIN_COLOR, edgecolor="k", label="Train")
        testBars = ax.bar(x + width / 2, testVals, width, color=TEST_COLOR, edgecolor="k", label="Test")
        ax.axhline(95, color="k", linestyle=":", linewidth=1.5, label="Nominal 95%")
        ax.set_ylim(80, 108)
        ax.set_yticks(np.arange(80, 101, 5))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_ylabel("Points inside the 95% interval (%)", fontsize=12, fontweight="bold")
        for bar in list(trainBars) + list(testBars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}",
                    ha="center", va="bottom", rotation=90, fontsize=10)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))
        fig.tight_layout()
        savePlot(fig, savePath)
        finishFigure(fig, show)
        return

    fig, (axTop, axBottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 5),
                                          gridspec_kw={"height_ratios": [2.4, 1]})
    trainBars = axTop.bar(x - width / 2, trainVals, width, label="Train", color=TRAIN_COLOR, edgecolor="k")
    testBars = axTop.bar(x + width / 2, testVals, width, label="Test", color=TEST_COLOR, edgecolor="k")
    axBottom.bar(x - width / 2, trainVals, width, color=TRAIN_COLOR, edgecolor="k")
    axBottom.bar(x + width / 2, testVals, width, color=TEST_COLOR, edgecolor="k")

    axTop.set_ylim(80, 108)
    axTop.set_yticks(np.arange(80, 101, 5))
    axBottom.set_ylim(0, 10)
    axBottom.set_yticks(np.arange(0, 11, 5))
    axTop.spines.bottom.set_visible(False)
    axBottom.spines.top.set_visible(False)
    axTop.tick_params(axis="x", which="both", top=False, bottom=False, labeltop=False, labelbottom=False)
    axBottom.xaxis.tick_bottom()
    axBottom.set_xticks(x)
    axBottom.set_xticklabels(labels, fontsize=13, fontweight="bold")

    d = 0.010
    breakStyle = dict(transform=axTop.transAxes, color="black", clip_on=False)
    axTop.plot((-d, +d), (-d, +d), **breakStyle)
    axTop.plot((1 - d, 1 + d), (-d, +d), **breakStyle)
    breakStyle.update(transform=axBottom.transAxes)
    axBottom.plot((-d, +d), (1 - d, 1 + d), **breakStyle)
    axBottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **breakStyle)

    for bar in list(trainBars) + list(testBars):
        axTop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}",
                   ha="center", va="bottom", rotation=90, fontsize=12)

    fig.text(-0.03, 0.5, "Points inside the 95% interval (%)", va="center",
             rotation="vertical", fontsize=13, fontweight="bold")
    axTop.tick_params(axis="y", labelsize=12)
    axBottom.tick_params(axis="y", labelsize=12)
    axTop.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)


def modelColorMap(modelNames):
    """Give every model a fixed colour so it looks the same in every panel."""
    names = sorted(modelNames, key=lambda n: (len(n), n))
    palette = sns.color_palette("tab10" if len(names) <= 10 else "tab20", n_colors=len(names))
    return {name: palette[i] for i, name in enumerate(names)}


def plotMetricBars(ax, df, metric, colors, title, errors=None):
    """One bar per model for one metric, with the model name written in the bar.

    ``errors`` (optional, one value per bar) draws a black line of that half-length on each bar.
    """
    sns.barplot(x="Model", y=metric, data=df, palette=colors, hue="Model", legend=False, ax=ax, width=1)
    if errors is not None:
        ax.errorbar(np.arange(len(df)), df[metric].to_numpy(), yerr=np.asarray(errors, dtype=float),
                    fmt="none", ecolor="k", elinewidth=1.2, capsize=3)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    yMax = ax.get_ylim()[1]
    for bar, modelName in zip(ax.patches, df["Model"]):
        height = bar.get_height()
        textY = height / 2 if height > 0.15 * yMax else height + 0.02 * yMax
        ax.text(bar.get_x() + bar.get_width() / 2, textY, modelName,
                ha="center", va="center", rotation=90, fontsize=11, color="k")


def plotTestParityGrid(predictionTables, title, savePath=None, show=True, nCols=4):
    """Small test-set parity panels, one per property, with 95% error bars.

    ``predictionTables`` maps a property code to a table with columns
    ``Set``, ``Experimental``, ``Predicted`` and ``CI95`` (the format written
    by :func:`gcgp.kernelCv.cvTestPredictions`).
    """
    codes = list(predictionTables)
    nRows = int(np.ceil(len(codes) / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(4 * nCols, 3.8 * nRows), squeeze=False)
    for ax, code in zip(axes.flat, codes):
        test = predictionTables[code]
        test = test[test["Set"] == "Test"]
        stats = calculateMetrics(test["Experimental"], test["Predicted"])
        ax.errorbar(test["Experimental"], test["Predicted"], yerr=test["CI95"], fmt="o", ms=4,
                    color=TEST_COLOR, ecolor="gray", alpha=0.7, label="Test ± 95%")
        low = min(test["Experimental"].min(), test["Predicted"].min())
        high = max(test["Experimental"].max(), test["Predicted"].max())
        ax.plot([low, high], [low, high], "k--", linewidth=1.5, label="Parity")
        ax.text(0.02, 0.98, f"{code}\nR² = {stats['R2']:.2f}, RMSE = {stats['RMSE']:.2f}",
                transform=ax.transAxes, fontsize=10, va="top", ha="left")
        ax.set_xlabel("Experimental")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8, loc="lower right")
    for ax in list(axes.flat)[len(codes):]:
        ax.axis("off")
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)


def plotModelComparison(summaryDf, metric="Overall_Test_R2", savePath=None, show=True):
    """Grouped bars: one group per property run, one bar per model."""
    pivot = summaryDf.pivot(index="Property", columns="Model", values=metric)
    pivot = pivot[[m for m in ["GP", "SVR", "RandomForest", "XGBoost", "NeuralNetwork"] if m in pivot]]
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(pivot)), 4.5))
    pivot.plot.bar(ax=ax, width=0.8, edgecolor="k")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("Overall_", "Pooled ").replace("_", " "), fontsize=12, fontweight="bold")
    ax.legend(title="Model", fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    lowest = pivot.min().min()
    if lowest < -1:
        # Very poor fits (R² far below zero) would squash the other bars, so the axis stops at -1.
        ax.set_ylim(bottom=-1)
        ax.text(0.01, 0.02, f"Axis clipped at -1 (lowest value {lowest:.1f})", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    savePlot(fig, savePath)
    finishFigure(fig, show)
