"""Outlier rule used by the data curation (notebook 00).

``filterByGcError`` first keeps a fixed range of experimental values, then
drops molecules whose GC error (experimental minus GC estimate) is outside the
IQR fences. It is applied only in ``gcgp.curation``; the modelling notebooks
read the curated datasets.
"""

import pandas as pd


def filterByGcError(df, targetCol, gcCol, targetRange=None, iqrMultiplier=None,
                    maxAbsError=None, maxPctError=None, zThreshold=None):
    """Remove molecules by target range and by size of the GC error.

    Filters are applied in this order, and each one only looks at the rows
    that survived the previous filters:

    1. ``targetRange``: keep ``low <= experimental <= high``.
    2. ``maxAbsError``: keep ``|experimental - GC| <= maxAbsError``.
    3. ``maxPctError``: keep ``|experimental - GC| / |experimental| * 100 <= maxPctError``.
    4. ``iqrMultiplier``: IQR fences on ``experimental - GC``.
    5. ``zThreshold``: z-score on ``experimental - GC``.

    Returns ``(filteredDf, keepMask, report)``; ``report`` is a small table of
    how many rows each step removed.
    """
    work = df.copy()
    gcError = work[targetCol] - work[gcCol]
    absError = gcError.abs()
    pctError = absError / work[targetCol].abs() * 100

    keep = pd.Series(True, index=work.index)
    steps = [{"Step": "Start", "Removed": 0, "Remaining": len(work)}]

    def applyStep(name, stepMask):
        nonlocal keep
        removed = int((~stepMask & keep).sum())
        keep &= stepMask
        steps.append({"Step": name, "Removed": removed, "Remaining": int(keep.sum())})

    if targetRange is not None:
        low, high = targetRange
        applyStep(f"Target in [{low}, {high}]", (work[targetCol] >= low) & (work[targetCol] <= high))
    if maxAbsError is not None:
        applyStep(f"|GC error| <= {maxAbsError}", absError <= maxAbsError)
    if maxPctError is not None:
        applyStep(f"|GC error| <= {maxPctError}%", pctError <= maxPctError)
    if iqrMultiplier is not None:
        q1 = gcError[keep].quantile(0.25)
        q3 = gcError[keep].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - iqrMultiplier * iqr, q3 + iqrMultiplier * iqr
        applyStep(f"GC error in IQR fences [{low:.2f}, {high:.2f}]", (gcError >= low) & (gcError <= high))
    if zThreshold is not None:
        subset = gcError[keep]
        z = ((gcError - subset.mean()) / subset.std()).abs()
        applyStep(f"|z(GC error)| <= {zThreshold}", z <= zThreshold)

    filtered = work.loc[keep].reset_index(drop=True)
    return filtered, keep, pd.DataFrame(steps)
