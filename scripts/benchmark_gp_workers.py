"""Time the log P GP fit with different numbers of worker processes.

Fits the same restarts of one cross-validation fold of log P with 1, 2 and 4
workers (or the numbers given with --workers), prints the run time of each,
and checks that every setting gives bit-for-bit the same restarts and
predictions. Use it to choose GCGP_N_WORKERS for this computer.

    .venv/bin/python scripts/benchmark_gp_workers.py
    .venv/bin/python scripts/benchmark_gp_workers.py --workers 1 4 --restarts 4 --rows 1500

Run 00_Data_Curation.ipynb first. Checkpoints are written to a temporary
folder and never read back, so every setting really fits the GP, and the
project's Output_Data/checkpoints/ folder is not touched.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

repoRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot / "src"))


def parseArguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4],
                        help="worker counts to compare (default: 1 2 4)")
    parser.add_argument("--restarts", type=int, default=4, help="restarts to fit (default: 4)")
    parser.add_argument("--rows", type=int, default=None,
                        help="use only the first ROWS molecules of the fold, for a quicker test")
    return parser.parse_args()


def main():
    args = parseArguments()
    curatedDir = Path(os.environ.get("GCGP_OUTPUT_DIR", repoRoot / "Output_Data")) / "curated_data"
    os.environ["GCGP_FAST_RUN"] = "0"
    os.environ["GCGP_RESUME"] = "0"
    os.environ["GCGP_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="gcgp_benchmark_")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

    from gcgp.finalConfig import getGpConfig
    from gcgp.gpModels import fitAndPredict, shutdownRestartPool
    from gcgp.properties import PROPERTIES
    from gcgp.settings import GLOBAL_SEED

    curatedFile = curatedDir / "logP_curated.csv"
    if not curatedFile.is_file():
        sys.exit(f"{curatedFile} not found. Run notebooks/00_Data_Curation.ipynb first.")
    info = PROPERTIES["logP"]
    data = pd.read_csv(curatedFile, float_precision="round_trip")
    X = data[info["featureCols"]].to_numpy(dtype=float)
    y = data[info["targetCol"]].to_numpy(dtype=float).reshape(-1, 1)
    trainIdx, testIdx = next(KFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED).split(X))
    if args.rows:
        trainIdx = trainIdx[:args.rows]
    print(f"log P, fold 1: {len(trainIdx)} training molecules, {args.restarts} restarts, "
          f"{os.cpu_count()} CPU cores")

    reference, timings = None, []
    for nWorkers in args.workers:
        os.environ["GCGP_N_WORKERS"] = str(nWorkers)
        start = time.time()
        fit = fitAndPredict(X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], getGpConfig("logP"), args.restarts)
        seconds = time.time() - start
        shutdownRestartPool()
        if fit.get("failed"):
            sys.exit(f"The fit failed with {nWorkers} workers: {fit['reason']}")
        result = {"losses": [r["loss"] for r in fit["restarts"]],
                  "states": [r["state"] for r in fit["restarts"]],
                  "prediction": fit["yTestPred"], "interval": fit["yTestCi"]}
        if reference is None:
            reference, identical = result, True
        else:
            identical = (result["losses"] == reference["losses"]
                         and all(np.array_equal(a, b) for sa, sb in zip(result["states"], reference["states"])
                                 for a, b in zip(sa, sb))
                         and np.array_equal(result["prediction"], reference["prediction"])
                         and np.array_equal(result["interval"], reference["interval"]))
        timings.append((nWorkers, seconds, identical))
        print(f"  {nWorkers} worker(s): {seconds:7.1f} s   identical to {args.workers[0]} worker(s): {identical}")

    base = timings[0][1]
    print("\nWorkers  Time (s)  Speed-up  Identical")
    for nWorkers, seconds, identical in timings:
        print(f"{nWorkers:7d}  {seconds:8.1f}  {base / seconds:8.2f}  {identical}")
    if not all(identical for _, _, identical in timings):
        print("\nSome settings did not give identical results on this computer. Keep GCGP_N_WORKERS=1.")


if __name__ == "__main__":
    main()
