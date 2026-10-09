#!/usr/bin/env bash
# Run every notebook from top to bottom on a fresh kernel, in numerical order.
#
#   bash scripts/run_notebooks.sh            full run (final settings; slow)
#   bash scripts/run_notebooks.sh --fast     quick check on small data subsets
#   bash scripts/run_notebooks.sh --fresh    ignore saved GP checkpoints and refit everything
#   bash scripts/run_notebooks.sh --workers 4   fit GP restarts in 4 parallel processes
#   bash scripts/run_notebooks.sh --fast 01 04 05   only notebooks whose names start with these numbers
#
# Finished GP restarts are saved in Output_Data/checkpoints/, so a run that was
# interrupted continues where it stopped when the script is started again.
# The number of workers changes only the run time, not the results.
#
# The notebooks in notebooks/ are left unchanged. Executed copies, with all
# outputs and figures, are saved in Output_Data/executed_notebooks/.
set -euo pipefail

repoRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repoRoot"

if [ ! -x .venv/bin/python ]; then
    echo "The .venv environment is missing. Run first:  bash scripts/setup_env.sh"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export GCGP_FAST_RUN=0
export GCGP_RESUME="${GCGP_RESUME:-1}"
export GCGP_N_WORKERS="${GCGP_N_WORKERS:-1}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fast) export GCGP_FAST_RUN=1; shift
                echo "Fast run: small data subsets, numbers are not final results." ;;
        --fresh) export GCGP_RESUME=0; shift
                 echo "Fresh run: saved checkpoints are ignored." ;;
        --workers) export GCGP_N_WORKERS="$2"; shift 2
                   echo "GP restarts are fitted in $GCGP_N_WORKERS parallel processes." ;;
        *) break ;;
    esac
done

outDir="Output_Data/executed_notebooks"
mkdir -p "$outDir"
logFile="$outDir/run_log.txt"
echo "Run started $(date)  fast=$GCGP_FAST_RUN  resume=$GCGP_RESUME  workers=$GCGP_N_WORKERS" > "$logFile"

failed=0
for notebook in notebooks/*.ipynb; do
    name="$(basename "$notebook")"
    if [ "$#" -gt 0 ]; then
        selected=0
        for prefix in "$@"; do
            case "$name" in "$prefix"*) selected=1 ;; esac
        done
        [ "$selected" -eq 1 ] || continue
    fi
    start=$(date +%s)
    if jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
            --ExecutePreprocessor.kernel_name=gcgp-amines \
            --output-dir "$outDir" "$notebook" >> "$logFile" 2>&1; then
        status="OK"
    else
        status="FAILED (see $logFile)"
        failed=$((failed + 1))
    fi
    echo "$name: $status ($(( $(date +%s) - start )) s)" | tee -a "$logFile"
done

echo "Executed notebooks are in $outDir"
exit "$failed"
