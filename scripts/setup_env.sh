#!/usr/bin/env bash
# Create the gcgp-amines virtual environment in .venv/ and register its Jupyter kernel.
# Run from anywhere:  bash scripts/setup_env.sh
set -euo pipefail

repoRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repoRoot"

# TensorFlow 2.16 needs Python 3.10, 3.11 or 3.12, built for this machine's CPU.
# An x86_64 Python (for example an old Anaconda) on an Apple Silicon Mac runs
# under Rosetta, where TensorFlow fails with "compiled to use AVX instructions".
machineArch="$(uname -m)"
pythonBin=""
for candidate in /opt/homebrew/bin/python3.11 python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        pythonArch="$("$candidate" -c 'import platform; print(platform.machine())')"
        if [ "$pythonArch" != "$machineArch" ]; then
            continue
        fi
        case "$version" in
            3.10|3.11|3.12) pythonBin="$candidate"; break ;;
        esac
    fi
done
if [ -z "$pythonBin" ]; then
    echo "No $machineArch Python 3.10, 3.11 or 3.12 found. On macOS install one with:  brew install python@3.11"
    exit 1
fi
echo "Using $pythonBin ($("$pythonBin" --version), $machineArch)"

if [ -d .venv ]; then
    venvArch="$(.venv/bin/python -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
    if [ "$venvArch" != "$machineArch" ]; then
        echo "The existing .venv uses a $venvArch Python, but this machine is $machineArch."
        echo "Remove it with  rm -rf .venv  and run this script again."
        exit 1
    fi
    echo "Reusing the existing .venv folder."
else
    "$pythonBin" -m venv --prompt gcgp-amines .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gcgp-amines --display-name "Python 3 (gcgp-amines)"

# Install the VS Code settings, extension recommendations and tasks, unless the
# folder already has its own .vscode settings.
if [ ! -d .vscode ]; then
    mkdir -p .vscode
    cp scripts/vscode/*.json .vscode/
    echo "VS Code settings installed in .vscode/"
fi

echo "Checking the installation..."
python - <<'PY'
import sys
sys.path.insert(0, "src")
import gpflow, tensorflow as tf, numpy, pandas, sklearn, xgboost, skmultilearn
import gcgp.settings as s
print(f"Python {sys.version.split()[0]}, TensorFlow {tf.__version__}, GPflow {gpflow.__version__}")
s.checkInputs(("model_building", "raw", "reference_results"))
print("All input files found in", s.INPUT_DIR)
PY

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
echo "In VS Code, pick the kernel 'Python 3 (gcgp-amines)' when you open a notebook."
