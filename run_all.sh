#!/usr/bin/env bash
# Runs the whole project from scratch: downloads the data and executes every notebook in order.
# Notebook 02 creates models/ (needed by 03-07, 09-11); notebook 06 creates the reduced model (needed by 07, 09-12).
# Approximate time on a laptop CPU: 25-40 minutes (notebooks 03, 04 and 05 are the slowest).
set -euo pipefail
cd "$(dirname "$0")"
# LightGBM needs the OpenMP runtime on macOS; fail early with a clear hint
if ! python -c "import lightgbm" 2>/dev/null; then
  echo "ERROR: LightGBM could not be imported. On macOS install OpenMP first: 'brew install libomp' (or 'conda install -c conda-forge llvm-openmp')." >&2
  exit 1
fi
python -m src.data
# Register a kernel for THIS environment so the notebooks never run with another Python (e.g. conda base)
python -m ipykernel install --sys-prefix --name xai-venv --display-name xai-venv >/dev/null
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  echo ">>> running $nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=xai-venv
done
echo "Done. Figures are in figures/, executed notebooks in notebooks/."
