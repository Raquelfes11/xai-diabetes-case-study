#!/usr/bin/env bash
# Runs the whole project from scratch: downloads the data and executes every notebook in order.
# Notebook 02 creates models/ (needed by 03-07, 09, 10); notebook 06 creates the reduced model (needed by 07, 09, 10).
# Approximate time on a laptop CPU: 25-40 minutes (notebooks 03, 04 and 05 are the slowest).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.data
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  echo ">>> running $nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=3600
done
echo "Done. Figures are in figures/, executed notebooks in notebooks/."
