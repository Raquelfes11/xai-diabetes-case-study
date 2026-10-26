# XAI case study: diabetes risk prediction

Final project for *Explainable AI* (Máster en IA). End-to-end case study that trains a
model to predict diabetes/prediabetes from survey data and uses explanations to improve
the model and derive actionable insights.

**Dataset:** [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(UCI, BRFSS survey, ~254k people, 21 features).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data      # downloads the dataset into data/
```

## Structure

```
src/         reusable code (data loading, models, XAI helpers)
notebooks/   analysis notebooks
figures/     plots used in the report
data/        downloaded data (not versioned)
```

_Work in progress._
