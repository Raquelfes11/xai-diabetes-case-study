"""Download and load the CDC Diabetes Health Indicators dataset (UCI id 891)."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cdc_diabetes.csv"
TARGET = "Diabetes_binary"


def download(force: bool = False) -> Path:
    """Fetch the dataset from the UCI repository and cache it as a CSV."""
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=891)
    df = ds.data.features.copy()
    df[TARGET] = ds.data.targets[TARGET]
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return DATA_PATH


def load() -> pd.DataFrame:
    return pd.read_csv(download())


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df[TARGET].value_counts(normalize=True))
