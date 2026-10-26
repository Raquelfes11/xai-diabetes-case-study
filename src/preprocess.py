"""Cleaning and train/validation/test split (70/15/15, stratified, fixed seed).

Decisions (see project diary):
- Exact duplicate rows are dropped *before* splitting, so the same profile cannot
  appear in both train and test and inflate the metrics.
- No resampling / class weights: the 14-15 % positive rate is handled through
  threshold selection on the validation set and imbalance-aware metrics.
"""
from sklearn.model_selection import train_test_split

from src.data import TARGET, load

SEED = 42


def get_splits(drop_duplicates: bool = True, seed: int = SEED):
    """Return X_train, X_val, X_test, y_train, y_val, y_test."""
    df = load()
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te
