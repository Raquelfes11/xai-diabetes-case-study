"""Helpers for global explanations: grouped permutation importance and 1-D ALE."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def permutation_importance_groups(model, X, y, groups, n_repeats=10, seed=0, metric=roc_auc_score):
    """Drop in `metric` when the columns of each group are permuted *together*.

    groups: dict name -> list of columns. Returns DataFrame with mean/std over repeats.
    Permuting the rows of a whole group jointly keeps the within-group correlation intact,
    which avoids the dilution that happens when correlated features are permuted one by one.
    """
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            drops.append(base - metric(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"group": name, "n_features": len(cols), "mean": np.mean(drops), "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def ale_1d(model, X, feature, n_bins=20, logit=True):
    """First-order ALE of one feature (quantile bins, centred). Returns (grid, effect)."""
    x = X[feature].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

    def f(Z):
        p = np.clip(model.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)) if logit else p

    local = np.zeros(len(edges) - 1); counts = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            continue
        lo, hi = X[m].copy(), X[m].copy()
        lo[feature], hi[feature] = edges[k], edges[k + 1]
        local[k] = (f(hi) - f(lo)).mean(); counts[k] = m.sum()
    acc = np.concatenate([[0], np.cumsum(local)])
    # centre using the data distribution
    mid = (acc[:-1] + acc[1:]) / 2
    acc = acc - (mid * counts).sum() / counts.sum()
    return edges, acc
