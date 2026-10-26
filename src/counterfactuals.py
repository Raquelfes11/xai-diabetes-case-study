"""Exhaustive, constraint-aware counterfactual search for the screening model.

Only *actionable* features may change, and only in the healthy direction:
PhysActivity/Fruits/Veggies 0->1, Smoker/HighBP/HighChol 1->0, BMI can only go down
(never below 18.5, at most `max_bmi_drop` units). Age, Sex, Education, Income and
self-reported health variables are immutable by construction.

The search space is tiny (<= 2^6 x ~16 candidates), so we enumerate it and keep the valid
counterfactuals (prediction below the screening threshold), ranked by
sparsity (number of changed features) and then proximity (BMI drop).
"""
from itertools import product

import numpy as np
import pandas as pd

HEALTHY = {"PhysActivity": 1, "Fruits": 1, "Veggies": 1, "Smoker": 0, "HighBP": 0, "HighChol": 0}


def find_counterfactuals(model, x: pd.Series, thr: float, bmi_floor=18.5, max_bmi_drop=15, k=3):
    """Return up to k distinct counterfactuals for person `x` (a row of features)."""
    levers = [f for f, good in HEALTHY.items() if x[f] != good]
    bmi_drops = [0]
    if x["BMI"] > bmi_floor:
        top = int(min(max_bmi_drop, np.floor(x["BMI"] - bmi_floor)))
        bmi_drops = list(range(0, top + 1))
    rows, meta = [], []
    for flips in product([0, 1], repeat=len(levers)):
        for d in bmi_drops:
            if not any(flips) and d == 0:
                continue
            z = x.copy()
            for f, flip in zip(levers, flips):
                if flip:
                    z[f] = HEALTHY[f]
            z["BMI"] = x["BMI"] - d
            rows.append(z)
            meta.append((tuple(f for f, fl in zip(levers, flips) if fl), d))
    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows).astype(x.dtype if x.dtype != object else float)
    p = model.predict_proba(cand)[:, 1]
    out = pd.DataFrame({"changed": [m[0] for m in meta], "bmi_drop": [m[1] for m in meta], "p_after": p})
    out["n_changed"] = out["changed"].map(len) + (out["bmi_drop"] > 0)
    out = out[out.p_after < thr].sort_values(["n_changed", "bmi_drop", "p_after"])
    out = out.drop_duplicates("changed")          # diversity: one option per set of binary changes
    return out.head(k).reset_index(drop=True)
