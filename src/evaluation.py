"""Metrics helpers shared by the notebooks."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y_true, p):
    """Threshold that maximises F1 (choose it on the validation set, never on test)."""
    ts = np.linspace(0.05, 0.95, 91)
    return ts[int(np.argmax([f1_score(y_true, p >= t) for t in ts]))]


def report(y_true, p, thr):
    return {
        "ROC-AUC": roc_auc_score(y_true, p),
        "PR-AUC": average_precision_score(y_true, p),
        "Brier": brier_score_loss(y_true, p),
        "F1": f1_score(y_true, p >= thr),
        "Precision": precision_score(y_true, p >= thr),
        "Recall": recall_score(y_true, p >= thr),
    }
