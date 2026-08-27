"""Classification scores used by the learner loop."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from representation.constants import STRESS_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy and F1 / precision / recall (macro and weighted)."""
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def target_names_for(y: np.ndarray) -> List[str]:
    """Class names covering every label that appears in y."""
    return [STRESS_NAMES.get(int(i), f"Class {i}") for i in sorted(np.unique(y))]
