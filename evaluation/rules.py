"""MixUp, residual Mixed rule, disease probe, ensemble, BRACOL labels."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

from representation.constants import RANDOM_STATE


def apply_mixup(
    images: np.ndarray,
    labels: np.ndarray,
    lam: float,
    index_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix each sample i with sample index_j[i] using weight lam."""
    images_m = lam * images + (1.0 - lam) * images[index_j]
    labels_m = lam * labels + (1.0 - lam) * labels[index_j]
    return images_m, labels_m


def apply_residual_rule(
    class_probs: np.ndarray,
    disease_probs: np.ndarray,
    tau: float,
    delta: float,
) -> np.ndarray:
    """
    Predict Mixed when two diseases fire or the 6-class top two are close.

    class_probs: (n, 6) softmax over predominant_stress.
    disease_probs: (n, 4) scores for miner, rust, phoma, cercospora.
    """
    argmax = np.argmax(class_probs, axis=1)
    part = np.partition(class_probs, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    n_hot = (disease_probs >= tau).sum(axis=1)
    pred = argmax.copy()
    mixed = (n_hot >= 2) | (margin < delta)
    pred[mixed] = 5
    return pred


class ResidualMixedRule:
    """Tune tau and delta on validation, then apply the Mixed residual rule."""

    def __init__(self, tau_grid=None, delta_grid=None):
        self.tau_grid = list(tau_grid or [0.30, 0.40, 0.50, 0.60])
        self.delta_grid = list(delta_grid or [0.05, 0.10, 0.15, 0.20])
        self.tau = 0.5
        self.delta = 0.10
        self.search_table: List[Dict] = []

    def search(self, y_true: np.ndarray, class_probs: np.ndarray, disease_probs: np.ndarray) -> Tuple[float, float]:
        """Pick tau and delta with the best macro F1 on validation."""
        best_f1 = -1.0
        self.search_table = []
        for tau in self.tau_grid:
            for delta in self.delta_grid:
                pred = apply_residual_rule(class_probs, disease_probs, tau, delta)
                f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
                acc = float(accuracy_score(y_true, pred))
                self.search_table.append({"tau": tau, "delta": delta, "macro_f1": f1, "acc": acc})
                if f1 > best_f1:
                    best_f1 = f1
                    self.tau, self.delta = tau, delta
        return self.tau, self.delta

    def predict(self, class_probs: np.ndarray, disease_probs: np.ndarray) -> np.ndarray:
        """Apply the locked rule."""
        return apply_residual_rule(class_probs, disease_probs, self.tau, self.delta)


def fit_disease_probe(X_train: np.ndarray, Y_train: np.ndarray, X_other: np.ndarray) -> np.ndarray:
    """
    Multi-label logistic probe for miner/rust/phoma/cercospora.

    Y_train is (n, 4) binary. Returns probabilities for X_other.
    """
    from sklearn.linear_model import LogisticRegression

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    )
    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_other)
    if isinstance(proba, list):
        return np.column_stack([p[:, -1] for p in proba]).astype(np.float32)
    return np.asarray(proba, dtype=np.float32)


def load_bracol_labels(bracol_dir: str) -> Dict[int, List[int]]:
    """Map original image id -> list of YOLO class ids."""
    bracol_data: Dict[int, List[int]] = {}
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(bracol_dir, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            match = re.match(r"(\d+)_jpg", fname)
            if not match:
                continue
            img_id = int(match.group(1))
            detections = []
            with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        detections.append(int(parts[0]))
            bracol_data[img_id] = detections
    return bracol_data


def ensemble_average(*probas: np.ndarray) -> np.ndarray:
    """Average probability tensors that share the same shape."""
    arrays = [np.asarray(p) for p in probas]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            "ensemble_average needs equal shapes (same split/cache); "
            f"got {[a.shape for a in arrays]}"
        )
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0)
