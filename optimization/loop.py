"""Shared grading loop: ingest → fit → search → refit → report."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from evaluation.metrics import classification_metrics, target_names_for
from evaluation.plots import save_and_show
from representation.constants import STRESS_NAMES


class LearnerLoop(ABC):
    """Shared grading loop: ingest, fit, search, refit, report. Subclasses do the work."""

    def __init__(self, name: str, results_dir: str, figures_dir: str, class_names: Optional[List[str]] = None):
        self.name = name
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.class_names = class_names or []
        self.y_pred_test: Optional[np.ndarray] = None
        self.y_true_test: Optional[np.ndarray] = None
        self.y_pred_val: Optional[np.ndarray] = None
        self.best_params: Dict = {}
        self.history: Dict = {}
        self.test_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.proba_test: Optional[np.ndarray] = None

    @abstractmethod
    def ingest(self) -> None:
        """Prepare model-specific inputs (scaling, image batches, MixUp, ...)."""

    @abstractmethod
    def fit(self) -> None:
        """Train a default / baseline config."""

    @abstractmethod
    def search(self) -> None:
        """Tune hyperparameters on validation only."""

    @abstractmethod
    def refit(self) -> None:
        """Lock the best hyperparameters and refit on train."""

    def report(self, y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> Dict[str, float]:
        """Write metrics, a classification report, and a confusion-matrix figure."""
        metrics = classification_metrics(y_true, y_pred)
        os.makedirs(self.results_dir, exist_ok=True)
        report_txt = classification_report(
            y_true, y_pred, target_names=target_names_for(y_true), zero_division=0
        )
        with open(os.path.join(self.results_dir, f"{tag}_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_txt)
            f.write("\n")
            f.write(json.dumps(metrics, indent=2))
        cm = confusion_matrix(y_true, y_pred)
        self._save_confusion(cm, tag, target_names_for(y_true))
        return metrics

    def _save_confusion(self, cm: np.ndarray, tag: str, names: List[str]) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[0])
        axes[0].set_title(f"{tag} counts")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[1])
        axes[1].set_title(f"{tag} row-normalised")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        fig.tight_layout()
        save_and_show(fig, f"cm_{tag}.png", self.figures_dir)

    def run(self) -> Dict[str, float]:
        """Run the five steps in order."""
        self.ingest()
        if not self.class_names:
            y = getattr(self, "y_train", None)
            if y is None:
                pack = getattr(self, "scaled", None) or getattr(self, "pack", None)
                if pack is not None:
                    y = getattr(pack, "y_train", None)
            if y is None:
                y = getattr(self, "y_true_test", None)
            if y is not None:
                n = int(np.max(y) + 1)
                self.class_names = [STRESS_NAMES[i] for i in range(n)]
        self.fit()
        self.search()
        self.refit()
        if self.y_true_test is None or self.y_pred_test is None:
            raise RuntimeError("refit() must set y_true_test and y_pred_test.")
        self.test_metrics = self.report(self.y_true_test, self.y_pred_test, self.name)
        return self.test_metrics


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights from sklearn balanced class weights."""
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    mapping = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([mapping[int(t)] for t in y], dtype=np.float32)


def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    """Keras-style class_weight mapping."""
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}
