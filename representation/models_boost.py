"""XGBoost hypothesis class on the same cached feature packs as SVM."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from evaluation.metrics import classification_metrics
from evaluation.plots import save_and_show
from optimization.loop import LearnerLoop, sample_weights
from representation.constants import RANDOM_STATE
from representation.features import FeaturePack, scale_pack


class XGBoostPipeline(LearnerLoop):
    """XGBoost on the same scaled matrices as SVM."""

    def __init__(self, pack: FeaturePack, name: str, results_dir: str, figures_dir: str):
        super().__init__(name, results_dir, figures_dir)
        self.pack = pack
        self.scaled: Optional[FeaturePack] = None
        self.model = None
        self.search_table: List[Dict] = []

    def ingest(self) -> None:
        """Scale features with a scaler fit on train only."""
        self.scaled, _ = scale_pack(self.pack)
        self.y_train = self.pack.y_train

    def fit(self) -> None:
        """Train a small default booster."""
        import xgboost as xgb

        assert self.scaled is not None
        n_classes = int(np.max(self.scaled.y_train) + 1)
        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
            tree_method="hist",
        )
        sw = sample_weights(self.scaled.y_train)
        self.model.fit(self.scaled.X_train, self.scaled.y_train, sample_weight=sw)
        pred_val = self.model.predict(self.scaled.X_val)
        self.val_metrics = classification_metrics(self.scaled.y_val, pred_val)

    def search(self) -> None:
        """Small grid scored on the validation set."""
        import xgboost as xgb

        assert self.scaled is not None
        n_classes = int(np.max(self.scaled.y_train) + 1)
        sw = sample_weights(self.scaled.y_train)
        best = {"acc": -1.0, "params": {}}
        self.search_table = []
        for n_est in (200, 500):
            for depth in (3, 5, 7):
                for lr in (0.05, 0.1):
                    clf = xgb.XGBClassifier(
                        n_estimators=n_est, max_depth=depth, learning_rate=lr,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=n_classes,
                        eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
                        tree_method="hist",
                    )
                    clf.fit(self.scaled.X_train, self.scaled.y_train, sample_weight=sw)
                    acc = float(accuracy_score(self.scaled.y_val, clf.predict(self.scaled.X_val)))
                    params = {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}
                    self.search_table.append({**params, "val_acc": acc})
                    if acc > best["acc"]:
                        best = {"acc": acc, "params": params}
        self.best_params = best["params"]
        self._plot_search()

    def _plot_search(self) -> None:
        import matplotlib.pyplot as plt

        if not self.search_table:
            return
        df = pd.DataFrame(self.search_table)
        fig, ax = plt.subplots(figsize=(8, 4))
        for depth, g in df.groupby("max_depth"):
            g2 = g.groupby("n_estimators")["val_acc"].mean()
            ax.plot(g2.index, g2.values, marker="o", label=f"max_depth={depth}")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("val accuracy")
        ax.set_title(f"XGBoost val accuracy — {self.name}")
        ax.legend()
        fig.tight_layout()
        save_and_show(fig, f"xgb_tune_{self.name}.png", self.figures_dir)

    def refit(self) -> None:
        """Refit the best booster on train."""
        import xgboost as xgb

        assert self.scaled is not None
        n_classes = int(np.max(self.scaled.y_train) + 1)
        params = dict(self.best_params)
        self.model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
            tree_method="hist",
        )
        sw = sample_weights(self.scaled.y_train)
        self.model.fit(self.scaled.X_train, self.scaled.y_train, sample_weight=sw)
        self.y_true_test = self.scaled.y_test
        self.y_pred_test = self.model.predict(self.scaled.X_test)
        self.y_pred_val = self.model.predict(self.scaled.X_val)
        self.proba_test = self.model.predict_proba(self.scaled.X_test)
        self.val_metrics = classification_metrics(self.scaled.y_val, self.y_pred_val)
