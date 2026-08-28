"""SVM hypothesis class. Grids live here; they are the search attached to this model."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation.metrics import classification_metrics
from evaluation.plots import save_and_show
from optimization.compute import fmt_hms
from optimization.loop import LearnerLoop
from representation.features import FeaturePack, scale_pack


class SVMPipeline(LearnerLoop):
    """RBF (and linear) SVM on a scaled feature pack."""

    def __init__(
        self,
        pack: FeaturePack,
        name: str,
        results_dir: str,
        figures_dir: str,
        C_grid: Optional[Sequence[float]] = None,
        gamma_grid: Optional[Sequence] = None,
        need_proba: bool = False,
        search_linear_first: bool = False,
        skip_default_fit: bool = False,
    ):
        super().__init__(name, results_dir, figures_dir)
        self.pack = pack
        self.scaled: Optional[FeaturePack] = None
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.C_grid = list(C_grid or [0.1, 1, 10, 100, 1000])
        self.gamma_grid = list(gamma_grid or ["scale", 1e-4, 1e-3, 1e-2, 1e-1])
        self.rbf_scores: Optional[np.ndarray] = None
        self.search_table: List[Dict] = []
        self.need_proba = need_proba
        self.search_linear_first = search_linear_first
        self.skip_default_fit = skip_default_fit
        self._t0: Optional[float] = None

    def _ensure_timer(self) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def _elapsed(self) -> float:
        self._ensure_timer()
        return time.perf_counter() - self._t0

    def _log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}  elapsed {fmt_hms(self._elapsed())}", flush=True)

    def run(self) -> Dict[str, float]:
        """Run the five steps in order, with a running elapsed clock."""
        self._t0 = time.perf_counter()
        n_rbf = len(self.C_grid) * len(self.gamma_grid)
        n_lin = len(self.C_grid)
        n, d = self.pack.X_train.shape
        print(
            f"[{self.name}] start  train={n} x {d}  "
            f"search={n_rbf} RBF + {n_lin} linear  proba={self.need_proba}",
            flush=True,
        )
        metrics = super().run()
        print(f"[{self.name}] total {fmt_hms(self._elapsed())}", flush=True)
        return metrics

    def ingest(self) -> None:
        """Scale features with a scaler fit on train only."""
        self._ensure_timer()
        n, d = self.pack.X_train.shape
        self._log(f"ingest  scaling {n} x {d}")
        t1 = time.perf_counter()
        self.scaled, self.scaler = scale_pack(self.pack)
        self.y_train = self.pack.y_train
        self._log(f"ingest  done in {fmt_hms(time.perf_counter() - t1)}")

    def fit(self) -> None:
        """Train a default RBF SVM (C=1, gamma=scale). Skip if it would duplicate a search point."""
        if self.skip_default_fit:
            self._log("fit  skipped (default RBF is a search point)")
            return
        assert self.scaled is not None
        self._log("fit  default RBF C=1 gamma=scale")
        t1 = time.perf_counter()
        self.model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1000)
        self.model.fit(self.scaled.X_train, self.scaled.y_train)
        pred_val = self.model.predict(self.scaled.X_val)
        self.val_metrics = classification_metrics(self.scaled.y_val, pred_val)
        self._log(
            f"fit  done in {fmt_hms(time.perf_counter() - t1)}  "
            f"val_acc={self.val_metrics['acc']:.3f}"
        )

    def search(self) -> None:
        """Grid C x gamma on RBF, plus a linear-C sweep. Score on the validation set."""
        assert self.scaled is not None
        Xtr, ytr = self.scaled.X_train, self.scaled.y_train
        Xva, yva = self.scaled.X_val, self.scaled.y_val
        scores = np.zeros((len(self.C_grid), len(self.gamma_grid)), dtype=np.float32)
        best = {"acc": -1.0, "params": {}}
        self.search_table = []

        jobs: List[Tuple[str, float, object]] = []
        linear_jobs = [("linear", C, None) for C in self.C_grid]
        rbf_jobs = [("rbf", C, g) for C in self.C_grid for g in self.gamma_grid]
        if self.search_linear_first:
            jobs = linear_jobs + rbf_jobs
        else:
            jobs = rbf_jobs + linear_jobs

        n_jobs = len(jobs)
        self._log(f"search  {n_jobs} configs")
        fit_times: List[float] = []

        for k, (kernel, C, g) in enumerate(jobs, start=1):
            t1 = time.perf_counter()
            clf = SVC(
                kernel=kernel,
                C=C,
                gamma="scale" if g is None else g,
                class_weight="balanced",
                cache_size=1000,
            )
            clf.fit(Xtr, ytr)
            acc = float(accuracy_score(yva, clf.predict(Xva)))
            dt = time.perf_counter() - t1
            fit_times.append(dt)
            self.search_table.append({"kernel": kernel, "C": C, "gamma": g, "val_acc": acc})
            if kernel == "rbf":
                i = self.C_grid.index(C)
                j = self.gamma_grid.index(g)
                scores[i, j] = acc
            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "params": {
                        "kernel": kernel,
                        "C": C,
                        "gamma": "scale" if g is None else g,
                    },
                }
            mean_dt = sum(fit_times) / len(fit_times)
            remaining = mean_dt * (n_jobs - k)
            gamma_txt = "scale" if g is None else g
            self._log(
                f"search  {k}/{n_jobs}  {kernel} C={C} gamma={gamma_txt}  "
                f"val_acc={acc:.3f}  {fmt_hms(dt)}  remaining ~{fmt_hms(remaining)}"
            )

        self.rbf_scores = scores
        self.best_params = best["params"]
        self._log(f"search  done  best={self.best_params}  val_acc={best['acc']:.3f}")
        self._plot_heatmap()

    def _plot_heatmap(self) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.rbf_scores is None:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        gamma_labels = [str(g) for g in self.gamma_grid]
        sns.heatmap(
            self.rbf_scores, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=gamma_labels, yticklabels=[str(c) for c in self.C_grid], ax=ax,
        )
        ax.set_xlabel("gamma")
        ax.set_ylabel("C")
        ax.set_title(f"SVM RBF val accuracy — {self.name}")
        fig.tight_layout()
        save_and_show(fig, f"svm_heatmap_{self.name}.png", self.figures_dir)

    def refit(self) -> None:
        """Refit the best config on train. Platt scaling only if need_proba is True.

        probability=True on ~27k-d handcrafted vectors can take tens of minutes.
        The ensemble uses the deep SVM, so handcrafted does not need probabilities.
        """
        assert self.scaled is not None
        params = dict(self.best_params)
        kernel = params.get("kernel", "rbf")
        C = params.get("C", 1.0)
        gamma = params.get("gamma", "scale")
        self._log(f"refit  {kernel} C={C} gamma={gamma}  proba={self.need_proba}")
        t1 = time.perf_counter()
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight="balanced",
            probability=self.need_proba,
            cache_size=1000,
        )
        self.model.fit(self.scaled.X_train, self.scaled.y_train)
        self.y_true_test = self.scaled.y_test
        self.y_pred_test = self.model.predict(self.scaled.X_test)
        self.y_pred_val = self.model.predict(self.scaled.X_val)
        if self.need_proba:
            self.proba_test = self.model.predict_proba(self.scaled.X_test)
        else:
            self.proba_test = None
        self.val_metrics = classification_metrics(self.scaled.y_val, self.y_pred_val)
        self._log(f"refit  done in {fmt_hms(time.perf_counter() - t1)}")
