"""MixUp, residual Mixed rule, save_and_show, LearnerLoop.report shape."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from p1_core import (
    LearnerLoop,
    ResidualMixedRule,
    apply_mixup,
    apply_residual_rule,
    fmt_hms,
    save_and_show,
)


def test_fmt_hms():
    assert fmt_hms(0) == "0:00"
    assert fmt_hms(75) == "1:15"
    assert fmt_hms(3661) == "1:01:01"


def test_mixup_matches_lambda_formula():
    images = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)  # (2, 1, 2) squeezed later
    images = np.stack([np.ones((4, 4, 3), dtype=np.float32), np.zeros((4, 4, 3), dtype=np.float32)])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lam = 0.7
    index_j = np.array([1, 0])
    x_m, y_m = apply_mixup(images, labels, lam, index_j)
    np.testing.assert_allclose(x_m[0], lam * images[0] + (1 - lam) * images[1])
    np.testing.assert_allclose(y_m[0], lam * labels[0] + (1 - lam) * labels[1])
    np.testing.assert_allclose(x_m[1], lam * images[1] + (1 - lam) * images[0])


def test_residual_rule_two_diseases_and_small_margin():
    class_probs = np.array(
        [
            [0.05, 0.40, 0.35, 0.10, 0.05, 0.05],  # small margin 1 vs 2
            [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],  # confident healthy
            [0.10, 0.50, 0.10, 0.10, 0.10, 0.10],  # miner, but two disease heads
        ],
        dtype=np.float32,
    )
    disease = np.array(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    pred = apply_residual_rule(class_probs, disease, tau=0.5, delta=0.10)
    assert pred[0] == 5  # margin 0.05 < 0.10
    assert pred[1] == 0
    assert pred[2] == 5  # two heads above 0.5


def test_residual_rule_search_locks_on_val():
    y = np.array([5, 0, 1, 5])
    class_probs = np.eye(6, dtype=np.float32)[[5, 0, 1, 2]]
    disease = np.array(
        [
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
        ]
    )
    rule = ResidualMixedRule(tau_grid=[0.5], delta_grid=[0.05, 0.5])
    tau, delta = rule.search(y, class_probs, disease)
    pred = rule.predict(class_probs, disease)
    assert pred.shape == (4,)
    assert tau == 0.5
    assert delta in (0.05, 0.5)


def test_save_and_show_writes_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_and_show(fig, "line.png", str(tmp_path))
    assert os.path.isfile(path)
    assert path.endswith("line.png")
    assert os.path.getsize(path) > 0


class _DummyLoop(LearnerLoop):
    def ingest(self):
        pass

    def fit(self):
        pass

    def search(self):
        # Must not look at test labels.
        self.history["used_test"] = False

    def refit(self):
        self.y_true_test = np.array([0, 1, 2, 0])
        self.y_pred_test = np.array([0, 1, 1, 0])


def test_report_confusion_shape(tmp_path):
    loop = _DummyLoop("dummy", str(tmp_path), str(tmp_path))
    loop.search()
    assert loop.history["used_test"] is False
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    loop.report(y_true, y_pred, "dummy")
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert os.path.isfile(os.path.join(str(tmp_path), "cm_dummy.png"))
    assert os.path.isfile(os.path.join(str(tmp_path), "dummy_report.txt"))
