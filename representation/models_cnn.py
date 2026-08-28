"""CNN hypothesis classes: EfficientNet-B0 and ResNet-50."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score

from evaluation.metrics import classification_metrics
from evaluation.plots import save_and_show
from evaluation.rules import apply_mixup
from optimization.loop import LearnerLoop, class_weight_dict
from representation.constants import IMG_SIZE, RANDOM_STATE, STRESS_NAMES
from representation.dataset import LeafDataset, TrainValTestSplit


def load_rgb_batch(paths: Sequence[str], size: int = IMG_SIZE) -> np.ndarray:
    """Load RGB uint8 images, resized to size x size."""
    out = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            out.append(np.zeros((size, size, 3), dtype=np.uint8))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        out.append(rgb)
    return np.stack(out)


def hsv_saturation_crop(bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Crop to the leaf using HSV saturation, then resize to size x size (Esgario-style)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    mask = (s > 40).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def load_rgb_batch_hsv_crop(paths: Sequence[str], size: int = IMG_SIZE) -> np.ndarray:
    """Load RGB images with an HSV-S leaf crop before the 224² resize."""
    out = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            out.append(np.zeros((size, size, 3), dtype=np.uint8))
            continue
        out.append(hsv_saturation_crop(bgr, size))
    return np.stack(out)


class EfficientNetPipeline(LearnerLoop):
    """Two-phase EfficientNet-B0 fine-tune with MixUp."""

    def __init__(
        self,
        dataset: LeafDataset,
        split: TrainValTestSplit,
        name: str,
        results_dir: str,
        figures_dir: str,
        models_dir: str,
        use_mixup: bool = True,
        epochs_head: int = 15,
        epochs_ft: int = 25,
        batch_size: int = 16,
        search_grid: Optional[List[Dict]] = None,
        ingest_hsv_crop: bool = False,
    ):
        super().__init__(name, results_dir, figures_dir)
        self.dataset = dataset
        self.split = split
        self.models_dir = models_dir
        self.use_mixup = use_mixup
        self.epochs_head = epochs_head
        self.epochs_ft = epochs_ft
        self.batch_size = batch_size
        self.search_grid = search_grid or [
            {"lr": 1e-4, "dropout": 0.3, "unfreeze": 20},
            {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20},
        ]
        self.ingest_hsv_crop = ingest_hsv_crop
        self.num_classes = 0
        self.model = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.train_history = None

    def ingest(self) -> None:
        """Load RGB arrays for train / val / test. No test augmentation."""
        df = self.dataset.frame
        loader = load_rgb_batch_hsv_crop if self.ingest_hsv_crop else load_rgb_batch
        self.X_train = loader(df.iloc[self.split.train_idx]["image_path"].tolist())
        self.X_val = loader(df.iloc[self.split.val_idx]["image_path"].tolist())
        self.X_test = loader(df.iloc[self.split.test_idx]["image_path"].tolist())
        self.y_train = df.iloc[self.split.train_idx]["predominant_stress"].values.astype(np.int32)
        self.y_val = df.iloc[self.split.val_idx]["predominant_stress"].values.astype(np.int32)
        self.y_test = df.iloc[self.split.test_idx]["predominant_stress"].values.astype(np.int32)
        self.num_classes = int(self.y_train.max() + 1)
        names = [STRESS_NAMES[i] for i in range(self.num_classes)]
        self.class_names = names

    @staticmethod
    def _backbone(model):
        """Find the pretrained CNN inside the small wrapper model."""
        for layer in model.layers:
            name = layer.name.lower()
            if "efficientnet" in name or "resnet" in name:
                return layer
        return model.layers[2]

    def _unfreeze_last(self, model, n: int) -> None:
        """Open the last n layers of the backbone; freeze the rest."""
        base = self._backbone(model)
        if n <= 0:
            base.trainable = False
            return
        base.trainable = True
        for layer in base.layers[:-n]:
            layer.trainable = False

    def _build(self, dropout: float = 0.3, unfreeze: int = 0):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input

        inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = layers.Lambda(lambda t: preprocess_input(tf.cast(t, tf.float32)))(inp)
        base = EfficientNetB0(include_top=False, pooling="avg", weights="imagenet")
        base.trainable = False
        x = base(x, training=False)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(self.num_classes, activation="softmax")(x)
        model = models.Model(inp, out)
        if unfreeze > 0:
            base.trainable = True
            for layer in base.layers[:-unfreeze]:
                layer.trainable = False
        return model, base

    def _augment(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flips and a cheap brightness jitter. MixUp is applied per batch in fit."""
        rng = np.random.default_rng(RANDOM_STATE)
        x = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                x[i] = np.fliplr(x[i])
            if rng.random() < 0.5:
                x[i] = np.flipud(x[i])
            factor = 0.85 + 0.3 * rng.random()
            x[i] = np.clip(x[i].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return x, y

    def _onehot(self, y: np.ndarray) -> np.ndarray:
        oh = np.zeros((len(y), self.num_classes), dtype=np.float32)
        oh[np.arange(len(y)), y] = 1.0
        return oh

    def _train_keras(self, model, epochs: int, lr: float, unfreeze_note: str):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        Xtr, ytr = self.X_train, self.y_train
        Xtr, ytr = self._augment(Xtr, ytr)
        ytr_oh = self._onehot(ytr)
        yva_oh = self._onehot(self.y_val)
        if self.use_mixup:
            rng = np.random.default_rng(RANDOM_STATE + epochs)
            lam = float(rng.beta(0.2, 0.2))
            perm = rng.permutation(len(Xtr))
            Xtr, ytr_oh = apply_mixup(Xtr.astype(np.float32), ytr_oh, lam, perm)
        weights = class_weight_dict(self.y_train)
        cb = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        hist = model.fit(
            Xtr, ytr_oh,
            validation_data=(self.X_val, yva_oh),
            epochs=epochs,
            batch_size=self.batch_size,
            class_weight=weights,
            callbacks=cb,
            verbose=1,
        )
        self.train_history = hist.history
        self.history[unfreeze_note] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return model

    def fit(self) -> None:
        """Phase 1 frozen backbone, then phase 2 with last 20 layers open."""
        model, _ = self._build(dropout=0.3, unfreeze=0)
        model = self._train_keras(model, self.epochs_head, lr=1e-3, unfreeze_note="phase1")
        self._unfreeze_last(model, 20)
        model = self._train_keras(model, self.epochs_ft, lr=1e-5, unfreeze_note="phase2")
        self.model = model
        self._plot_curves(f"{self.name}_default")
        pred_val = np.argmax(model.predict(self.X_val, verbose=0), axis=1)
        self.val_metrics = classification_metrics(self.y_val, pred_val)
        self.best_params = {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20}

    def search(self) -> None:
        """Small manual grid. Each config is trained from ImageNet weights."""
        best = {"acc": -1.0, "params": dict(self.best_params), "model": self.model}
        rows = []
        for cfg in self.search_grid:
            model, _ = self._build(dropout=cfg["dropout"], unfreeze=0)
            model = self._train_keras(model, self.epochs_head, lr=1e-3, unfreeze_note=f"s1_{cfg}")
            self._unfreeze_last(model, int(cfg["unfreeze"]))
            model = self._train_keras(model, self.epochs_ft, lr=cfg["lr"], unfreeze_note=f"s2_{cfg}")
            pred_val = np.argmax(model.predict(self.X_val, verbose=0), axis=1)
            acc = float(accuracy_score(self.y_val, pred_val))
            rows.append({**cfg, "val_acc": acc})
            if acc > best["acc"]:
                best = {"acc": acc, "params": cfg, "model": model}
        self.search_table = rows
        self.best_params = best["params"]
        self.model = best["model"]
        self._plot_search_bars(rows)

    def _plot_search_bars(self, rows: List[Dict]) -> None:
        import matplotlib.pyplot as plt

        if not rows:
            return
        labels = [f"lr={r['lr']}\nu={r['unfreeze']}" for r in rows]
        accs = [r["val_acc"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(accs)), accs)
        ax.set_xticks(range(len(accs)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("val accuracy")
        ax.set_title(f"{self.name} hyperparameter search")
        fig.tight_layout()
        save_and_show(fig, f"cnn_search_{self.name}.png", self.figures_dir)

    def _plot_curves(self, tag: str) -> None:
        import matplotlib.pyplot as plt

        if not self.train_history:
            return
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(self.train_history.get("loss", []), label="train")
        axes[0].plot(self.train_history.get("val_loss", []), label="val")
        axes[0].set_title("loss")
        axes[0].legend()
        axes[1].plot(self.train_history.get("accuracy", []), label="train")
        axes[1].plot(self.train_history.get("val_accuracy", []), label="val")
        axes[1].set_title("accuracy")
        axes[1].legend()
        fig.suptitle(self.name)
        fig.tight_layout()
        save_and_show(fig, f"curves_{tag}.png", self.figures_dir)

    def refit(self) -> None:
        """Keep the best search model and score test once."""
        if self.model is None:
            raise RuntimeError("Call fit/search before refit.")
        os.makedirs(self.models_dir, exist_ok=True)
        save_path = os.path.join(self.models_dir, f"{self.name}.keras")
        try:
            self.model.save(save_path)
        except Exception:
            pass
        proba = self.model.predict(self.X_test, verbose=0)
        self.proba_test = proba
        self.y_true_test = self.y_test
        self.y_pred_test = np.argmax(proba, axis=1)
        pred_val = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        self.val_metrics = classification_metrics(self.y_val, pred_val)
        self._plot_curves(self.name)


class ResNetPipeline(EfficientNetPipeline):
    """ResNet-50 with the same training protocol as EfficientNet."""

    def _build(self, dropout: float = 0.3, unfreeze: int = 0):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input

        inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = layers.Lambda(lambda t: preprocess_input(tf.cast(t, tf.float32)))(inp)
        base = ResNet50(include_top=False, pooling="avg", weights="imagenet")
        base.trainable = False
        x = base(x, training=False)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(self.num_classes, activation="softmax")(x)
        model = models.Model(inp, out)
        if unfreeze > 0:
            base.trainable = True
            for layer in base.layers[:-unfreeze]:
                layer.trainable = False
        return model, base
