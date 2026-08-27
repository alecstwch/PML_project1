"""
Shared classes for Project 1: coffee leaf disease classification.

The notebook imports this module and walks the grading loop.
Tests import the same classes and use tiny fake images.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRESS_NAMES = {
    0: "Healthy",
    1: "Miner",
    2: "Rust",
    3: "Phoma",
    4: "Cercospora",
    5: "Mixed",
}

SEVERITY_NAMES = {
    0: "Healthy",
    1: "Very Low",
    2: "Low",
    3: "High",
    4: "Very High",
}

# BRACOL YOLO class ids are not the same as predominant_stress.
BRACOL_CLASSES = {0: "Cercospora", 1: "Miner", 2: "Phoma", 3: "Rust"}
BRACOL_TO_STRESS = {0: 4, 1: 1, 2: 3, 3: 2}

RANDOM_STATE = 42
IMG_SIZE = 224
HANDCRAFTED_SIZE = 224
HOG_CELL = (8, 8)
HOG_BLOCK = (2, 2)
HOG_ORIENT = 9
HSV_BINS = (8, 8, 8)
LBP_POINTS = 24
LBP_RADIUS = 3

# White paper: low saturation, high value. Leaf pixels are the rest.
MASK_S_MAX = 40
MASK_V_MIN = 180


def fmt_hms(seconds: float) -> str:
    """Format a duration as m:ss or h:mm:ss."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def resolve_paths(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Return data and output folders next to this project (Windows or WSL)."""
    if base_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        if os.path.basename(cwd) == "Project_1":
            base_dir = cwd
        elif os.path.basename(here) == "Project_1":
            base_dir = here
        else:
            base_dir = os.path.join(cwd, "Project_1")
    data_dir = os.path.join(base_dir, "..", "coffee-datasets", "coffee-datasets", "leaf")
    paths = {
        "base_dir": os.path.abspath(base_dir),
        "data_dir": os.path.abspath(data_dir),
        "image_dir": os.path.abspath(os.path.join(data_dir, "images")),
        "csv_path": os.path.abspath(os.path.join(data_dir, "dataset.csv")),
        "bracol_dir": os.path.abspath(
            os.path.join(base_dir, "..", "BRACOL_REVIEWED_ANNOTATIONS", "BRACOL_REVIEWED")
        ),
        "ninja_dir": os.path.abspath(os.path.join(base_dir, "..", "dataset-ninja")),
        "figures_dir": os.path.abspath(os.path.join(base_dir, "figures")),
        "features_dir": os.path.abspath(os.path.join(base_dir, "features")),
        "results_dir": os.path.abspath(os.path.join(base_dir, "results")),
        "models_dir": os.path.abspath(os.path.join(base_dir, "models")),
    }
    for key in ("figures_dir", "features_dir", "results_dir", "models_dir"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def report_compute() -> Dict[str, object]:
    """Print CPU/GPU devices and allow TensorFlow to grow GPU memory as needed."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    info = {
        "tf": tf.__version__,
        "cuda_built": bool(tf.test.is_built_with_cuda()),
        "gpu_names": [gpu.name for gpu in gpus],
    }
    print("TensorFlow", info["tf"], "| CUDA build:", info["cuda_built"])
    if gpus:
        print("Using GPU:", info["gpu_names"])
    else:
        print("No GPU visible. For CUDA, run this notebook with the WSL kernel (pml_venv).")
    return info


def _leaf_id_from_filename(name: str) -> Optional[int]:
    """Parse a CSV leaf id from `{id}.jpg` or a Roboflow `{id}_jpg.rf.*.jpg` name."""
    stem, ext = os.path.splitext(name)
    if ext.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    if stem.isdigit():
        return int(stem)
    match = re.match(r"^(\d+)_jpg\.rf\.", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def copy_missing_leaf_jpgs(csv_path: str, image_dir: str, extra_roots: Sequence[str]) -> int:
    """Copy `{id}.jpg` from extra folders into image_dir when the CSV row has no file yet.

    Accepts plain `{id}.jpg` (Dataset Ninja / Supervisely) and Roboflow exports
    named `{id}_jpg.rf.<hash>.jpg`. Does not overwrite a file that is already present.
    """
    import shutil

    df = pd.read_csv(csv_path)
    wanted = {int(i) for i in df["id"].tolist()}
    os.makedirs(image_dir, exist_ok=True)
    copied = 0
    for root in extra_roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                img_id = _leaf_id_from_filename(name)
                if img_id is None or img_id not in wanted:
                    continue
                dest = os.path.join(image_dir, f"{img_id}.jpg")
                if os.path.exists(dest):
                    continue
                src = os.path.join(dirpath, name)
                shutil.copy2(src, dest)
                copied += 1
    return copied


def extra_image_roots(paths: Dict[str, str]) -> List[str]:
    """Folders that may hold extra `{id}.jpg` files (Dataset Ninja / YOLO exports)."""
    base = paths["base_dir"]
    roots = [
        paths.get("ninja_dir", ""),
        os.path.expanduser("~/dataset-ninja"),
        os.path.abspath(os.path.join(base, "..", "BRACOL_REVIEWED_ANNOTATIONS")),
        os.path.abspath(os.path.join(base, "..", "BRACOL-ORIGINAL-ANNOTATIONS")),
    ]
    seen = set()
    out = []
    for root in roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root in seen:
            continue
        seen.add(abs_root)
        out.append(abs_root)
    return out


def save_and_show(fig, name: str, figures_dir: str) -> str:
    """Save a figure to figures_dir and embed the PNG in the notebook output.

    Display the file (image/png), not the matplotlib Figure object. display(fig)
    under Agg / nbclient only stores the text '<Figure size ... with N Axes>'.
    """
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        from IPython.display import Image as IPyImage
        from IPython.display import display

        display(IPyImage(filename=os.path.abspath(path)))
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def is_jpeg_valid(path: str) -> bool:
    """Return True if the file opens as a complete image."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the leaf (255) vs white background (0)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    background = (s < MASK_S_MAX) & (v > MASK_V_MIN)
    return ((~background).astype(np.uint8)) * 255


def read_bgr(path: str, size: Optional[int] = None) -> Optional[np.ndarray]:
    """Read a BGR image; optionally resize to size x size."""
    img = cv2.imread(path)
    if img is None:
        return None
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length (guard against a zero vector)."""
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


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


# ---------------------------------------------------------------------------
# MixUp and class-5 residual rule (pure functions, easy to test)
# ---------------------------------------------------------------------------

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
    n = class_probs.shape[0]
    argmax = np.argmax(class_probs, axis=1)
    part = np.partition(class_probs, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    n_hot = (disease_probs >= tau).sum(axis=1)
    pred = argmax.copy()
    mixed = (n_hot >= 2) | (margin < delta)
    pred[mixed] = 5
    return pred


# ---------------------------------------------------------------------------
# Dataset corridor
# ---------------------------------------------------------------------------

class LeafDataset:
    """Load the leaf CSV and keep rows whose image file can be read."""

    def __init__(self, csv_path: str, image_dir: str, check_jpeg: bool = True):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.check_jpeg = check_jpeg
        self.frame: Optional[pd.DataFrame] = None
        self.n_csv: int = 0
        self.n_missing: int = 0
        self.n_corrupt: int = 0
        self.corrupt_ids: List[int] = []
        self.duplicate_groups: List[List[int]] = []

    def load_and_match(self) -> pd.DataFrame:
        """Load the CSV and drop rows whose image file is missing or corrupt."""
        df = pd.read_csv(self.csv_path)
        self.n_csv = len(df)
        df["image_path"] = df["id"].apply(lambda x: os.path.join(self.image_dir, f"{x}.jpg"))
        exists = df["image_path"].apply(os.path.exists)
        self.n_missing = int((~exists).sum())
        df = df[exists].reset_index(drop=True)

        if self.check_jpeg:
            valid = []
            self.corrupt_ids = []
            for _, row in df.iterrows():
                ok = is_jpeg_valid(row["image_path"])
                valid.append(ok)
                if not ok:
                    self.corrupt_ids.append(int(row["id"]))
            self.n_corrupt = len(self.corrupt_ids)
            df = df[pd.Series(valid)].reset_index(drop=True)
        else:
            self.n_corrupt = 0
            self.corrupt_ids = []

        self.frame = df
        return df

    def class_counts(self) -> pd.Series:
        """Count of each predominant_stress label."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        return self.frame["predominant_stress"].value_counts().sort_index()

    def find_duplicates(self, hash_size: int = 8) -> List[List[int]]:
        """Group image ids that share the same perceptual hash."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        try:
            import imagehash
        except ImportError:
            imagehash = None

        buckets: Dict[str, List[int]] = {}
        for _, row in self.frame.iterrows():
            key = self._hash_image(row["image_path"], hash_size, imagehash)
            buckets.setdefault(key, []).append(int(row["id"]))
        self.duplicate_groups = [ids for ids in buckets.values() if len(ids) > 1]
        return self.duplicate_groups

    @staticmethod
    def _hash_image(path: str, hash_size: int, imagehash_mod) -> str:
        """Perceptual hash of one file (imagehash if present, else a tiny average hash)."""
        img = Image.open(path).convert("RGB")
        if imagehash_mod is not None:
            return str(imagehash_mod.phash(img, hash_size=hash_size))
        small = img.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert("L")
        arr = np.asarray(small, dtype=np.float32)
        bits = arr > arr.mean()
        return "".join("1" if b else "0" for b in bits.ravel())

    def eda_summary(self) -> Dict[str, object]:
        """Small dict used by plots and by tests."""
        if self.frame is None:
            raise RuntimeError("Call load_and_match first.")
        counts = self.class_counts()
        return {
            "n_csv": self.n_csv,
            "n_kept": int(len(self.frame)),
            "n_missing": self.n_missing,
            "n_corrupt": self.n_corrupt,
            "class_counts": {int(k): int(v) for k, v in counts.items()},
        }


class TrainValTestSplit:
    """Stratified 70 / 15 / 15 split. Indices are positions in dataset.frame."""

    def __init__(self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        self.train_idx = np.asarray(train_idx)
        self.val_idx = np.asarray(val_idx)
        self.test_idx = np.asarray(test_idx)

    @classmethod
    def make(
        cls,
        dataset: LeafDataset,
        seed: int = RANDOM_STATE,
        path: Optional[str] = None,
    ) -> "TrainValTestSplit":
        """Build a new split, or load one from path if that file already exists."""
        if path and os.path.exists(path):
            return cls.load(path)
        if dataset.frame is None:
            raise RuntimeError("Call load_and_match first.")
        y = dataset.frame["predominant_stress"].values
        idx = np.arange(len(dataset.frame))
        # Split test off first (15%), then val (15% of the original, i.e. 15/85 of the rest).
        # Doing 70/30 then 50/50 can leave one sample of a rare class in the 30% pool, and
        # sklearn refuses to stratify that.
        train_val, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
        try:
            train, val = train_test_split(
                train_val, test_size=0.15 / 0.85, stratify=y[train_val], random_state=seed
            )
        except ValueError:
            train, val = train_test_split(
                train_val, test_size=0.15 / 0.85, random_state=seed
            )
        split = cls(train, val, test)
        if path:
            split.save(path)
        return split

    def save(self, path: str) -> None:
        """Write train / val / test indices to an .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, train_idx=self.train_idx, val_idx=self.val_idx, test_idx=self.test_idx)

    @classmethod
    def load(cls, path: str) -> "TrainValTestSplit":
        """Read a split saved by save()."""
        data = np.load(path)
        return cls(data["train_idx"], data["val_idx"], data["test_idx"])

    def sizes(self) -> Dict[str, int]:
        """Number of samples in each split."""
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }

    def assert_disjoint(self) -> None:
        """Raise if any image index sits in two splits."""
        t, v, s = set(self.train_idx), set(self.val_idx), set(self.test_idx)
        if t & v or t & s or v & s:
            raise ValueError("Split indices overlap.")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class FeaturePack:
    """Train / val / test matrices for one representation."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cache_path: str = ""
    name: str = ""


class FeatureExtractor(ABC):
    """Turn a dataset split into numeric feature matrices."""

    @abstractmethod
    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Return train/val/test matrices. Fit nothing that could leak from val or test."""


def hsv_histogram(bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """512-d HSV colour histogram, optionally masked to the leaf."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, HSV_BINS, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def lbp_histogram(bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """26-bin uniform LBP histogram."""
    from skimage.feature import local_binary_pattern

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = np.where(mask > 0, gray, 0)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    if mask is not None:
        vals = lbp[mask > 0]
        if vals.size == 0:
            vals = lbp.ravel()
    else:
        vals = lbp.ravel()
    hist, _ = np.histogram(vals, bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7
    return hist


def hog_vector(bgr: np.ndarray, size: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """HOG descriptor on a square resize."""
    from skimage.feature import hog

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = np.where(mask > 0, gray, 0)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    vec = hog(
        gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_CELL,
        cells_per_block=HOG_BLOCK,
        feature_vector=True,
    )
    return np.asarray(vec, dtype=np.float32)


def handcrafted_vector(bgr: np.ndarray, size: int = HANDCRAFTED_SIZE) -> np.ndarray:
    """Concatenate masked HSV histogram, LBP, and HOG, then L2-normalise."""
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    mask = leaf_mask(img)
    color = hsv_histogram(img, mask)
    tex = lbp_histogram(img, mask)
    edges = hog_vector(img, size, mask)
    return l2_normalize(np.concatenate([color, tex, edges]))


class HandcraftedFeatures(FeatureExtractor):
    """Build a vector from colour, texture, and edge histograms of one leaf image."""

    def __init__(self, cache_path: str, image_size: int = HANDCRAFTED_SIZE):
        self.cache_path = cache_path
        self.image_size = image_size

    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Return train/val/test matrices. Fit nothing that could leak from val or test."""
        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=False)
            return FeaturePack(
                data["X_train"], data["X_val"], data["X_test"],
                data["y_train"], data["y_val"], data["y_test"],
                cache_path=self.cache_path, name="handcrafted",
            )
        X_train, y_train = self._matrix(dataset, split.train_idx)
        X_val, y_val = self._matrix(dataset, split.val_idx)
        X_test, y_test = self._matrix(dataset, split.test_idx)
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        np.savez(
            self.cache_path,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )
        return FeaturePack(
            X_train, X_val, X_test, y_train, y_val, y_test,
            cache_path=self.cache_path, name="handcrafted",
        )

    def _matrix(self, dataset: LeafDataset, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = dataset.frame.iloc[idx]
        feats = []
        labels = []
        for _, row in rows.iterrows():
            img = cv2.imread(row["image_path"])
            if img is None:
                continue
            feats.append(handcrafted_vector(img, self.image_size))
            labels.append(int(row["predominant_stress"]))
        return np.stack(feats), np.asarray(labels, dtype=np.int32)


class FrozenDeepFeatures(FeatureExtractor):
    """1280-d vectors from a frozen ImageNet EfficientNet-B0 (or a stub in tests)."""

    def __init__(
        self,
        cache_path: str,
        model=None,
        input_size: int = IMG_SIZE,
        batch_size: int = 32,
        use_keras_preprocess: bool = True,
    ):
        self.cache_path = cache_path
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_keras_preprocess = use_keras_preprocess

    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Return cached embeddings, or extract them once and write the cache."""
        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=False)
            return FeaturePack(
                data["X_train"], data["X_val"], data["X_test"],
                data["y_train"], data["y_val"], data["y_test"],
                cache_path=self.cache_path, name="frozen_deep",
            )
        model = self._get_model()
        X_train, y_train = self._matrix(dataset, split.train_idx, model)
        X_val, y_val = self._matrix(dataset, split.val_idx, model)
        X_test, y_test = self._matrix(dataset, split.test_idx, model)
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        np.savez(
            self.cache_path,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )
        return FeaturePack(
            X_train, X_val, X_test, y_train, y_val, y_test,
            cache_path=self.cache_path, name="frozen_deep",
        )

    def _get_model(self):
        if self.model is not None:
            return self.model
        from tensorflow.keras.applications import EfficientNetB0

        self.model = EfficientNetB0(
            include_top=False, pooling="avg", weights="imagenet",
            input_shape=(self.input_size, self.input_size, 3),
        )
        return self.model

    def _preprocess(self, batch: np.ndarray) -> np.ndarray:
        if not self.use_keras_preprocess:
            return batch.astype(np.float32)
        from tensorflow.keras.applications.efficientnet import preprocess_input

        return preprocess_input(batch.astype(np.float32))

    def _matrix(self, dataset: LeafDataset, idx: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        rows = dataset.frame.iloc[idx]
        feats = []
        labels = []
        batch_imgs: List[np.ndarray] = []
        batch_y: List[int] = []

        def flush():
            if not batch_imgs:
                return
            x = self._preprocess(np.stack(batch_imgs))
            pred = model.predict(x, verbose=0)
            feats.append(np.asarray(pred, dtype=np.float32))
            labels.extend(batch_y)
            batch_imgs.clear()
            batch_y.clear()

        for _, row in rows.iterrows():
            bgr = cv2.imread(row["image_path"])
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            batch_imgs.append(rgb)
            batch_y.append(int(row["predominant_stress"]))
            if len(batch_imgs) >= self.batch_size:
                flush()
        flush()
        return np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int32)


class PCAFeatures(FeatureExtractor):
    """Reduce frozen deep features with PCA fit on the train split only."""

    def __init__(self, n_components: int = 256, source: Optional[FeaturePack] = None):
        self.n_components = n_components
        self.source = source
        self.pca: Optional[PCA] = None

    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Fit PCA on train embeddings, then transform val and test."""
        if self.source is None:
            raise ValueError("PCAFeatures needs a source FeaturePack.")
        src = self.source
        n = min(self.n_components, src.X_train.shape[0], src.X_train.shape[1])
        self.pca = PCA(n_components=n, random_state=RANDOM_STATE)
        X_train = self.pca.fit_transform(src.X_train)
        X_val = self.pca.transform(src.X_val)
        X_test = self.pca.transform(src.X_test)
        return FeaturePack(
            X_train.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32),
            src.y_train, src.y_val, src.y_test,
            name=f"pca_{n}",
        )


def scale_pack(pack: FeaturePack) -> Tuple[FeaturePack, StandardScaler]:
    """StandardScaler fit on train only, then applied to val and test."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(pack.X_train)
    X_val = scaler.transform(pack.X_val)
    X_test = scaler.transform(pack.X_test)
    scaled = FeaturePack(
        X_train, X_val, X_test, pack.y_train, pack.y_val, pack.y_test,
        cache_path=pack.cache_path, name=pack.name + "_scaled",
    )
    return scaled, scaler


# ---------------------------------------------------------------------------
# Learner loop
# ---------------------------------------------------------------------------

class LearnerLoop(ABC):
    """Shared grading loop: ingest, fit, search, refit, report. Subclasses do the work."""

    def __init__(self, name: str, results_dir: str, figures_dir: str, class_names: Optional[List[str]] = None):
        self.name = name
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.class_names = class_names or [STRESS_NAMES[i] for i in range(6)]
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
        self.fit()
        self.search()
        self.refit()
        if self.y_true_test is None or self.y_pred_test is None:
            raise RuntimeError("refit() must set y_true_test and y_pred_test.")
        self.test_metrics = self.report(self.y_true_test, self.y_pred_test, self.name)
        return self.test_metrics


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
        self._log(f"ingest  done in {fmt_hms(time.perf_counter() - t1)}")

    def fit(self) -> None:
        """Train a default RBF SVM (C=1, gamma=scale)."""
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
        for C in self.C_grid:
            for g in self.gamma_grid:
                jobs.append(("rbf", C, g))
        for C in self.C_grid:
            jobs.append(("linear", C, None))

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
        sw = _sample_weights(self.scaled.y_train)
        self.model.fit(self.scaled.X_train, self.scaled.y_train, sample_weight=sw)
        pred_val = self.model.predict(self.scaled.X_val)
        self.val_metrics = classification_metrics(self.scaled.y_val, pred_val)

    def search(self) -> None:
        """Small grid scored on the validation set."""
        import xgboost as xgb

        assert self.scaled is not None
        n_classes = int(np.max(self.scaled.y_train) + 1)
        sw = _sample_weights(self.scaled.y_train)
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
        sw = _sample_weights(self.scaled.y_train)
        self.model.fit(self.scaled.X_train, self.scaled.y_train, sample_weight=sw)
        self.y_true_test = self.scaled.y_test
        self.y_pred_test = self.model.predict(self.scaled.X_test)
        self.y_pred_val = self.model.predict(self.scaled.X_val)
        self.proba_test = self.model.predict_proba(self.scaled.X_test)
        self.val_metrics = classification_metrics(self.scaled.y_val, self.y_pred_val)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    mapping = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([mapping[int(t)] for t in y], dtype=np.float32)


def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    """Keras-style class_weight mapping."""
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# ---------------------------------------------------------------------------
# CNN pipelines
# ---------------------------------------------------------------------------

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
        self.num_classes = 6
        self.model = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.train_history = None

    def ingest(self) -> None:
        """Load RGB arrays for train / val / test. No test augmentation."""
        df = self.dataset.frame
        self.X_train = load_rgb_batch(df.iloc[self.split.train_idx]["image_path"].tolist())
        self.X_val = load_rgb_batch(df.iloc[self.split.val_idx]["image_path"].tolist())
        self.X_test = load_rgb_batch(df.iloc[self.split.test_idx]["image_path"].tolist())
        self.y_train = df.iloc[self.split.train_idx]["predominant_stress"].values.astype(np.int32)
        self.y_val = df.iloc[self.split.val_idx]["predominant_stress"].values.astype(np.int32)
        self.y_test = df.iloc[self.split.test_idx]["predominant_stress"].values.astype(np.int32)
        self.num_classes = int(max(6, self.y_train.max() + 1))

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
        self._plot_curves("effnet_default")
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


# ---------------------------------------------------------------------------
# Residual rule helper
# ---------------------------------------------------------------------------

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
    stacked = np.stack(probas, axis=0)
    return stacked.mean(axis=0)
