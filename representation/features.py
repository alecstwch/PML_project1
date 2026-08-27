"""Handcrafted, frozen-deep, and PCA feature packs."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from representation.constants import (
    HANDCRAFTED_SIZE,
    HOG_BLOCK,
    HOG_CELL,
    HOG_ORIENT,
    HSV_BINS,
    IMG_SIZE,
    LBP_POINTS,
    LBP_RADIUS,
    MASK_S_MAX,
    MASK_V_MIN,
    RANDOM_STATE,
)
from representation.dataset import LeafDataset, TrainValTestSplit


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


def _cache_matches_split(data, split: TrainValTestSplit) -> bool:
    """True if a feature cache has one row per split index."""
    return (
        data["X_train"].shape[0] == len(split.train_idx)
        and data["X_val"].shape[0] == len(split.val_idx)
        and data["X_test"].shape[0] == len(split.test_idx)
    )


class HandcraftedFeatures(FeatureExtractor):
    """Build a vector from colour, texture, and edge histograms of one leaf image."""

    def __init__(self, cache_path: str, image_size: int = HANDCRAFTED_SIZE):
        self.cache_path = cache_path
        self.image_size = image_size

    @staticmethod
    def _cache_matches_split(data, split: TrainValTestSplit) -> bool:
        return _cache_matches_split(data, split)

    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Return train/val/test matrices. Fit nothing that could leak from val or test."""
        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=False)
            if self._cache_matches_split(data, split):
                return FeaturePack(
                    data["X_train"], data["X_val"], data["X_test"],
                    data["y_train"], data["y_val"], data["y_test"],
                    cache_path=self.cache_path, name="handcrafted",
                )
            os.remove(self.cache_path)
            print("Handcrafted cache row counts do not match the split; rebuilding.")
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

    @staticmethod
    def _cache_matches_split(data, split: TrainValTestSplit) -> bool:
        return _cache_matches_split(data, split)

    def extract(self, dataset: LeafDataset, split: TrainValTestSplit) -> FeaturePack:
        """Return cached embeddings, or extract them once and write the cache."""
        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=False)
            if self._cache_matches_split(data, split):
                return FeaturePack(
                    data["X_train"], data["X_val"], data["X_test"],
                    data["y_train"], data["y_val"], data["y_test"],
                    cache_path=self.cache_path, name="frozen_deep",
                )
            os.remove(self.cache_path)
            print("Frozen-deep cache row counts do not match the split; rebuilding.")
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
