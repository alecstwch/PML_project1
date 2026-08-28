"""Handcrafted features, frozen-deep stub, PCA train-only, scaler leak check."""

import os

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from p1_core import (
    FrozenDeepFeatures,
    HandcraftedFeatures,
    LeafDataset,
    PCAFeatures,
    TrainValTestSplit,
    handcrafted_vector,
    hsv_histogram,
    leaf_mask,
    scale_pack,
)


def test_handcrafted_fixed_length_and_deterministic(tmp_leaf_dir):
    path = os.path.join(tmp_leaf_dir["images"], "1.jpg")
    img = cv2.imread(path)
    a = handcrafted_vector(img, size=32)
    b = handcrafted_vector(img, size=32)
    np.testing.assert_allclose(a, b)
    assert a.ndim == 1
    assert a.shape == b.shape
    # L2-normalised
    assert abs(np.linalg.norm(a) - 1.0) < 1e-3


def test_white_background_mask_zeros_background_bins():
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (8, 8), (24, 24), (0, 180, 0), -1)
    mask = leaf_mask(img)
    assert mask[0, 0] == 0
    assert mask[16, 16] == 255
    hist_masked = hsv_histogram(img, mask)
    hist_full = hsv_histogram(img, None)
    assert hist_masked.shape == (512,)
    assert not np.allclose(hist_masked, hist_full)


def test_handcrafted_extractor_cache(tmp_leaf_dir, tmp_path):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"])
    ds.load_and_match()
    split = TrainValTestSplit.make(ds, seed=42)
    cache = str(tmp_path / "handcrafted.npz")
    fe = HandcraftedFeatures(cache, image_size=32)
    pack = fe.extract(ds, split)
    assert pack.X_train.shape[0] == len(split.train_idx)
    assert pack.X_train.shape[1] == pack.X_val.shape[1] == pack.X_test.shape[1]
    pack2 = fe.extract(ds, split)
    np.testing.assert_array_equal(pack.X_test, pack2.X_test)


class _StubBackbone:
    def predict(self, x, verbose=0):
        n = len(x)
        return np.zeros((n, 1280), dtype=np.float32) + x.mean(axis=(1, 2, 3), keepdims=False).reshape(n, 1)


def test_frozen_deep_stub_shape_and_cache(tmp_leaf_dir, tmp_path):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"])
    ds.load_and_match()
    split = TrainValTestSplit.make(ds, seed=42)
    cache = str(tmp_path / "frozen.npz")
    fe = FrozenDeepFeatures(
        cache, model=_StubBackbone(), input_size=32, batch_size=8, use_keras_preprocess=False
    )
    pack = fe.extract(ds, split)
    assert pack.X_train.shape[1] == 1280
    assert pack.X_val.shape[1] == 1280
    pack2 = fe.extract(ds, split)
    np.testing.assert_array_equal(pack.X_train, pack2.X_train)


def test_pca_fit_on_train_only():
    rng = np.random.default_rng(1)
    from p1_core import FeaturePack

    X_train = rng.normal(size=(40, 32))
    X_val = rng.normal(size=(10, 32)) + 50  # shifted, must not affect PCA mean
    X_test = rng.normal(size=(10, 32)) + 50
    y = np.zeros(40, dtype=np.int32)
    yv = np.zeros(10, dtype=np.int32)
    src = FeaturePack(X_train, X_val, X_test, y, yv, yv, name="src")
    pca_fe = PCAFeatures(n_components=4, source=src)
    out = pca_fe.extract(None, None)
    # Reconstruct train mean in original space via inverse; val mean should not match train PCA mean.
    recon_train = pca_fe.pca.inverse_transform(out.X_train)
    np.testing.assert_allclose(recon_train.mean(axis=0), X_train.mean(axis=0), atol=1e-5)
    assert out.X_train.shape[1] == 4
    assert out.X_val.shape[1] == 4


def test_scaler_fit_on_train_not_full_set():
    from p1_core import FeaturePack

    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(30, 8))
    X_val = rng.normal(loc=10.0, scale=1.0, size=(10, 8))
    X_test = rng.normal(loc=10.0, scale=1.0, size=(10, 8))
    y = np.zeros(30, dtype=np.int32)
    yv = np.zeros(10, dtype=np.int32)
    pack = FeaturePack(X_train, X_val, X_test, y, yv, yv)
    scaled, scaler = scale_pack(pack)
    leak = StandardScaler().fit(np.vstack([X_train, X_val, X_test]))
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-6)
    assert not np.allclose(scaler.mean_, leak.mean_)
    np.testing.assert_allclose(scaled.X_train.mean(axis=0), 0.0, atol=1e-6)
