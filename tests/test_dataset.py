"""Dataset loading, EDA counts, duplicates, and the 70/15/15 split."""

import os
import shutil

import numpy as np
from PIL import Image

from p1_core import (
    LeafDataset,
    TrainValTestSplit,
    _leaf_id_from_filename,
    copy_missing_leaf_jpgs,
)


def test_load_and_match_drops_missing_files(tmp_leaf_dir):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], check_jpeg=True)
    df = ds.load_and_match()
    assert "image_path" in df.columns
    assert "predominant_stress" in df.columns
    assert ds.n_csv == 49  # 48 images + 1 missing row
    assert ds.n_missing == 1
    assert len(df) == 48
    assert df["image_path"].apply(os.path.exists).all()
    assert set(df["predominant_stress"].unique()) == set(range(6))


def test_eda_class_counts(tmp_leaf_dir):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], check_jpeg=False)
    ds.load_and_match()
    summary = ds.eda_summary()
    assert summary["n_kept"] == 48
    assert summary["class_counts"][0] == 8
    assert summary["class_counts"][5] == 8
    counts = ds.class_counts()
    assert int(counts.loc[2]) == 8


def test_duplicate_hash_identical_vs_changed(tmp_leaf_dir, tmp_path):
    src = os.path.join(tmp_leaf_dir["images"], "1.jpg")
    copy = tmp_path / "copy.jpg"
    shutil.copy(src, copy)
    other = tmp_path / "other.jpg"
    img = Image.open(src).convert("RGB")
    pix = np.array(img)
    pix[:32, :32] = 0
    Image.fromarray(pix).save(other)

    h_src = LeafDataset._hash_image(src, 8, None)
    h_copy = LeafDataset._hash_image(str(copy), 8, None)
    h_other = LeafDataset._hash_image(str(other), 8, None)
    assert h_src == h_copy
    assert h_src != h_other


def test_split_sizes_disjoint_reproducible(tmp_leaf_dir, tmp_path):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], check_jpeg=False)
    ds.load_and_match()
    path = str(tmp_path / "split.npz")
    a = TrainValTestSplit.make(ds, seed=42, path=path)
    a.assert_disjoint()
    sizes = a.sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == 48
    assert sizes["train"] >= 32
    assert sizes["val"] >= 6
    assert sizes["test"] >= 6
    b = TrainValTestSplit.make(ds, seed=42)
    np.testing.assert_array_equal(a.train_idx, b.train_idx)
    np.testing.assert_array_equal(a.val_idx, b.val_idx)
    loaded = TrainValTestSplit.load(path)
    np.testing.assert_array_equal(loaded.test_idx, a.test_idx)
    y = ds.frame["predominant_stress"].values
    assert set(y[a.train_idx]).issubset(set(range(6)))
    assert set(y[a.val_idx]).issubset(set(range(6)))
    assert set(y[a.test_idx]).issubset(set(range(6)))


def test_copy_missing_leaf_jpgs_from_plain_and_roboflow_names(tmp_leaf_dir, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    src = os.path.join(tmp_leaf_dir["images"], "1.jpg")
    shutil.copy(src, extra / "9999.jpg")
    shutil.copy(src, extra / "1_jpg.rf.deadbeef.jpg")
    n = copy_missing_leaf_jpgs(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], [str(extra)])
    assert n == 1
    assert os.path.isfile(os.path.join(tmp_leaf_dir["images"], "9999.jpg"))
    n2 = copy_missing_leaf_jpgs(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], [str(extra)])
    assert n2 == 0
    assert _leaf_id_from_filename("1007_jpg.rf.abc.jpg") == 1007
    assert _leaf_id_from_filename("12.jpg") == 12
    assert _leaf_id_from_filename("notes.txt") is None
