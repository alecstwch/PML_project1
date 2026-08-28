"""Dataset loading, EDA counts, and the 70/15/15 split."""

import os

import numpy as np

from p1_core import LeafDataset, TrainValTestSplit


def test_load_and_match_keeps_all_files(tmp_leaf_dir):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"])
    df = ds.load_and_match()
    assert "image_path" in df.columns
    assert "predominant_stress" in df.columns
    assert ds.n_csv == 48
    assert len(df) == 48
    assert df["image_path"].apply(os.path.exists).all()
    assert set(df["predominant_stress"].unique()) == set(range(6))


def test_eda_class_counts(tmp_leaf_dir):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"])
    ds.load_and_match()
    summary = ds.eda_summary()
    assert summary["n_kept"] == 48
    assert summary["drop_mixed"] is False
    assert summary["class_counts"][0] == 8
    assert summary["class_counts"][5] == 8
    counts = ds.class_counts()
    assert int(counts.loc[2]) == 8


def test_drop_mixed_removes_class_5_and_split_is_five_way(tmp_leaf_dir, tmp_path):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"], drop_mixed=True)
    df = ds.load_and_match()
    assert ds.n_csv == 48
    assert len(df) == 40
    assert 5 not in set(df["predominant_stress"].unique())
    assert set(df["predominant_stress"].unique()) == set(range(5))
    summary = ds.eda_summary()
    assert summary["drop_mixed"] is True
    assert summary["n_kept"] == 40
    assert 5 not in summary["class_counts"]

    split = TrainValTestSplit.make(ds, seed=42, path=str(tmp_path / "split.npz"))
    split.assert_disjoint()
    sizes = split.sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == 40
    y = ds.frame["predominant_stress"].values
    assert set(y[split.train_idx]) <= set(range(5))
    assert set(y[split.val_idx]) <= set(range(5))
    assert set(y[split.test_idx]) <= set(range(5))
    assert 5 not in y[split.train_idx]
    assert 5 not in y[split.val_idx]
    assert 5 not in y[split.test_idx]


def test_split_sizes_disjoint_reproducible(tmp_leaf_dir, tmp_path):
    ds = LeafDataset(tmp_leaf_dir["csv"], tmp_leaf_dir["images"])
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
