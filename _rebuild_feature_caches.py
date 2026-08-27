"""Rebuild handcrafted and frozen-deep caches to match the current CSV rows and split."""

from __future__ import annotations

import os
import time

from p1_core import (
    FrozenDeepFeatures,
    HandcraftedFeatures,
    LeafDataset,
    RANDOM_STATE,
    TrainValTestSplit,
    report_compute,
    resolve_paths,
)


def main() -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    paths = resolve_paths()
    report_compute()

    dataset = LeafDataset(paths["csv_path"], paths["image_dir"], check_jpeg=True)
    dataset.load_and_match()
    n = len(dataset.frame)
    print(f"n_csv={dataset.n_csv} n_kept={n} missing={dataset.n_missing} corrupt={dataset.n_corrupt}")
    print("corrupt ids:", dataset.corrupt_ids)

    split_path = os.path.join(paths["features_dir"], "split_ids.npz")
    split = None
    if os.path.isfile(split_path):
        split = TrainValTestSplit.load(split_path)
        n_split = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
        if n_split != n:
            print(f"Split has {n_split} rows, dataset has {n}; remaking the split.")
            os.remove(split_path)
            split = None
        else:
            print("Reusing split:", split.sizes())
    if split is None:
        split = TrainValTestSplit.make(dataset, seed=RANDOM_STATE, path=split_path)
        print("Wrote split:", split.sizes())
    split.assert_disjoint()

    for fname in ("handcrafted.npz", "frozen_deep.npz"):
        p = os.path.join(paths["features_dir"], fname)
        if os.path.isfile(p):
            os.remove(p)
            print("Deleted", p)

    t0 = time.perf_counter()
    hc_path = os.path.join(paths["features_dir"], "handcrafted.npz")
    pack_hc = HandcraftedFeatures(hc_path, image_size=224).extract(dataset, split)
    print("Handcrafted:", pack_hc.X_train.shape, pack_hc.X_val.shape, pack_hc.X_test.shape,
          f"in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    deep_path = os.path.join(paths["features_dir"], "frozen_deep.npz")
    pack_deep = FrozenDeepFeatures(deep_path, input_size=224, batch_size=32).extract(dataset, split)
    print("Frozen deep:", pack_deep.X_train.shape, pack_deep.X_val.shape, pack_deep.X_test.shape,
          f"in {time.perf_counter() - t1:.1f}s")

    assert pack_hc.X_train.shape[0] == len(split.train_idx)
    assert pack_deep.X_test.shape[0] == len(split.test_idx)
    print("Caches match the split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
