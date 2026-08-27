"""Loop A: rebuild handcrafted features (HOG 16x16) and fit only the handcrafted SVM."""

from __future__ import annotations

import csv
import json
import os
import time

from p1_core import (
    HOG_CELL,
    HandcraftedFeatures,
    LeafDataset,
    SVMPipeline,
    TrainValTestSplit,
    resolve_paths,
)


BASELINE = {"test_acc": 0.6221374045801527, "macro_f1": 0.47043616682337924}


def _upsert_row(csv_path: str, row: dict) -> None:
    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    fieldnames = ["model", "features", "val_acc", "test_acc", "macro_f1", "best_params"]
    replaced = False
    for i, old in enumerate(rows):
        if old.get("model") == row["model"] and old.get("features") == row["features"]:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    paths = resolve_paths()
    print("HOG_CELL", HOG_CELL)
    if tuple(HOG_CELL) != (16, 16):
        raise SystemExit(f"Expected HOG_CELL (16, 16), got {HOG_CELL}")

    dataset = LeafDataset(paths["csv_path"], paths["image_dir"], check_jpeg=True)
    dataset.load_and_match()
    split = TrainValTestSplit.load(os.path.join(paths["features_dir"], "split_ids.npz"))
    n_split = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    print("n_kept", len(dataset.frame), "split", split.sizes())
    if n_split != len(dataset.frame):
        raise SystemExit("Split does not match kept rows; rebuild caches first.")

    hc_path = os.path.join(paths["features_dir"], "handcrafted.npz")
    if os.path.isfile(hc_path):
        os.remove(hc_path)
        print("Deleted", hc_path)

    t_extract = time.perf_counter()
    pack = HandcraftedFeatures(hc_path, image_size=224).extract(dataset, split)
    extract_s = time.perf_counter() - t_extract
    print("Handcrafted shapes:", pack.X_train.shape, pack.X_val.shape, pack.X_test.shape, f"extract {extract_s:.1f}s")

    pipe = SVMPipeline(
        pack,
        name="svm_handcrafted",
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        need_proba=False,
        C_grid=[0.1, 1, 10, 100],
        gamma_grid=["scale"],
        search_linear_first=True,
        skip_default_fit=True,
    )
    t0 = time.perf_counter()
    metrics = pipe.run()
    svm_s = time.perf_counter() - t0
    print("SVM seconds", svm_s)
    print("best", pipe.best_params)
    print("val", pipe.val_metrics)
    print("test", metrics)

    row = {
        "model": "SVM",
        "features": "handcrafted",
        "val_acc": str(pipe.val_metrics.get("acc")),
        "test_acc": str(metrics.get("acc")),
        "macro_f1": str(metrics.get("macro_f1")),
        "best_params": str(pipe.best_params),
    }
    _upsert_row(os.path.join(paths["results_dir"], "final_results.csv"), row)

    acc_ok = metrics["acc"] + 1e-9 >= BASELINE["test_acc"]
    f1_ok = metrics["macro_f1"] + 1e-9 >= BASELINE["macro_f1"]
    faster = svm_s < 3600  # baseline search was ~100 minutes
    win = faster and (f1_ok or acc_ok)
    summary = {
        "extract_s": extract_s,
        "svm_s": svm_s,
        "test_acc": metrics["acc"],
        "macro_f1": metrics["macro_f1"],
        "baseline_acc": BASELINE["test_acc"],
        "baseline_f1": BASELINE["macro_f1"],
        "faster": faster,
        "acc_ok": acc_ok,
        "f1_ok": f1_ok,
        "commit": bool(win and f1_ok),
        "best_params": pipe.best_params,
    }
    out = os.path.join(paths["results_dir"], "loop_a_handcrafted.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
