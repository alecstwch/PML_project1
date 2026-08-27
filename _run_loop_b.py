"""Loop B: EfficientNet-B0 only, HSV-S leaf crop at ingest. Do not re-run SVM/XGBoost/ResNet."""

from __future__ import annotations

import csv
import json
import os
import time

from p1_core import (
    EfficientNetPipeline,
    LeafDataset,
    TrainValTestSplit,
    report_compute,
    resolve_paths,
)


# Last committed official EfficientNet row (pre-HSV-crop).
BASELINE = {"test_acc": 0.767175572519084, "macro_f1": 0.6517284641397751}


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
    report_compute()

    dataset = LeafDataset(paths["csv_path"], paths["image_dir"], check_jpeg=True)
    dataset.load_and_match()
    split = TrainValTestSplit.load(os.path.join(paths["features_dir"], "split_ids.npz"))
    print("n_kept", len(dataset.frame), "split", split.sizes())

    pipe = EfficientNetPipeline(
        dataset,
        split,
        name="efficientnet_b0",
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        models_dir=paths["models_dir"],
        use_mixup=True,
        ingest_hsv_crop=True,
        epochs_head=12,
        epochs_ft=20,
        batch_size=16,
        search_grid=[
            {"lr": 1e-4, "dropout": 0.3, "unfreeze": 20},
            {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20},
        ],
    )
    t0 = time.perf_counter()
    metrics = pipe.run()
    elapsed = time.perf_counter() - t0
    params = dict(pipe.best_params)
    params["hsv_crop"] = True
    pipe.best_params = params
    print("best", pipe.best_params)
    print("val", pipe.val_metrics)
    print("test", metrics, f"in {elapsed:.1f}s")

    row = {
        "model": "EfficientNet-B0",
        "features": "raw images",
        "val_acc": str(pipe.val_metrics.get("acc")),
        "test_acc": str(metrics.get("acc")),
        "macro_f1": str(metrics.get("macro_f1")),
        "best_params": str(pipe.best_params),
    }
    _upsert_row(os.path.join(paths["results_dir"], "final_results.csv"), row)

    acc_ok = metrics["acc"] + 1e-9 >= BASELINE["test_acc"]
    f1_ok = metrics["macro_f1"] + 1e-9 >= BASELINE["macro_f1"]
    summary = {
        "elapsed_s": elapsed,
        "test_acc": metrics["acc"],
        "macro_f1": metrics["macro_f1"],
        "baseline_acc": BASELINE["test_acc"],
        "baseline_f1": BASELINE["macro_f1"],
        "acc_ok": acc_ok,
        "f1_ok": f1_ok,
        "commit": bool(acc_ok and f1_ok),
        "best_params": pipe.best_params,
    }
    out = os.path.join(paths["results_dir"], "loop_b_effnet.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
