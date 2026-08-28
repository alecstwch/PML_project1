"""CLI entry for Project 1. Default: 6-class official SVM-RBF + EfficientNet."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

from p1_core import (
    EfficientNetPipeline,
    FrozenDeepFeatures,
    HandcraftedFeatures,
    LeafDataset,
    RANDOM_STATE,
    ResNetPipeline,
    SVMPipeline,
    TrainValTestSplit,
    XGBoostPipeline,
    classification_metrics,
    ensemble_average,
    report_compute,
    resolve_paths,
)

RESULT_FIELDS = ["model", "features", "val_acc", "test_acc", "macro_f1", "best_params"]
KNOWN_MODELS = ("svm", "effnet", "resnet", "xgb")


def _parse_models(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--models is empty")
    out: List[str] = []
    for p in parts:
        if p == "all":
            for name in KNOWN_MODELS:
                if name not in out:
                    out.append(name)
            continue
        if p not in KNOWN_MODELS:
            raise SystemExit(f"Unknown model {p!r}. Choose from {', '.join(KNOWN_MODELS)}, all")
        if p not in out:
            out.append(p)
    return out


def _tag(drop_mixed: bool) -> str:
    return "_nomix" if drop_mixed else ""


def _row(model: str, features: str, pipe=None, **extra) -> Dict[str, object]:
    if pipe is not None:
        return {
            "model": model,
            "features": features,
            "val_acc": pipe.val_metrics.get("acc"),
            "test_acc": pipe.test_metrics.get("acc"),
            "macro_f1": pipe.test_metrics.get("macro_f1"),
            "best_params": pipe.best_params,
        }
    row = {"model": model, "features": features}
    row.update(extra)
    return row


def _write_results(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in RESULT_FIELDS})
    print(f"Wrote {path}")


def _print_row(row: Dict[str, object]) -> None:
    print(
        f"{row['model']:28s}  {str(row['features']):24s}  "
        f"val={row.get('val_acc')}  test={row.get('test_acc')}  f1={row.get('macro_f1')}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Project 1 learners. Default is the official 6-class pair: SVM-RBF + EfficientNet-B0.",
    )
    p.add_argument(
        "--drop-mixed",
        action="store_true",
        help="Esgario-style 5-class protocol: drop predominant_stress==5 before the split (1,685 images).",
    )
    p.add_argument(
        "--models",
        default="svm,effnet",
        help="Comma-separated list: svm,effnet,resnet,xgb,all. Default: svm,effnet (official pair).",
    )
    p.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost even if listed in --models.")
    p.add_argument("--skip-cnn", action="store_true", help="Skip EfficientNet and ResNet even if listed in --models.")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Folder that contains dataset.csv and images/. Default: ../coffee-datasets/coffee-datasets/leaf",
    )
    return p


def run(args: argparse.Namespace) -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    models = _parse_models(args.models)
    if args.skip_xgb:
        models = [m for m in models if m != "xgb"]
    if args.skip_cnn:
        models = [m for m in models if m not in ("effnet", "resnet")]
    if not models:
        raise SystemExit("Nothing to run after --skip-xgb / --skip-cnn.")

    drop_mixed = bool(args.drop_mixed)
    tag = _tag(drop_mixed)
    paths = resolve_paths(data_dir=args.data_dir)
    report_compute()

    dataset = LeafDataset(paths["csv_path"], paths["image_dir"], drop_mixed=drop_mixed)
    dataset.load_and_match()
    summary = dataset.eda_summary()
    print("n_csv", summary["n_csv"], "n_kept", summary["n_kept"], "drop_mixed", drop_mixed)
    print("class counts", summary["class_counts"])

    split_path = os.path.join(paths["features_dir"], f"split_ids{tag}.npz")
    hc_path = os.path.join(paths["features_dir"], f"handcrafted{tag}.npz")
    deep_path = os.path.join(paths["features_dir"], f"frozen_deep{tag}.npz")
    split = TrainValTestSplit.make(dataset, seed=RANDOM_STATE, path=split_path)
    split.assert_disjoint()
    print("split", split.sizes())

    need_features = any(m in models for m in ("svm", "xgb"))
    pack_hc = pack_deep = None
    if need_features:
        pack_hc = HandcraftedFeatures(hc_path, image_size=224).extract(dataset, split)
        print("handcrafted", pack_hc.X_train.shape)
        pack_deep = FrozenDeepFeatures(deep_path, input_size=224, batch_size=32).extract(dataset, split)
        print("frozen deep", pack_deep.X_train.shape)

    rows: List[Dict[str, object]] = []
    svm_deep = svm_hc = None
    effnet = resnet = None

    if "svm" in models:
        assert pack_hc is not None and pack_deep is not None
        svm_hc = SVMPipeline(
            pack_hc,
            name=f"svm_handcrafted{tag}",
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
            need_proba=False,
            C_grid=[0.1, 1, 10, 100],
            gamma_grid=["scale"],
            search_linear_first=True,
            skip_default_fit=True,
        )
        svm_hc.run()
        rows.append(_row("SVM", "handcrafted", svm_hc))
        _print_row(rows[-1])

        svm_deep = SVMPipeline(
            pack_deep,
            name=f"svm_deep{tag}",
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
            need_proba=True,
        )
        svm_deep.run()
        rows.append(_row("SVM", "deep (frozen EffNet)", svm_deep))
        _print_row(rows[-1])

    if "xgb" in models:
        assert pack_hc is not None and pack_deep is not None
        xgb_hc = XGBoostPipeline(
            pack_hc,
            name=f"xgb_handcrafted{tag}",
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
        )
        xgb_hc.run()
        rows.append(_row("XGBoost", "handcrafted", xgb_hc))
        _print_row(rows[-1])

        xgb_deep = XGBoostPipeline(
            pack_deep,
            name=f"xgb_deep{tag}",
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
        )
        xgb_deep.run()
        rows.append(_row("XGBoost", "deep (frozen EffNet)", xgb_deep))
        _print_row(rows[-1])

    if "effnet" in models:
        effnet = EfficientNetPipeline(
            dataset,
            split,
            name=f"efficientnet_b0{tag}",
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
        effnet.run()
        rows.append(_row("EfficientNet-B0", "raw images", effnet))
        _print_row(rows[-1])

    if "resnet" in models:
        resnet = ResNetPipeline(
            dataset,
            split,
            name=f"resnet50{tag}",
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
            models_dir=paths["models_dir"],
            use_mixup=True,
            ingest_hsv_crop=True,
            epochs_head=10,
            epochs_ft=16,
            batch_size=16,
            search_grid=[{"lr": 1e-5, "dropout": 0.3, "unfreeze": 20}],
        )
        resnet.run()
        rows.append(_row("ResNet-50", "raw images", resnet))
        _print_row(rows[-1])

    if svm_deep is not None and effnet is not None and svm_deep.proba_test is not None and effnet.proba_test is not None:
        y_ens = ensemble_average(svm_deep.proba_test, effnet.proba_test).argmax(axis=1)
        ens = classification_metrics(svm_deep.y_true_test, y_ens)
        rows.append(
            _row(
                "Ensemble",
                "SVM-deep + EffNet",
                val_acc=None,
                test_acc=ens["acc"],
                macro_f1=ens["macro_f1"],
                best_params={"combine": "mean_proba"},
            )
        )
        svm_deep.report(svm_deep.y_true_test, y_ens, f"ensemble_svm_effnet{tag}")
        _print_row(rows[-1])

    # Official 6-class notebook owns final_results.csv; CLI writes cli_results.csv.
    # Five-class uses the tagged name so it matches the nomix notebook.
    csv_name = "final_results_nomix.csv" if drop_mixed else "cli_results.csv"
    out_csv = os.path.join(paths["results_dir"], csv_name)
    _write_results(out_csv, rows)

    payload = {
        "n_csv": summary["n_csv"],
        "n_kept": summary["n_kept"],
        "drop_mixed": drop_mixed,
        "models": models,
        "split": split.sizes(),
        "results": rows,
    }
    json_name = f"cli_summary{tag}.json"
    with open(os.path.join(paths["results_dir"], json_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print("Done.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
