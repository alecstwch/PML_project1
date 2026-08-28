"""Patch P1_notebook.ipynb (strip Ninja) and write P1_no_mixed_notebook.ipynb."""

from __future__ import annotations

import copy
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "P1_notebook.ipynb")
DST6 = SRC
DST5 = os.path.join(ROOT, "P1_no_mixed_notebook.ipynb")


def to_source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]


def set_cell(nb, idx: int, text: str) -> None:
    nb["cells"][idx]["source"] = to_source(text)


def cell_src(nb, idx: int) -> str:
    src = nb["cells"][idx]["source"]
    return "".join(src) if isinstance(src, list) else src


def clear_outputs(nb) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


IMPORTS_6 = '''
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from p1_core import (
    RANDOM_STATE, STRESS_NAMES, BRACOL_CLASSES, BRACOL_TO_STRESS,
    resolve_paths, save_and_show, report_compute,
    LeafDataset, TrainValTestSplit,
    HandcraftedFeatures, FrozenDeepFeatures, PCAFeatures, scale_pack,
    SVMPipeline, XGBoostPipeline, EfficientNetPipeline, ResNetPipeline,
    ResidualMixedRule, fit_disease_probe, apply_residual_rule,
    load_bracol_labels, ensemble_average, classification_metrics,
    target_names_for,
)

rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PATHS = resolve_paths()
print("Data:", PATHS["data_dir"])
print("Figures:", PATHS["figures_dir"])
print("CSV exists:", os.path.isfile(PATHS["csv_path"]))
print("Image folder exists:", os.path.isdir(PATHS["image_dir"]))
report_compute()
'''

MD_LOAD_6 = """
## 2. Dataset corridor — load the complete table

Every CSV row has a matching `{id}.jpg` (1,747 images). Mixed (class 5) is kept for this official 6-class run. The Esgario-style 5-class experiment lives in `P1_no_mixed_notebook.ipynb`.
"""

LOAD_6 = '''
dataset = LeafDataset(PATHS["csv_path"], PATHS["image_dir"], drop_mixed=False)
df = dataset.load_and_match()
summary = dataset.eda_summary()
print(json.dumps({k: v for k, v in summary.items() if k != "class_counts"}, indent=2))
print("Class counts:", summary["class_counts"])
print(df.head())
'''

MD_DUP_6 = """
A perceptual-hash scan on the current leaf files found no duplicate groups, so that check is omitted from the run.
"""

SVM_HC_6 = '''
svm_hc = SVMPipeline(
    pack_hc, name="svm_handcrafted",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=False,
    C_grid=[0.1, 1, 10, 100],
    gamma_grid=["scale"],
    search_linear_first=True,
    skip_default_fit=True,
)
svm_hc.run()
record("SVM", "handcrafted", svm_hc)
print(svm_hc.best_params)
'''

LIT_6 = """
## 12. Final results table and literature

Esgario et al. (2020), *Computers and Electronics in Agriculture*:

- Leaf **classification** set: 1,685 images, **five** classes (they dropped 62 mixed-equal leaves).
- Split: 70 / 15 / 15 (not 5-fold).
- ResNet50 single-task **95.63%** accuracy; multi-task **95.24%**, precision 95.29%, recall 91.14%.
- They crop the leaf with an HSV-S threshold before 224×224.

This notebook is the **6-class official** run (Mixed kept, 1,747 images). Dataset load is complete: every CSV row has a JPEG. A matched 5-class protocol is `P1_no_mixed_notebook.ipynb`. Remaining gap vs 95.63% is protocol / split / crop, not missing files.
"""

CONCL_6 = """
## 13. Conclusions

- Two representations (handcrafted vs frozen CNN embeddings) and two official models (SVM, EfficientNet) are implemented end to end, each with its own ingest → fit → search → refit → report loop.
- The split is shared. Test is not used to pick hyperparameters.
- Mixed (class 5) is kept. The residual rule is an extra, not a replacement of the official 6-class task.
- ResNet-50 is the designated third method and the closest architecture family to Esgario et al. Numbers in this notebook are not directly comparable (6 vs 5 classes). See `P1_no_mixed_notebook.ipynb` for the 5-class protocol.
- MixUp follows Esgario's augmentation experiments; it is not presented as a new idea.

Next cell writes a compact JSON of the run so the report can copy numbers without re-typing them.
"""

PAYLOAD_6 = '''
payload = {
    "n_csv": summary["n_csv"],
    "n_kept": summary["n_kept"],
    "drop_mixed": summary["drop_mixed"],
    "split": split.sizes(),
    "results": results_rows(),
}
with open(os.path.join(PATHS["results_dir"], "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, default=str)
print("Wrote results/run_summary.json")
'''

TITLE_5 = """
# Project 1 (5-class): Esgario-style labels, Mixed dropped

**Not the official artefact.** The graded 6-class run is `P1_notebook.ipynb`.

This notebook drops `predominant_stress == 5` (Mixed) **before** the split: **1,685** images, five classes (Healthy, Miner, Rust, Phoma, Cercospora). Feature caches and result/figure tags use `_nomix` so 6-class files are not overwritten.

**Comparison target:** Esgario et al. (2020) leaf ResNet50 **95.63%** accuracy on 1,685 images, 70/15/15, HSV-S crop. We still use a random stratified split and a simpler HSV-S crop (no specialist lesion crops), so a remaining gap is expected.
"""

MD_LOAD_5 = """
## 2. Dataset corridor — drop Mixed (class 5)

CSV has 1,747 rows. After dropping Mixed we keep **1,685** images and five labels, matching Esgario's leaf classification count (they dropped 62 mixed-equal leaves).
"""

IMPORTS_5 = '''
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from p1_core import (
    RANDOM_STATE, STRESS_NAMES, BRACOL_CLASSES, BRACOL_TO_STRESS,
    resolve_paths, save_and_show, report_compute,
    LeafDataset, TrainValTestSplit,
    HandcraftedFeatures, FrozenDeepFeatures, PCAFeatures, scale_pack,
    SVMPipeline, XGBoostPipeline, EfficientNetPipeline, ResNetPipeline,
    load_bracol_labels, ensemble_average, classification_metrics,
    target_names_for,
)

rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PATHS = resolve_paths()
TAG = "_nomix"
print("Data:", PATHS["data_dir"])
print("Figures:", PATHS["figures_dir"])
print("CSV exists:", os.path.isfile(PATHS["csv_path"]))
print("Image folder exists:", os.path.isdir(PATHS["image_dir"]))
print("Protocol: 5-class, drop Mixed, tag", TAG)
report_compute()
'''

LOAD_5 = '''
dataset = LeafDataset(PATHS["csv_path"], PATHS["image_dir"], drop_mixed=True)
df = dataset.load_and_match()
summary = dataset.eda_summary()
print(json.dumps({k: v for k, v in summary.items() if k != "class_counts"}, indent=2))
print("Class counts:", summary["class_counts"])
print("n_kept (expect 1685):", summary["n_kept"])
print(df.head())
assert 5 not in set(df["predominant_stress"].unique())
'''

EDA_COUNTS_5 = '''
counts = dataset.class_counts()
fig, ax = plt.subplots(figsize=(8, 4))
names = [STRESS_NAMES.get(int(i), str(i)) for i in counts.index]
ax.bar(names, counts.values, color="#3d7a4a")
ax.set_ylabel("images")
ax.set_title("Class distribution (5-class, Mixed dropped)")
for i, v in enumerate(counts.values):
    ax.text(i, v + 2, str(int(v)), ha="center", fontsize=8)
fig.tight_layout()
save_and_show(fig, "class_distribution_nomix.png", PATHS["figures_dir"])
print("Saved class_distribution_nomix.png")
'''

SAMPLES_5 = '''
labels = sorted(df["predominant_stress"].unique())
fig, axes = plt.subplots(2, len(labels), figsize=(14, 5))
for col, label in enumerate(labels):
    sub = df[df["predominant_stress"] == label]
    for row in range(2):
        ax = axes[row, col]
        sample = sub.iloc[row % len(sub)]
        img = plt.imread(sample["image_path"])
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(STRESS_NAMES.get(int(label), str(label)), fontsize=9)
fig.suptitle("Two sample leaves per class (5-class)")
fig.tight_layout()
save_and_show(fig, "sample_images_nomix.png", PATHS["figures_dir"])
'''

HEAT_5 = '''
ct = pd.crosstab(df["predominant_stress"].map(STRESS_NAMES), df["severity"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Predominant stress vs severity (5-class)")
fig.tight_layout()
save_and_show(fig, "eda_heatmaps_nomix.png", PATHS["figures_dir"])
'''

SPLIT_5 = '''
split_path = os.path.join(PATHS["features_dir"], "split_ids_nomix.npz")
hc_name, deep_name = "handcrafted_nomix.npz", "frozen_deep_nomix.npz"
split = None
if os.path.exists(split_path):
    split = TrainValTestSplit.load(split_path)
    n_split = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    if n_split != len(dataset.frame):
        print(f"Split has {n_split} rows, dataset has {len(dataset.frame)}; rebuilding.")
        os.remove(split_path)
        split = None
        for fname in (hc_name, deep_name):
            p = os.path.join(PATHS["features_dir"], fname)
            if os.path.isfile(p):
                os.remove(p)
                print("Deleted stale feature cache:", p)
if split is None:
    split = TrainValTestSplit.make(dataset, seed=RANDOM_STATE, path=split_path)
split.assert_disjoint()
print("Sizes:", split.sizes())

for name, idx in [("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)]:
    dist = df.iloc[idx]["predominant_stress"].value_counts(normalize=True).sort_index()
    print(name, {STRESS_NAMES[int(k)]: round(float(v), 3) for k, v in dist.items()})
'''

HC_5 = '''
hc_path = os.path.join(PATHS["features_dir"], "handcrafted_nomix.npz")
fe1 = HandcraftedFeatures(hc_path, image_size=224)
pack_hc = fe1.extract(dataset, split)
print("Handcrafted shapes:", pack_hc.X_train.shape, pack_hc.X_val.shape, pack_hc.X_test.shape)
'''

DEEP_5 = '''
deep_path = os.path.join(PATHS["features_dir"], "frozen_deep_nomix.npz")
fe2 = FrozenDeepFeatures(deep_path, input_size=224, batch_size=32)
pack_deep = fe2.extract(dataset, split)
print("Deep shapes:", pack_deep.X_train.shape, pack_deep.X_val.shape, pack_deep.X_test.shape)
'''

SVM_HC_5 = '''
svm_hc = SVMPipeline(
    pack_hc, name="svm_handcrafted_nomix",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=False,
    C_grid=[0.1, 1, 10, 100],
    gamma_grid=["scale"],
    search_linear_first=True,
    skip_default_fit=True,
)
svm_hc.run()
record("SVM", "handcrafted", svm_hc)
print(svm_hc.best_params)
'''

SVM_DEEP_5 = '''
svm_deep = SVMPipeline(
    pack_deep, name="svm_deep_nomix",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=True,
)
svm_deep.run()
record("SVM", "deep (frozen EffNet)", svm_deep)
print(svm_deep.best_params)
'''

SVM_LIN_5 = '''
linear_rows = [r for r in svm_deep.search_table if r["kernel"] == "linear"]
best_linear = max(linear_rows, key=lambda r: r["val_acc"])
print("Best linear SVM on deep features (val):", best_linear)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

scaled_deep, _ = scale_pack(pack_deep)
lin = SVC(kernel="linear", C=best_linear["C"], class_weight="balanced")
lin.fit(scaled_deep.X_train, scaled_deep.y_train)
pred_lin = lin.predict(scaled_deep.X_test)
pred_lin_val = lin.predict(scaled_deep.X_val)
record(
    "SVM-linear (ablation)",
    "deep (frozen EffNet)",
    val_acc=float(accuracy_score(scaled_deep.y_val, pred_lin_val)),
    test_acc=float(accuracy_score(scaled_deep.y_test, pred_lin)),
    macro_f1=float(f1_score(scaled_deep.y_test, pred_lin, average="macro", zero_division=0)),
    best_params={"kernel": "linear", "C": best_linear["C"]},
)
svm_deep.report(scaled_deep.y_test, pred_lin, "svm_linear_deep_nomix")
'''

SVM_PCA_5 = '''
svm_pca = SVMPipeline(
    pack_pca, name="svm_pca256_nomix",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=False,
)
svm_pca.run()
record("SVM", "deep PCA-256", svm_pca)
'''

EFFNET_5 = '''
effnet = EfficientNetPipeline(
    dataset, split,
    name="efficientnet_b0_nomix",
    results_dir=PATHS["results_dir"],
    figures_dir=PATHS["figures_dir"],
    models_dir=PATHS["models_dir"],
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
record("EfficientNet-B0", "raw images", effnet)
print("EffNet best params:", effnet.best_params)
'''

XGB_5 = '''
xgb_hc = XGBoostPipeline(
    pack_hc, name="xgb_handcrafted_nomix",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
)
xgb_hc.run()
record("XGBoost", "handcrafted", xgb_hc)

xgb_deep = XGBoostPipeline(
    pack_deep, name="xgb_deep_nomix",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
)
xgb_deep.run()
record("XGBoost", "deep (frozen EffNet)", xgb_deep)
'''

RESNET_5 = '''
resnet = ResNetPipeline(
    dataset, split,
    name="resnet50_nomix",
    results_dir=PATHS["results_dir"],
    figures_dir=PATHS["figures_dir"],
    models_dir=PATHS["models_dir"],
    use_mixup=True,
    ingest_hsv_crop=True,
    epochs_head=10,
    epochs_ft=16,
    batch_size=16,
    search_grid=[
        {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20},
    ],
)
resnet.run()
record("ResNet-50", "raw images", resnet)
'''

NOMIXUP_5 = '''
effnet_nomixup = EfficientNetPipeline(
    dataset, split,
    name="efficientnet_b0_nomixup_nomix",
    results_dir=PATHS["results_dir"],
    figures_dir=PATHS["figures_dir"],
    models_dir=PATHS["models_dir"],
    use_mixup=False,
    ingest_hsv_crop=True,
    epochs_head=8,
    epochs_ft=12,
    batch_size=16,
    search_grid=[{"lr": 1e-5, "dropout": 0.3, "unfreeze": 20}],
)
effnet_nomixup.ingest()
effnet_nomixup.fit()
effnet_nomixup.best_params = {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20, "mixup": False}
effnet_nomixup.refit()
effnet_nomixup.test_metrics = effnet_nomixup.report(
    effnet_nomixup.y_true_test, effnet_nomixup.y_pred_test, effnet_nomixup.name
)
record("EfficientNet-B0 (no MixUp)", "raw images", effnet_nomixup)
'''

ENS_5 = '''
from sklearn.metrics import accuracy_score, f1_score

proba_svm = svm_deep.proba_test
proba_eff = effnet.proba_test
proba_ens = ensemble_average(proba_svm, proba_eff)
y_ens = np.argmax(proba_ens, axis=1)
y_test = svm_deep.y_true_test
ens_metrics = classification_metrics(y_test, y_ens)
print("Ensemble SVM-deep + EffNet:", ens_metrics)
svm_deep.report(y_test, y_ens, "ensemble_svm_effnet_nomix")
record(
    "Ensemble",
    "SVM-deep + EffNet",
    val_acc=None,
    test_acc=ens_metrics["acc"],
    macro_f1=ens_metrics["macro_f1"],
    best_params={"combine": "mean_proba"},
)

if resnet.proba_test is not None:
    proba_3 = ensemble_average(proba_svm, proba_eff, resnet.proba_test)
    y3 = np.argmax(proba_3, axis=1)
    m3 = classification_metrics(y_test, y3)
    print("Ensemble + ResNet:", m3)
    record(
        "Ensemble",
        "SVM-deep + EffNet + ResNet",
        val_acc=None,
        test_acc=m3["acc"],
        macro_f1=m3["macro_f1"],
        best_params={"combine": "mean_proba"},
    )
'''

MD_SKIP_RULE = """
### 10.5 Residual Mixed rule — skipped

No class 5 in this notebook, so the residual Mixed rule is not run.
"""

BRACOL_5 = '''
bracol = load_bracol_labels(PATHS["bracol_dir"])
print("BRACOL images with labels:", len(bracol))

test_ids = df.iloc[split.test_idx]["id"].values
best_preds = y_ens if "y_ens" in dir() else svm_deep.y_pred_test
y_true = svm_deep.y_true_test

rows_cmp = []
for i, img_id in enumerate(test_ids):
    if img_id not in bracol:
        continue
    dets = bracol[img_id]
    if len(dets) == 0:
        bracol_lab = 0
    else:
        most = max(set(dets), key=dets.count)
        bracol_lab = BRACOL_TO_STRESS.get(most, -1)
    rows_cmp.append({
        "id": int(img_id),
        "true": STRESS_NAMES.get(int(y_true[i]), str(y_true[i])),
        "ours": STRESS_NAMES.get(int(best_preds[i]), str(best_preds[i])),
        "bracol": STRESS_NAMES.get(int(bracol_lab), str(bracol_lab)),
        "n_det": len(dets),
        "multi_lesion": len(set(dets)) > 1,
        "agree": int(bracol_lab == best_preds[i]),
    })

comp = pd.DataFrame(rows_cmp)
if len(comp):
    print("Compared:", len(comp), "agreement:", float(comp["agree"].mean()))
    print("Errors where YOLO has multiple lesion types:",
          int(((comp["agree"] == 0) & comp["multi_lesion"]).sum()))
    comp.to_csv(os.path.join(PATHS["results_dir"], "bracol_crosscheck_nomix.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].pie(
        [comp["agree"].sum(), (1 - comp["agree"]).sum()],
        labels=["Agree", "Disagree"], autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    axes[0].set_title("Predictions vs BRACOL majority box (5-class)")
    disagree = comp[comp["agree"] == 0]
    if len(disagree):
        vc = disagree["true"].value_counts()
        axes[1].bar(vc.index.astype(str), vc.values, color="#e74c3c")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_title("Disagreements by true class")
    fig.tight_layout()
    save_and_show(fig, "bracol_crosscheck_nomix.png", PATHS["figures_dir"])
else:
    print("No overlap with BRACOL labels. Folder missing or different filenames.")
'''

LIT_5 = """
## 12. Final results table and literature

Esgario et al. (2020) leaf ResNet50: **95.63%** accuracy on **1,685** images, **five** classes, 70/15/15, HSV-S crop.

This run matches the class set and image count (Mixed dropped). It does **not** copy their exact crop, MixUp recipe, or split draw, so accuracy is expected to sit below 95.63%. The Mixed-class gap is closed for this experiment; remaining gap is protocol / split / crop.
"""

FINAL_5 = '''
results_df = pd.DataFrame(results_rows())
results_df.to_csv(os.path.join(PATHS["results_dir"], "final_results_nomix.csv"), index=False)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
plot_df = results_df.dropna(subset=["test_acc"]).copy()
labels = plot_df["model"] + " / " + plot_df["features"]
ax.barh(range(len(plot_df)), plot_df["test_acc"], color="#3d7a4a")
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("test accuracy")
ax.set_xlim(0, 1)
ax.set_title("Test accuracy by configuration (5-class)")
fig.tight_layout()
save_and_show(fig, "final_accuracy_bars_nomix.png", PATHS["figures_dir"])
'''

CONCL_5 = """
## 13. Conclusions

- Same two representations and the same grading loop as the 6-class notebook, on Esgario's five-class label set (1,685 images).
- Official 6-class numbers remain in `P1_notebook.ipynb` / `results/final_results.csv`.
- Residual Mixed rule is not used (no class 5).
- Compare ResNet-50 / EfficientNet here to Esgario's 95.63% with the protocol caveats in section 12.
"""

PAYLOAD_5 = '''
payload = {
    "n_csv": summary["n_csv"],
    "n_kept": summary["n_kept"],
    "drop_mixed": summary["drop_mixed"],
    "split": split.sizes(),
    "results": results_rows(),
}
with open(os.path.join(PATHS["results_dir"], "run_summary_nomix.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, default=str)
print("Wrote results/run_summary_nomix.json")
'''


def patch_six(nb) -> None:
    set_cell(nb, 2, IMPORTS_6)
    set_cell(nb, 3, MD_LOAD_6)
    set_cell(nb, 4, LOAD_6)
    set_cell(nb, 10, MD_DUP_6)
    set_cell(nb, 21, SVM_HC_6)
    set_cell(nb, 42, LIT_6)
    set_cell(nb, 44, CONCL_6)
    set_cell(nb, 45, PAYLOAD_6)


def patch_five(nb) -> None:
    set_cell(nb, 0, TITLE_5)
    set_cell(nb, 2, IMPORTS_5)
    set_cell(nb, 3, MD_LOAD_5)
    set_cell(nb, 4, LOAD_5)
    set_cell(nb, 6, EDA_COUNTS_5)
    set_cell(nb, 7, SAMPLES_5)
    set_cell(nb, 9, HEAT_5)
    set_cell(nb, 10, MD_DUP_6)
    set_cell(nb, 12, SPLIT_5)
    set_cell(nb, 14, HC_5)
    set_cell(nb, 16, DEEP_5)
    set_cell(nb, 21, SVM_HC_5)
    set_cell(nb, 22, SVM_DEEP_5)
    set_cell(nb, 23, SVM_LIN_5)
    set_cell(nb, 25, SVM_PCA_5)
    set_cell(nb, 28, EFFNET_5)
    set_cell(nb, 31, XGB_5)
    set_cell(nb, 33, RESNET_5)
    set_cell(nb, 35, NOMIXUP_5)
    set_cell(nb, 37, ENS_5)
    set_cell(nb, 38, MD_SKIP_RULE)
    set_cell(nb, 39, "print('Residual Mixed rule skipped (no class 5).')")
    set_cell(nb, 41, BRACOL_5)
    set_cell(nb, 42, LIT_5)
    set_cell(nb, 43, FINAL_5)
    set_cell(nb, 44, CONCL_5)
    set_cell(nb, 45, PAYLOAD_5)


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        nb6 = json.load(f)
    patch_six(nb6)
    with open(DST6, "w", encoding="utf-8") as f:
        json.dump(nb6, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Patched", DST6, "cells", len(nb6["cells"]))

    nb5 = copy.deepcopy(nb6)
    patch_five(nb5)
    clear_outputs(nb5)
    with open(DST5, "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Wrote", DST5, "cells", len(nb5["cells"]))


if __name__ == "__main__":
    main()
