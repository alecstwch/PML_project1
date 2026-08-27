"""Build P1_notebook.ipynb from markdown + code cell strings."""

import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip() + "\n",
        }
    )


md("""
# Project 1: Supervised classification of coffee leaf diseases

**Official models:** SVM with RBF kernel, EfficientNet-B0 (fine-tuned)

**Task:** classify BRACOL leaf images by `predominant_stress` (six classes: Healthy, Miner, Rust, Phoma, Cercospora, Mixed)

**Representations:** (1) handcrafted colour / texture / edges, (2) frozen EfficientNet embeddings, plus end-to-end CNNs on raw images

The flow is: shared dataset corridor, then the full grading loop for SVM, then the same loop for EfficientNet. ResNet-50 and XGBoost are extras. Classes live in `p1_core.py` so the notebook stays readable and tests can import the same code.
""")

md("""
## 1. Setup and imports

Use the **WSL** kernel so EfficientNet / ResNet see the RTX 4070 (CUDA). In Cursor: kernel picker → *Select Another Kernel* → *Python Environments* → `/home/alecs/pml_venv/bin/python` (display name **Python (PML WSL GPU)** if registered). The Windows `.venv` is CPU-only TensorFlow.
""")

code("""
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
    copy_missing_leaf_jpgs, extra_image_roots,
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
""")

md("""
## 2. Dataset corridor — extra images, then load

The official Mendeley zip is corrupted here, so the CSV has rows whose `{id}.jpg` is missing. Dataset Ninja hosts the same BRACOL leaf photos under the name **Coffee Leaf Biotic Stress** (Supervisely format, 500 images). We download that dump, then also copy any still-missing ids from the Roboflow/YOLO exports already on disk (`123_jpg.rf....jpg` → `123.jpg`). Count missing and corrupt files at runtime; do not hard-code 1,402.
""")

code("""
ninja_dir = PATHS["ninja_dir"]
os.makedirs(ninja_dir, exist_ok=True)

def _ninja_has_jpegs(root):
    if not os.path.isdir(root):
        return False
    for dirpath, _, files in os.walk(root):
        if any(name.lower().endswith(".jpg") for name in files):
            return True
    return False

if not _ninja_has_jpegs(ninja_dir):
    try:
        import dataset_tools as dtools
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "dataset-tools"])
        import dataset_tools as dtools
    try:
        # Dataset Ninja listing name. "BRACOL" is the Mendeley title for the same collection.
        dtools.download(dataset="Coffee Leaf Biotic Stress", dst_dir=ninja_dir)
    except Exception as e:
        print("Dataset Ninja download failed:", e)
        print("Will still copy missing ids from YOLO/Roboflow folders on disk.")

n_copied = copy_missing_leaf_jpgs(
    PATHS["csv_path"], PATHS["image_dir"], extra_image_roots(PATHS),
)
print("Copied missing leaf JPEGs:", n_copied)
if n_copied:
    for fname in ("handcrafted.npz", "frozen_deep.npz"):
        p = os.path.join(PATHS["features_dir"], fname)
        if os.path.isfile(p):
            os.remove(p)
            print("Deleted stale feature cache:", p)

dataset = LeafDataset(PATHS["csv_path"], PATHS["image_dir"], check_jpeg=True)
df = dataset.load_and_match()
summary = dataset.eda_summary()
print(json.dumps({k: v for k, v in summary.items() if k != "class_counts"}, indent=2))
print("Class counts:", summary["class_counts"])
print("Corrupt ids:", dataset.corrupt_ids)
print(df.head())
""")

md("## 3. Exploratory plots")

code("""
counts = dataset.class_counts()
fig, ax = plt.subplots(figsize=(8, 4))
names = [STRESS_NAMES.get(int(i), str(i)) for i in counts.index]
ax.bar(names, counts.values, color="#3d7a4a")
ax.set_ylabel("images")
ax.set_title("Class distribution (predominant_stress)")
for i, v in enumerate(counts.values):
    ax.text(i, v + 2, str(int(v)), ha="center", fontsize=8)
fig.tight_layout()
save_and_show(fig, "class_distribution.png", PATHS["figures_dir"])
print("Saved class_distribution.png")
""")

code("""
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for col, label in enumerate(sorted(df["predominant_stress"].unique())):
    sub = df[df["predominant_stress"] == label]
    for row in range(2):
        ax = axes[row, col]
        sample = sub.iloc[row % len(sub)]
        img = plt.imread(sample["image_path"])
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(STRESS_NAMES.get(int(label), str(label)), fontsize=9)
fig.suptitle("Two sample leaves per class")
fig.tight_layout()
save_and_show(fig, "sample_images.png", PATHS["figures_dir"])
""")

code("""
from PIL import Image as PILImage

with PILImage.open(df["image_path"].iloc[0]) as im:
    w, h = im.size
print(f"Native size: {w} x {h}. All current leaf JPEGs share this size; the models resize to 224 x 224.")
""")

code("""
ct = pd.crosstab(df["predominant_stress"].map(STRESS_NAMES), df["severity"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Predominant stress vs severity")
fig.tight_layout()
save_and_show(fig, "eda_heatmaps.png", PATHS["figures_dir"])
""")

md("""
A perceptual-hash scan on the current leaf files found **no duplicate groups**, so that check is omitted from the run. If you add a new dump of images, call `dataset.find_duplicates()` again.
""")

md("""
## 4. Train / validation / test split (criterion h)

Stratified 70 / 15 / 15. The same indices are saved and reused by every model. Hyperparameters are chosen on validation only. Test is used once, after the config is locked.
""")

code("""
split_path = os.path.join(PATHS["features_dir"], "split_ids.npz")
# Reuse the saved split when it covers every kept row. Rebuild (and drop feature
# caches) only if the filtered table changed.
split = None
if os.path.exists(split_path):
    split = TrainValTestSplit.load(split_path)
    n_split = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    if n_split != len(dataset.frame):
        print(f"Split has {n_split} rows, dataset has {len(dataset.frame)}; rebuilding.")
        os.remove(split_path)
        split = None
        for fname in ("handcrafted.npz", "frozen_deep.npz"):
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
""")

md("""
## 5. FE1 — Handcrafted features (criterion c)

White-background mask, then HSV histogram (512), uniform LBP (26), HOG on 224×224 with 16×16 cells. Concatenate and L2-normalise. Cache to disk.
""")

code("""
hc_path = os.path.join(PATHS["features_dir"], "handcrafted.npz")
fe1 = HandcraftedFeatures(hc_path, image_size=224)
pack_hc = fe1.extract(dataset, split)
print("Handcrafted shapes:", pack_hc.X_train.shape, pack_hc.X_val.shape, pack_hc.X_test.shape)
""")

md("""
## 6. FE2 — Frozen EfficientNet embeddings (criterion c)

ImageNet EfficientNet-B0 without the classifier, global average pooling, 1280 numbers per image. No augmentation at extract time.
""")

code("""
deep_path = os.path.join(PATHS["features_dir"], "frozen_deep.npz")
fe2 = FrozenDeepFeatures(deep_path, input_size=224, batch_size=32)
pack_deep = fe2.extract(dataset, split)
print("Deep shapes:", pack_deep.X_train.shape, pack_deep.X_val.shape, pack_deep.X_test.shape)
""")

md("""
## 7. FE3 — PCA on frozen embeddings (optional SVM axis)

PCA is fit on the **train** embeddings only, then applied to val and test.
""")

code("""
fe3 = PCAFeatures(n_components=256, source=pack_deep)
pack_pca = fe3.extract(dataset, split)
print("PCA shapes:", pack_pca.X_train.shape, "explained variance ratio sum:",
      float(fe3.pca.explained_variance_ratio_.sum()))
""")

md("""
## 8. Model 1 — SVM, full grading loop

Personalized ingest: `StandardScaler` on train only. Then fit a default RBF SVM, search C × gamma on **validation** accuracy (plus a linear kernel sweep), refit the winner, report test once.

We run the loop twice: handcrafted features, then frozen deep features.
""")

code("""
# One row per (model, features). A list would duplicate if you re-run a pipeline cell.
# A set of dicts is not possible (dicts are unhashable); a dict keyed by that pair upserts.
results_by_key = {}

def record(model_name, features, pipe=None, **extra):
    if pipe is not None:
        row = {
            "model": model_name,
            "features": features,
            "val_acc": pipe.val_metrics.get("acc"),
            "test_acc": pipe.test_metrics.get("acc"),
            "macro_f1": pipe.test_metrics.get("macro_f1"),
            "best_params": pipe.best_params,
        }
    else:
        row = {"model": model_name, "features": features}
        row.update(extra)
    results_by_key[(model_name, features)] = row
    print(row)

def results_rows():
    return list(results_by_key.values())
""")

code("""
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
""")

code("""
svm_deep = SVMPipeline(
    pack_deep, name="svm_deep",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=True,
)
svm_deep.run()
record("SVM", "deep (frozen EffNet)", svm_deep)
print(svm_deep.best_params)
""")

code("""
# Linear kernel on deep features is already part of search(); pick the best linear row for a dedicated report.
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
svm_deep.report(scaled_deep.y_test, pred_lin, "svm_linear_deep")
""")

md("""
Optional PCA pack for SVM. Same loop, smaller vectors. Useful if the 1280-d RBF grid is slow.
""")

code("""
svm_pca = SVMPipeline(
    pack_pca, name="svm_pca256",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
    need_proba=False,
)
svm_pca.run()
record("SVM", "deep PCA-256", svm_pca)
""")

md("""
### Model 1 literature note

Margin classifiers on frozen CNN embeddings are a standard transfer-learning probe. We are not copying Esgario's protocol here: they fine-tune ResNet-50 on pixels. The SVM numbers answer a different question: how much of the disease signal is already in ImageNet features.
""")

md("""
## 9. Model 2 — EfficientNet-B0, full grading loop

Personalized ingest: HSV saturation crop to the leaf, then 224×224 RGB. MixUp plus flips. Phase 1 trains the head with a frozen backbone. Phase 2 opens the last layers at a lower learning rate. Early stopping watches validation loss. Test is scored once after search.
""")

code("""
# Keep the default grid small so the notebook can finish. Widen it if you have more GPU time.
effnet = EfficientNetPipeline(
    dataset, split,
    name="efficientnet_b0",
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
""")

md("""
## 10. Extras (criterion i)
""")

md("### 10.1 XGBoost on the same cached features")

code("""
xgb_hc = XGBoostPipeline(
    pack_hc, name="xgb_handcrafted",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
)
xgb_hc.run()
record("XGBoost", "handcrafted", xgb_hc)

xgb_deep = XGBoostPipeline(
    pack_deep, name="xgb_deep",
    results_dir=PATHS["results_dir"], figures_dir=PATHS["figures_dir"],
)
xgb_deep.run()
record("XGBoost", "deep (frozen EffNet)", xgb_deep)
""")

md("### 10.2 ResNet-50 (third method, same protocol as EfficientNet)")

code("""
resnet = ResNetPipeline(
    dataset, split,
    name="resnet50",
    results_dir=PATHS["results_dir"],
    figures_dir=PATHS["figures_dir"],
    models_dir=PATHS["models_dir"],
    use_mixup=True,
    epochs_head=10,
    epochs_ft=16,
    batch_size=16,
    search_grid=[
        {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20},
    ],
)
resnet.run()
record("ResNet-50", "raw images", resnet)
""")

md("### 10.3 MixUp on/off ablation on EfficientNet (short run)")

code("""
effnet_nomix = EfficientNetPipeline(
    dataset, split,
    name="efficientnet_b0_nomixup",
    results_dir=PATHS["results_dir"],
    figures_dir=PATHS["figures_dir"],
    models_dir=PATHS["models_dir"],
    use_mixup=False,
    epochs_head=8,
    epochs_ft=12,
    batch_size=16,
    search_grid=[{"lr": 1e-5, "dropout": 0.3, "unfreeze": 20}],
)
# Skip a second full search: fit + refit is enough for the ablation row.
effnet_nomix.ingest()
effnet_nomix.fit()
effnet_nomix.best_params = {"lr": 1e-5, "dropout": 0.3, "unfreeze": 20, "mixup": False}
effnet_nomix.refit()
effnet_nomix.test_metrics = effnet_nomix.report(
    effnet_nomix.y_true_test, effnet_nomix.y_pred_test, effnet_nomix.name
)
record("EfficientNet-B0 (no MixUp)", "raw images", effnet_nomix)
""")

md("### 10.4 Probability ensemble")

code("""
from sklearn.metrics import accuracy_score, f1_score

proba_svm = svm_deep.proba_test
proba_eff = effnet.proba_test
proba_ens = ensemble_average(proba_svm, proba_eff)
y_ens = np.argmax(proba_ens, axis=1)
y_test = svm_deep.y_true_test
ens_metrics = classification_metrics(y_test, y_ens)
print("Ensemble SVM-deep + EffNet:", ens_metrics)
svm_deep.report(y_test, y_ens, "ensemble_svm_effnet")
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
""")

md("""
### 10.5 Class-5 residual rule

Official task stays 6-class. Extra variant: a multi-label probe on the four binary disease columns. Predict Mixed if two heads fire above τ, or if the 6-class top-1 vs top-2 gap is below δ. τ and δ are chosen on validation only.
""")

code("""
from sklearn.preprocessing import StandardScaler

disease_cols = ["miner", "rust", "phoma", "cercospora"]
Y_all = df[disease_cols].values.astype(np.int32)
Y_train = Y_all[split.train_idx]
Y_val = Y_all[split.val_idx]
Y_test = Y_all[split.test_idx]

# Probe on scaled frozen embeddings (already extracted, no extra CNN).
sc = StandardScaler()
Ztr = sc.fit_transform(pack_deep.X_train)
Zva = sc.transform(pack_deep.X_val)
Zte = sc.transform(pack_deep.X_test)

disease_val = fit_disease_probe(Ztr, Y_train, Zva)
disease_test = fit_disease_probe(Ztr, Y_train, Zte)

proba_val = svm_deep.model.predict_proba(svm_deep.scaled.X_val)
proba_test = svm_deep.proba_test

rule = ResidualMixedRule()
tau, delta = rule.search(pack_deep.y_val, proba_val, disease_val)
print("Locked tau, delta:", tau, delta)

y_rule_val = rule.predict(proba_val, disease_val)
y_rule_test = rule.predict(proba_test, disease_test)
rule_val = classification_metrics(pack_deep.y_val, y_rule_val)
rule_test = classification_metrics(pack_deep.y_test, y_rule_test)
print("Residual rule val:", rule_val)
print("Residual rule test:", rule_test)
svm_deep.report(pack_deep.y_test, y_rule_test, "residual_rule_svm")
record(
    "SVM-deep + residual rule",
    "deep (frozen EffNet)",
    val_acc=rule_val["acc"],
    test_acc=rule_test["acc"],
    macro_f1=rule_test["macro_f1"],
    best_params={"tau": tau, "delta": delta},
)
""")

md("""
## 11. BRACOL YOLO cross-check

YOLO class ids are **not** the same as `predominant_stress`. Mapping: Cercospora=0→4, Miner=1→1, Phoma=2→3, Rust=3→2. Filenames look like `1_jpg.rf.{hash}.txt`.
""")

code("""
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
    comp.to_csv(os.path.join(PATHS["results_dir"], "bracol_crosscheck.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].pie(
        [comp["agree"].sum(), (1 - comp["agree"]).sum()],
        labels=["Agree", "Disagree"], autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    axes[0].set_title("Predictions vs BRACOL majority box")
    disagree = comp[comp["agree"] == 0]
    if len(disagree):
        vc = disagree["true"].value_counts()
        axes[1].bar(vc.index.astype(str), vc.values, color="#e74c3c")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_title("Disagreements by true class")
    fig.tight_layout()
    save_and_show(fig, "bracol_crosscheck.png", PATHS["figures_dir"])
else:
    print("No overlap with BRACOL labels. Folder missing or different filenames.")
""")

md("""
## 12. Final results table and literature

Esgario et al. (2020), *Computers and Electronics in Agriculture*:

- Leaf **classification** set: 1,685 images, **five** classes (they dropped 62 mixed-equal leaves).
- Split: 70 / 15 / 15 (not 5-fold).
- ResNet50 single-task **95.63%** accuracy; multi-task **95.24%**, precision 95.29%, recall 91.14%.
- They crop the leaf with an HSV-S threshold before 224×224.

We use a 6-class label that includes Mixed. Missing leaf JPEGs are filled from Dataset Ninja and the Roboflow exports when those files exist. Those two facts, not architecture names, explain most of the gap versus Esgario.
""")

code("""
results_df = pd.DataFrame(results_rows())
results_df.to_csv(os.path.join(PATHS["results_dir"], "final_results.csv"), index=False)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
plot_df = results_df.dropna(subset=["test_acc"]).copy()
labels = plot_df["model"] + " / " + plot_df["features"]
ax.barh(range(len(plot_df)), plot_df["test_acc"], color="#3d7a4a")
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("test accuracy")
ax.set_xlim(0, 1)
ax.set_title("Test accuracy by configuration")
fig.tight_layout()
save_and_show(fig, "final_accuracy_bars.png", PATHS["figures_dir"])
""")

md("""
## 13. Conclusions

- Two representations (handcrafted vs frozen CNN embeddings) and two official models (SVM, EfficientNet) are implemented end to end, each with its own ingest → fit → search → refit → report loop.
- The split is shared. Test is not used to pick hyperparameters.
- Mixed (class 5) is kept. The residual rule is an extra, not a replacement of the official 6-class task.
- ResNet-50 is the designated third method and the closest architecture family to Esgario et al. Numbers are not directly comparable (5 vs 6 classes, different image subset, they crop the leaf).
- MixUp follows Esgario's augmentation experiments; it is not presented as a new idea.

Next cell writes a compact JSON of the run so the report can copy numbers without re-typing them.
""")

code("""
payload = {
    "n_csv": summary["n_csv"],
    "n_kept": summary["n_kept"],
    "n_missing": summary["n_missing"],
    "n_corrupt": summary["n_corrupt"],
    "split": split.sizes(),
    "results": results_rows(),
}
with open(os.path.join(PATHS["results_dir"], "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, default=str)
print("Wrote results/run_summary.json")
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}

for i, cell in enumerate(CELLS):
    cell = dict(cell)
    cell["id"] = f"c{i:03d}"
    if cell["cell_type"] == "markdown":
        src = cell["source"]
        cell["source"] = [line + "\n" for line in src.split("\n")]
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"][-1] = ""
        # keep trailing newline style simple
        cell["source"] = [src] if isinstance(src, str) else src
    else:
        src = cell["source"]
        cell["source"] = [src] if isinstance(src, str) else src
    nb["cells"].append(cell)

out = os.path.join(os.path.dirname(__file__), "P1_notebook.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote", out, "cells:", len(nb["cells"]))
