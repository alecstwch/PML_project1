# Project 1 — Supervised Classification Plan (v3)

## Coffee Leaf Disease Classification using SVM (RBF) and EfficientNet

---

## 0. Registration Status (FINAL)

| Slot | Method | Role |
|---|---|---|
| Supervised Method 1 | **SVM with RBF kernel** (on both feature representations) | Official — criterion b) |
| Supervised Method 2 | **EfficientNet-B0** (fine-tuned end-to-end) | Official — criterion b) |
| Third method | **ResNet-50** (fine-tuned) | Criterion i) and literature f) |
| Extra baseline | **XGBoost** on the same cached features | Criterion i) bonus |
| Dataset | **BRACOL** leaf subset | Shared with Project 2 |

SVM with RBF is a strong learner on pretrained embeddings with about 1,400 samples. Its C / gamma grid is easy to plot (criterion d). XGBoost stays as a cheap comparison row. ResNet-50 matches the backbone family in Esgario et al. (2020).

---

## 1. Project Objective

Classify coffee leaf images by predominant biotic stress using **SVM (RBF)** and **EfficientNet-B0**. Keep all six classes (`predominant_stress` 0–5).

Secondary goals:

- Compare handcrafted vs frozen deep features with the same classifier (SVM).
- Compare SVM vs XGBoost vs end-to-end CNNs on a shared split.
- Compare our ResNet-50 run with Esgario et al. (2020), stating protocol differences honestly.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | BRACOL coffee leaf dataset (leaf subset) |
| CSV rows | 1,747 |
| Files on disk (previous count) | 1,402 JPEGs; 1,401 after dropping a bad file. Re-count at runtime. |
| Label | `predominant_stress` (0–5) |
| Extra columns | `miner`, `rust`, `phoma`, `cercospora` (binary), `severity` (0–4) |

Esgario et al. collected 1,747 photos but trained the Leaf task on **1,685** images and **five** classes. They dropped 62 leaves where two stresses had similar severity. We keep Mixed as class 5. That is a protocol difference, not the same task.

### Class map

| Code | Name | Notes |
|------|------|-------|
| 0 | Healthy | Severity under 0.1% |
| 1 | Miner | *Leucoptera coffeella* |
| 2 | Rust | *Hemileia vastatrix* |
| 3 | Phoma | Brown leaf spot (*Phoma* spp.) |
| 4 | Cercospora | *Cercospora coffeicola* |
| 5 | Mixed | More than one stress, no clear predominant |

### Severity map (CSV codes 0–4)

Paper cut-points use 5.1% and 10.1%. We follow the CSV codes.

| Code | Level | Affected area (paper) |
|------|-------|------------------------|
| 0 | Healthy | under 0.1% |
| 1 | Very low | 0.1%–5% |
| 2 | Low | about 5%–10% |
| 3 | High | about 10%–15% |
| 4 | Very high | over 15% |

---

## 3. Code layout

Two directions:

1. **Dataset corridor** (once): load → EDA → split → FE1 → FE2 → FE3.
2. **Learner loop** (per model): `ingest` → `fit` → `search` → `refit` → `report`.

| Class | Role |
|---|---|
| `LeafDataset` | Load CSV, drop missing files, EDA, duplicate check |
| `TrainValTestSplit` | Stratified 70/15/15, save `features/split_ids.npz` |
| `FeatureExtractor` | ABC: `extract(dataset, split) -> FeaturePack` |
| `HandcraftedFeatures` | Masked HSV histogram + LBP + HOG |
| `FrozenDeepFeatures` | EfficientNet-B0 GAP, 1280-d, cache `.npy` |
| `PCAFeatures` | PCA on frozen embeddings, fit on train only |
| `LearnerLoop` | ABC for the per-model loop |
| `SVMPipeline` | Model 1 |
| `EfficientNetPipeline` | Model 2 |
| `XGBoostPipeline` | Extra |
| `ResNetPipeline` | Third method |
| `save_and_show` | Write PNG under `figures/` and show it in the notebook |

Common code on the ABC is the exception. Each model has its own `ingest` / `fit` / `search`.

Every class and public method has a short plain-language docstring.

Implementation lives in `p1_core.py`. The notebook imports it and walks the grading flow. Tests import the same module.

---

## 4. Feature representations (criterion c)

### FE1 — `HandcraftedFeatures`

- Mask the white background (low saturation, high value in HSV) so colour stats describe the leaf.
- HSV 8×8×8 histogram (512-d), with the mask.
- Uniform LBP, radius 3, 24 points (26-d).
- HOG on 224×224, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, 9 orientations.
- Concatenate and L2-normalise. Cache `features/handcrafted.npy`.

### FE2 — `FrozenDeepFeatures`

- ImageNet EfficientNet-B0, no top, global average pooling → 1280-d.
- Resize 224×224, EfficientNet preprocess. No augmentation at extract time.
- Cache `features/frozen_deep.npy`.

### FE3 — `PCAFeatures` (optional, SVM only)

- PCA 256 or 128, **fit on train only**.

These are different representation families (handcrafted texture/colour vs learned CNN embeddings). Changing stop-word-style knobs on one family is not enough.

---

## 5. Models

### Model 1 — `SVMPipeline`

- Both FE1 and FE2. `StandardScaler` fit on train only. `class_weight='balanced'`.
- Default sklearn `SVC` is one-vs-one; state that in the report.
- Search on **validation accuracy** (fit on train, score on val):
  - `C`: 0.1, 1, 10, 100, 1000
  - `gamma`: scale, 1e-4, 1e-3, 1e-2, 1e-1 (RBF only)
  - `kernel`: rbf and linear (linear is an ablation on deep features)
- Plot a C × gamma heatmap of validation accuracy per representation.
- Refit best config on train. Touch test once.

### Model 2 — `EfficientNetPipeline`

- Raw 224×224 images. MixUp plus flips, small rotation, brightness.
- Phase 1: freeze backbone, train head, class weights.
- Phase 2: unfreeze last N layers, lower learning rate, early stopping.
- Search on validation accuracy (small grid by default so the notebook finishes): learning rate, dropout, unfrozen layer count.
- Training curves. Test once.

### ResNet-50 — `ResNetPipeline`

Same split, augmentation, MixUp, and early stopping as EfficientNet. Light grid only.

### XGBoost — `XGBoostPipeline`

Same cached matrices as SVM. Reduced grid: `n_estimators` {200, 500}, `max_depth` {3, 5, 7}, `learning_rate` {0.05, 0.1}.

### MixUp and ensemble

- MixUp on the CNN path only (Esgario already used MixUp on this dataset; we follow that, we do not claim it is new).
- Ensemble: average test probabilities of best SVM-on-deep and fine-tuned EfficientNet. Optional ResNet vote.

### Class 5 residual rule

Keep the official 6-class task. Extra variant:

1. Train a multi-label probe on `miner`, `rust`, `phoma`, `cercospora` (train split only).
2. If two or more disease heads are above τ, **or** the 6-class top-1 vs top-2 margin is below δ, predict Mixed.
3. Else keep the 6-class argmax.
4. Tune τ and δ on validation only. Report as an extra row.

---

## 6. Evaluation

- Shared stratified **70 / 15 / 15**, seed 42, indices saved and reused.
- Metrics: accuracy, macro and weighted precision / recall / F1, confusion matrix, classification report.
- Hyperparameters never chosen on test.
- SVM search uses the validation set for the heatmap (not nested CV on train+val), so the CNN and SVM paths share the same hold-out rule.

### Results table

| Model | Features | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|---|
| SVM-RBF | handcrafted | | | |
| SVM-RBF | deep (frozen EffNet) | | | |
| SVM-linear | deep (frozen EffNet) | | | |
| XGBoost | handcrafted | | | |
| XGBoost | deep (frozen EffNet) | | | |
| EfficientNet-B0 | raw images | | | |
| ResNet-50 | raw images | | | |
| Ensemble | SVM-deep + EffNet | | | |
| Residual-rule variant | (best 6-class + probe) | | | |

---

## 7. Literature (criterion f)

**Esgario et al. (2020)**, *Computers and Electronics in Agriculture*:

- Leaf classification set: **1,685** images, **5** classes (no Mixed).
- Split: **70 / 15 / 15** (not 5-fold).
- Best ResNet50: **95.63%** accuracy (single-task), **95.24%** (multi-task), precision 95.29%, recall 91.14%.
- They crop the leaf with an HSV-S threshold before 224×224. We may not; if we do not, say so.
- Architectures: AlexNet, GoogLeNet, VGG16, ResNet50. Not ResNet152 / Inception-v3.
- Do not quote a 93% macro F1 unless we derive it from their P/R and label it as derived.

**Manso et al. (2019)**: lesion segmentation; miner vs rust with handcrafted features.

Our gaps come from a smaller image subset (~20% of CSV rows missing), six classes vs five, and Mixed being rare.

---

## 8. Notebook order (grading loop)

1. Setup, `save_and_show`, imports from `p1_core`.
2. Dataset corridor: load, EDA, duplicates, split.
3. FE1, FE2, FE3.
4. **Model 1 full loop** (SVM on both representations): ingest, fit, search, refit, report.
5. **Model 2 full loop** (EfficientNet): same five steps.
6. Extras: XGBoost, ResNet-50, MixUp on/off, ensemble, residual rule.
7. BRACOL YOLO cross-check (class index mismatch documented).
8. Final table and literature.
9. Conclusions.

---

## 9. Tests and report

- `tests/` with pytest and synthetic images. Not in the submission zip.
- Report: plain language + IET technical report guide. Decimal headings. ≥ 2 pages of prose excluding figures. Verified citations only.

---

## 10. Packages

numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, tensorflow, opencv-python, scikit-image, Pillow, tqdm, imagehash, pytest (tests only)

---

## 11. Outputs

- `P1_notebook.ipynb` → `P1_notebook.py`
- `p1_core.py` (classes; same folder as the notebook script)
- `P1_documentation.md` → PDF in the `_doc` folder
- `figures/`, `features/`, `results/`, `models/` (local; do not zip data)

Submission zip: code `.py` files in `P1_Soare_Alecsandru_{group}/` (no code subfolders) and the PDF report in `P1_Soare_Alecsandru_{group}_doc/`. No images or CSV.
