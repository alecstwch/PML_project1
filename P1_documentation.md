# Technical Report — Project 1

**Title:** Supervised classification of coffee leaf biotic stress using SVM and EfficientNet

**Author:** Soare Alecsandru Florin

**Dataset:** BRACOL leaf images (Brazilian Arabica coffee)

---

## Summary

This report describes a six-class classifier for the predominant biotic stress on Arabica coffee leaves. Two official models are used: a support vector machine with an RBF kernel, and a fine-tuned EfficientNet-B0. They are tested on two different feature families: handcrafted colour/texture/edge descriptors, and frozen ImageNet embeddings. A shared 70 / 15 / 15 split is used so that hyperparameters are never chosen on the test set.

ResNet-50 is run as a third method because it is the backbone family used by Esgario et al. (2020). XGBoost is kept as a cheap baseline on the same cached vectors. MixUp, a probability ensemble, and a residual rule for the rare Mixed class are extra checks, not replacements for the official 6-class task.

Numeric scores are produced by `P1_notebook.ipynb` and written to `results/final_results.csv`. They must be read from that file after a full run. They are not copied from an older notebook.

---

## Contents

1. Introduction
2. Dataset and split
3. Feature representations
4. Procedure — Model 1 (SVM)
5. Procedure — Model 2 (EfficientNet)
6. Extra methods
7. Findings (how to read the artefacts)
8. Comparison with literature
9. Conclusions
10. References
- Appendix A — Hyperparameter grids
- Appendix B — Submission layout

---

## 1. Introduction

Coffee production is reduced by leaf miner, rust, brown leaf spot (phoma) and cercospora leaf spot. The task here is to label a photograph of one leaf by its **predominant** stress. The six labels are Healthy, Miner, Rust, Phoma, Cercospora and Mixed.

The two official models were chosen for different reasons. SVM with an RBF kernel is a standard strong learner on a few thousand samples with high-dimensional embeddings, and the C / gamma grid is easy to plot. EfficientNet-B0 is a compact convolutional network that can be fine-tuned on the raw pixels. Together they cover criterion (b) of the project (two supervised methods) and criterion (c) (two feature types that are not a small variant of the same idea).

The reader of this report is assumed to know basic supervised learning (train / validation / test, accuracy, F1, a confusion matrix) but not the BRACOL papers in detail. Those papers are cited in Section 8 with the numbers taken from the published tables, not from memory.

---

## 2. Dataset and split

### 2.1 Source

The images come from the BRACOL leaf set collected in Espírito Santo, Brazil (Esgario et al., 2020). Leaves were photographed from the abaxial side on a white background with five phones: ASUS Zenfone 2, Xiaomi Redmi 5A, Xiaomi S2, Galaxy S8 and iPhone 6S.

The CSV has 1,747 rows. A row is kept only if `{id}.jpg` exists and the JPEG fully decodes. A previous copy of the files had 1,402 images on disk and 1,401 after dropping one truncated file. The notebook prints the actual counts. About one fifth of the CSV rows have no file in this download; that is a **20% gap**, not an 80% cut.

Each row has:

- `predominant_stress` in {0,…,5} — the classification target
- binary flags `miner`, `rust`, `phoma`, `cercospora`
- `severity` in {0,…,4}

Severity bins follow the paper in spirit (healthy under 0.1%, then very low / low / high / very high). The paper writes some cuts as 5.1% and 10.1%. We follow the integer codes in the CSV.

Class 5 (Mixed) is kept. Esgario et al. dropped 62 leaves where two stresses had similar severity and trained a **five-class** Leaf task on 1,685 images. That difference is stated again in Section 8.

### 2.2 Split (criterion h)

One stratified split is used for every model:

- train 70%
- validation 15%
- test 15%

Indices are saved to `features/split_ids.npz`. The test set is scored once, after the hyperparameters are locked. The validation set is used for grids and for early stopping.

Test is split off first (15% of the full table), then validation is 15/85 of what remains. That order avoids a rare class being left with a single sample in a 30% pool, which sklearn cannot stratify.

---

## 3. Feature representations

### 3.1 Representation 1 — handcrafted

White paper is removed first. In HSV, background pixels have low saturation and high value. The mask keeps the rest.

Three descriptors are concatenated and L2-normalised:

1. HSV histogram, 8×8×8 bins (512 numbers), computed with the mask.
2. Uniform local binary pattern, radius 3, 24 points (26 bins).
3. Histogram of oriented gradients on a 224×224 image, 8×8 cells, 2×2 blocks, 9 orientations.

This is colour plus texture plus edges. It is not a bag-of-words variant. The code that builds it is `HandcraftedFeatures` in `p1_core.py`. Vectors are cached in `features/handcrafted.npz`.

### 3.2 Representation 2 — frozen deep features

EfficientNet-B0 pretrained on ImageNet, classifier removed, global average pooling: 1,280 numbers per image. Images are resized to 224×224 and passed through the EfficientNet preprocess function. No augmentation is used at extract time. The cache is `features/frozen_deep.npz`.

An optional third step, `PCAFeatures`, reduces those 1,280 numbers to 256. PCA is fit on the **train** split only.

These two families are different: one is designed histograms, the other is a learned embedding. That is what criterion (c) asks for.

---

## 4. Procedure — Model 1 (SVM)

The loop is the same five steps for every learner: ingest, fit, search, refit, report.

### 4.1 Model

A support vector machine finds a maximum-margin separator in a kernel space. For two vectors \(x\) and \(x'\), the RBF kernel is

\[
K(x, x') = \exp(-\gamma \|x - x'\|^2).
\]

\(C\) trades off margin width against training errors. \(\gamma\) sets how local the kernel is. sklearn `SVC` uses one-vs-one for more than two classes. Class weights are balanced because Mixed and Cercospora are rare.

Features are scaled with `StandardScaler` fit on train only. Distances used by SVM are sensitive to scale, so this step is required.

### 4.2 Search (criterion d)

The grid is scored on **validation accuracy**, not on test:

- \(C \in \{0.1, 1, 10, 100, 1000\}\)
- \(\gamma \in \{\texttt{scale}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}\) for RBF
- kernel: RBF and linear (linear is an ablation, mainly on deep features)

The RBF slice is drawn as a heatmap (`figures/svm_heatmap_*.png`). The best pair is refit on train with `probability=True` so later ensembles can average scores. Test is then used once. Confusion matrices are written under `figures/cm_*.png` (criterion e).

The loop is run on handcrafted vectors and again on frozen embeddings. That is one model, two representations.

---

## 5. Procedure — Model 2 (EfficientNet)

### 5.1 Model

EfficientNet scales depth, width and resolution together (Tan and Le, 2019):

\[
d = \alpha^{\phi},\quad w = \beta^{\phi},\quad r = \gamma^{\phi},
\quad \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2.
\]

For B0 the constants are \(\alpha = 1.2\), \(\beta = 1.1\), \(\gamma = 1.15\). We use B0 with ImageNet weights and a new head: global average pooling, dropout, softmax over six classes. It is a compact CNN, not described here as “state of the art”.

### 5.2 Training

Ingest loads 224×224 RGB arrays. Training uses horizontal and vertical flips, a small brightness change, and MixUp. MixUp forms a new example

\[
\tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad
\tilde{y} = \lambda y_i + (1-\lambda) y_j,
\]

with \(\lambda\) drawn from a Beta(0.2, 0.2) law. Esgario et al. already tried MixUp on ResNet-50 for this dataset. We follow that idea; we do not claim it is new.

Phase 1 freezes the backbone and trains the head. Phase 2 opens the last layers at a lower learning rate. Early stopping watches validation loss. Class weights are balanced.

A small grid over learning rate and how many layers to unfreeze is scored on validation accuracy. Training curves are saved (`figures/curves_*.png`). Test is scored once.

---

## 6. Extra methods (criterion i)

**ResNet-50.** Same split, same MixUp and early stopping as EfficientNet. This is the designated third method. It is also the architecture family in Esgario et al., so differences in accuracy are about protocol and data, not about swapping ResNet for an unrelated model.

**XGBoost.** Same cached matrices as the SVM. A short grid over number of trees, depth and learning rate. It is the method used in an earlier draft of this project, kept as a baseline row.

**Ensemble.** Average the test probabilities of the best SVM-on-deep model and the fine-tuned EfficientNet. A three-way average with ResNet is optional.

**Residual rule for Mixed.** The official task stays six classes. An extra variant trains a multi-label logistic probe on the four binary disease columns (train split only). At inference, if two disease heads exceed a threshold \(\tau\), or the gap between the top two 6-class probabilities is below \(\delta\), the label is Mixed. \(\tau\) and \(\delta\) are chosen on validation only.

**BRACOL YOLO cross-check.** Detection files use a different class index: Cercospora=0, Miner=1, Phoma=2, Rust=3. Those ids are mapped to `predominant_stress` before any comparison. Filenames look like `1_jpg.rf.{hash}.txt`. Cases with several lesion types in YOLO and a Mixed or wrong 6-class label are discussed as ambiguous leaves, not only as model errors.

---

## 7. Findings

Run `P1_notebook.ipynb` (or `P1_notebook.py`) to fill `results/final_results.csv` and the figures. The table has one row per configuration:

| Model | Features | Val acc | Test acc | Macro F1 |
|---|---|---|---|---|
| SVM-RBF | handcrafted | | | |
| SVM-RBF | frozen EfficientNet | | | |
| SVM-linear | frozen EfficientNet | | | |
| XGBoost | handcrafted | | | |
| XGBoost | frozen EfficientNet | | | |
| EfficientNet-B0 | raw images | | | |
| ResNet-50 | raw images | | | |
| Ensemble | SVM-deep + EfficientNet | | | |
| SVM-deep + residual rule | frozen EfficientNet | | | |

What to look at, in order:

1. **Figure `class_distribution.png`.** Mixed is small. Macro F1 will be harsh on that class.
2. **SVM heatmaps.** If deep features are already easy to separate, the linear kernel should be close to RBF and the useful \(\gamma\) values will sit near `scale`.
3. **Confusion matrices.** Phoma vs Cercospora (both brown spots) and Mixed vs everything are the likely collisions.
4. **CNN curves.** If validation loss turns up while training loss falls, stop earlier or freeze more layers.
5. **Residual-rule row vs plain SVM.** If macro F1 rises and accuracy stays similar, the rule is helping Mixed without wrecking the other classes.

Do not treat a previous XGBoost run (about 82% accuracy on a similar split) as a result of this notebook.

---

## 8. Comparison with literature (criterion f)

Esgario, J.G.M., Krohling, R.A. and Ventura, J.A. (2020), “Deep learning for classification and severity estimation of coffee leaf biotic stress”, *Computers and Electronics in Agriculture*, 169, 105162.

From their Leaf **classification** experiments (not the symptom crops):

| Setting | Accuracy | Precision | Recall |
|---|---|---|---|
| ResNet50, single-task | 95.63% | 94.12% | 92.70% |
| ResNet50, multi-task | 95.24% | 95.29% | 91.14% |

They do not headline a 93% macro F1. They used **1,685** images and **five** classes. They collected 1,747 photos and dropped 62 mixed-equal leaves. Their split is 70 / 15 / 15, the same ratios as ours, not 5-fold cross-validation. Architectures in that paper are AlexNet, GoogLeNet, VGG16 and ResNet50. They crop the leaf with a threshold on the HSV saturation channel before resizing to 224×224. They also tried MixUp on ResNet50.

Manso, G.L., Knidel, H., Krohling, R.A. and Ventura, J.A. (2019) is the earlier segmentation work (miner vs rust, handcrafted features on lesions) that the 2020 paper extends.

A gap between our test accuracy and 95.63% is expected, and should be explained by:

1. fewer images (missing files in this download)
2. six classes including Mixed, versus five
3. no specialist leaf crop unless we add one
4. a different random split, even at the same 70 / 15 / 15 ratios

Our ResNet-50 run is the fairest row to put next to their ResNet50, with those caveats written beside the number.

---

## 9. Conclusions

The project is built as a shared data corridor and then a full grading loop per official model: SVM first, EfficientNet second. Handcrafted and frozen-deep vectors are different representations. The split is stored and reused. Test is not used to pick C, gamma, learning rate or MixUp thresholds.

Mixed is kept. The residual rule is an extra attempt to catch leaves that are not healthy and not clearly one of classes 1–4. MixUp and the ensemble are extra as well.

The honest comparison with Esgario et al. is a ResNet-50 trained on our 6-class subset, not a claim that EfficientNet-B0 should match 95.63% on a different task.

---

## 10. References

1. Esgario, J.G.M., Krohling, R.A. and Ventura, J.A. (2020) ‘Deep learning for classification and severity estimation of coffee leaf biotic stress’, *Computers and Electronics in Agriculture*, 169, 105162. https://doi.org/10.1016/j.compag.2019.105162
2. Manso, G.L., Knidel, H., Krohling, R.A. and Ventura, J.A. (2019) ‘A smartphone application to detection and classification of coffee leaf miner and coffee leaf rust’, *arXiv:1904.00742*.
3. Tan, M. and Le, Q. (2019) ‘EfficientNet: Rethinking model scaling for convolutional neural networks’, *Proceedings of ICML*.
4. Zhang, H., Cisse, M., Dauphin, Y.N. and Lopez-Paz, D. (2018) ‘mixup: Beyond empirical risk minimization’, *Proceedings of ICLR*.
5. The Institution of Engineering and Technology (2015) *A Guide to Technical Report Writing*.

---

## Appendix A — Hyperparameter grids

**SVM:** C in {0.1, 1, 10, 100, 1000}; gamma in {scale, 1e-4, 1e-3, 1e-2, 1e-1}; kernel in {rbf, linear}.

**XGBoost:** n_estimators in {200, 500}; max_depth in {3, 5, 7}; learning_rate in {0.05, 0.1}; subsample 0.8; colsample_bytree 0.8.

**EfficientNet / ResNet (default small grid):** learning rate in {1e-4, 1e-5}; dropout 0.3; last 20 layers unfrozen in phase 2. Widen the list in the notebook if GPU time allows.

**Residual rule:** \(\tau\) in {0.30, 0.40, 0.50, 0.60}; \(\delta\) in {0.05, 0.10, 0.15, 0.20}, chosen on validation macro F1.

---

## Appendix B — Submission layout

```
P1_Soare_Alecsandru_{group}/
    P1_notebook.py
    p1_core.py
P1_Soare_Alecsandru_{group}_doc/
    P1_documentation.pdf
    figures used in the PDF
```

Do not zip the leaf images or the CSV. Unit tests in `tests/` are for local checks and are not required in the zip.
