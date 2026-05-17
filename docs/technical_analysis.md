---
title: "COVID-19 Detection from CT Scans: A Deep Learning and Classical Baseline Study"
author: "EPICODE - Computer Vision Course"
date: "2026"
geometry: "margin=2.5cm"
fontsize: 11pt
toc: true
numbersections: true
---

# COVID-19 Detection from CT Scans

\newpage

## Problem Statement

### Motivation

The COVID-19 pandemic created unprecedented demand for rapid, scalable diagnostic tools.
Chest computed tomography (CT) is sensitive to the ground-glass opacities and
consolidation patterns characteristic of SARS-CoV-2 pneumonia, and radiological
triage has been proposed as a complement to RT-PCR testing when laboratory capacity
is constrained.

### Task Definition

Given a single axial CT-scan slice as a PNG image, the system must output a binary label:
**COVID** (SARS-CoV-2 pneumonia present) or **Non-COVID** (other pathologies or healthy).
The primary performance criterion is **sensitivity (recall) >= 0.95** for the positive
class, prioritising the reduction of false negatives over false positives, as missing a
COVID case carries higher clinical risk than an unnecessary quarantine.

### Dataset

The SARS-CoV-2 CT-Scan dataset [1] contains 2 481 PNG images collected from Brazilian
hospitals and released under CC BY 4.0:

| Split | COVID | Non-COVID | Total |
|-------|------:|----------:|------:|
| Train (70%) | 876 | 860 | 1 736 |
| Val   (15%) | 188 | 184 |   372 |
| Test  (15%) | 188 | 185 |   373 |

Splits are stratified by label with `random_state=42`.

**Known limitation:** the dataset provides no patient identifiers.
Slices from the same patient may appear in different splits, introducing a
potential data-leakage bias that could overstate generalisation performance.
This limitation is declared explicitly throughout the analysis.

\newpage

## Methodology

### Preprocessing

Raw images are processed through a fixed pipeline before any model ingests them:

1. **Load** - OpenCV reads PNG as uint8; BGRA images are converted to RGB.
2. **Resize with padding** - aspect-ratio-preserving resize to 224 x 224 with
   black padding (no distortion).
3. **Lung masking** - Otsu thresholding + morphological closing/opening + two-
   largest-component selection isolates the lung field and zeroes out background
   tissue. The deep learning model (lungmask U-Net [2]) is used when available.
4. **Three-channel construction** - the single grayscale slice is expanded to
   three channels: (original, CLAHE clip=2.0, CLAHE clip=4.0). This increases
   visual contrast for the CNN without adding separate images.
5. **Normalisation** - per-channel mean/standard-deviation normalisation with
   statistics computed on the **training split only** by
   `scripts/compute_norm_stats.py` and stored in `configs/norm_stats.json`
   (mean = [0.475, 0.446, 0.433], std = [0.366, 0.348, 0.336]). If the file is
   absent the pipeline falls back to ImageNet statistics; test-set statistics
   are never used.

**Training augmentations** (no vertical flip; anatomically inappropriate for CT):
horizontal flip (p=0.5), rotation ±10° (p=0.5), brightness/contrast jitter (p=0.5),
affine scale 0.9-1.1 (p=0.5), Gaussian noise (p=0.5).

### Classical Feature Extraction

Handcrafted features are extracted from the preprocessed grayscale image and
concatenated into a single 26 366-dimensional vector:

| Descriptor | Dimensions | Parameters |
|------------|-----------:|------------|
| Uniform LBP | 26 | P=24, R=3 |
| GLCM       | 96 | 4 distances x 4 angles x 6 statistics |
| HOG        | 26 244 | 9 bins, 8x8 cells, 2x2 blocks |

### Classical Baseline Models

Two pipelines are trained with 5-fold stratified GridSearchCV:

**SVM (RBF kernel):** StandardScaler -> PCA(200) -> SVC.
PCA reduces the 26 366-d vector to 200 principal components for computational
tractability; 200 components retain > 95% of variance.
Grid: C in {0.1, 1, 10, 100}, γ in {scale, 0.001, 0.01}.

**Random Forest:** RandomForestClassifier.
Grid: n\_estimators in {100, 300, 500}, max\_depth in {None, 10, 20}.

### Deep Learning Model

**Architecture:** EfficientNet-B0 (timm, pretrained on ImageNet) with a custom
classification head: GlobalAveragePool -> Dropout(0.5) -> Linear(1280, 2).

**Two-phase fine-tuning:**

- *Phase 1 (head only, 10 epochs):* backbone frozen, Adam lr=1 x 10⁻³,
  ReduceLROnPlateau(patience=3). Best checkpoint saved on val\_loss.
- *Phase 2 (partial unfreeze, up to 30 epochs):* last 2 block stages +
  conv\_head + bn2 unfrozen, AdamW lr=1 x 10⁻⁵, weight\_decay=1 x 10⁻⁴,
  early stopping patience=7.

Mixed-precision training (AMP) is enabled on GPU (auto-disabled on CPU).

### Post-Processing

**Temperature scaling:** scalar T > 0 is optimised with LBFGS on the validation-
set negative log-likelihood to correct overconfidence before threshold selection.

**Threshold tuning:** the decision threshold is swept from high to low probability;
the lowest value that achieves recall >= 0.95 on the calibrated validation probabilities
is selected, prioritising sensitivity.

**TTA (Test-Time Augmentation):** at inference time, five augmented views of each
test image are generated; their softmax probabilities are averaged.

**Ensemble:** weighted average of probabilities from multiple independently-trained
checkpoints (weights set proportionally to validation AUC).

**Grad-CAM:** class activation maps highlight the CT regions most influential for
the positive-class prediction, enabling visual inspection of model behaviour.

\newpage

## Experimental Results

### Classical Baselines

Results on the held-out test set (n = 373):

| Model | Accuracy | Recall | F1 | AUC-ROC |
|-------|:--------:|:------:|:--:|:-------:|
| Random Forest | 0.887 | 0.878 | 0.887 | 0.952 |
| SVM (PCA-200, C=100) | **0.962** | **0.968** | **0.963** | **0.996** |

The SVM achieves near-perfect AUC, substantially outperforming the Random Forest.
The gap is consistent with the literature: hand-crafted texture features (HOG + LBP +
GLCM) are highly discriminative for this task, and the RBF kernel exploits
non-linear structure in the PCA-projected space efficiently.

![ROC curve comparison - classical models](figures/roc_comparison.png){ width=70% }

### Confusion Matrices

![Confusion matrix - SVM](figures/confusion_matrix_svm.png){ width=45% }
![Confusion matrix - Random Forest](figures/confusion_matrix_random_forest.png){ width=45% }

### Deep Learning (Full Training)

EfficientNet-B0 was fine-tuned for 40 epochs (10 head-only + 30 full
fine-tuning) on a local RTX 3080 in 3 h 59 min. The two-phase schedule ran to
completion without early stopping - validation loss kept improving, best
checkpoint at phase-2 epoch 27 (val_loss = 0.165). Test-set metrics (373
images):

| Model | Accuracy | Recall | F1 | AUC-ROC |
|-------|:--------:|:------:|:--:|:-------:|
| EfficientNet-B0 (fine-tuned, thr=0.50)         | 0.933 | 0.920 | 0.933 | 0.981 |
| + Temperature scaling (T=1.002)                | 0.933 | 0.920 | 0.933 | 0.981 |
| + Threshold tuned for sensitivity (thr=0.358)  | 0.912 | 0.931 | 0.914 | 0.981 |
| + TTA (5 deterministic views)                  | 0.944 | 0.942 | 0.944 | 0.990 |

Notes on the post-processing steps:

- **Lung masking** was enabled for the whole run, but the `lungmask` U-Net
  rejects 8-bit PNG input (it expects Hounsfield-unit volumes), so the pipeline
  silently fell back to the Otsu + morphology mask for every slice. The
  reported "lung mask" is therefore the classical fallback, not the U-Net.
- **Temperature scaling** found T = 1.002: the model was already well
  calibrated, so scaling leaves the decision metrics essentially unchanged.
- **Threshold tuning** lowered the decision threshold to 0.358 to reach the
  >= 0.95 sensitivity target *on the validation set* (val recall = 0.952). On
  the held-out test set the same threshold yields recall = 0.931 - below
  target: the threshold does not fully generalise, a consequence of the small
  validation split and possible patient-level leakage.
- **Test-Time Augmentation (TTA)** averages softmax probabilities over five
  deterministic views (original, horizontal flip, rotation +/-7 degrees, zoom
  +8%). It improves every metric - recall rises from 0.920 to 0.942 and AUC
  from 0.981 to 0.990 - at the cost of 5x inference time.
- **Ensembling** is left as future work: it would require training several
  independent checkpoints (each ~4 h on the available GPU) and was out of
  scope for this iteration.

Even with TTA the deep model (AUC 0.990) approaches but does not surpass the
SVM baseline (AUC 0.996): on this small, well-separated dataset the
handcrafted-feature SVM remains the strongest model.

### Calibration

Calibration of the SVM is already excellent (ECE = 0.027, Brier = 0.024 - see
the comparison table). For the deep model, temperature scaling on the
validation logits is implemented in `src/postprocess.py:temperature_scaling`
(LBFGS over scalar T) and is invoked by
`scripts/calibrate_and_evaluate.py` once a trained checkpoint is available.
The reliability diagram before/after scaling is saved to
`models/postprocess/reliability_diagram.png`. With T = 1.002 the two curves
nearly overlap, confirming the model needed no meaningful calibration.

\newpage

## Failure Analysis

### Data Leakage

The dataset contains no patient identifiers. If multiple slices from the same
patient appear in both training and test splits, the model may memorise patient-
specific features (e.g., rib shadows, pleural thickening) rather than learning
pathology-relevant patterns. This inflates the measured AUC and may not reflect
true out-of-distribution generalisation.

**Mitigation:** results are clearly labelled as potentially optimistic;
a patient-stratified split is recommended for production evaluation.

### Distribution Shift

Images were collected from a single country (Brazil) across two hospitals.
Performance may degrade on data acquired with different scanner models, slice
thicknesses, or patient demographics.

### Class Imbalance

The dataset is approximately balanced (1252 COVID vs. 1229 non-COVID), so
class imbalance is not a primary concern. However, the threshold-tuning step
intentionally sacrifices specificity to meet the recall >= 0.95 target.

### Qualitative Examples

The figure below shows TP / TN / FP / FN slices selected from the SVM
test-set predictions. The two false cases are the most informative: in
the FP slice, ground-glass-like artefacts in a non-COVID image fool the
classifier; in the FN slice, the lesion area is small or peripheral and
the global texture descriptors fail to capture it.

![SVM qualitative examples (TP/TN/FP/FN)](figures/qualitative_examples.png){ width=85% }

### Smoke-Test vs. Full Training

Early development used CPU smoke tests (50 images, 1 epoch per phase) purely as
integration checks; those numbers do not reflect trained performance and are
not reported. The deep-model metrics in the table above come from the full
40-epoch run on the complete dataset (RTX 3080, 3 h 59 min). The SVM and Random
Forest numbers derive from their respective pipelines, trained to convergence
on CPU with the full dataset.

\newpage

## Ethical Considerations

### Not for Clinical Use

This project is an academic exercise and has not undergone the validation required
for medical device certification (e.g., CE marking, FDA 510(k)).
It must not be used to support clinical diagnosis or treatment decisions.

### Model Transparency

Grad-CAM (`scripts/generate_gradcam.py`) highlights the regions that drive each
prediction. The figure below shows one example per confusion-matrix category
(TP / TN / FP / FN) from the test set: for true and false positives the
activation concentrates on the lung parenchyma, supporting the plausibility of
the learned features. Saliency maps nonetheless remain insufficient for
clinical-grade explainability.

![Grad-CAM on EfficientNet-B0 test predictions](figures/gradcam_examples.png)

### Bias and Fairness

The dataset demographics (age, sex, comorbidity) are not publicly documented,
so performance disparities across patient subgroups cannot be assessed. Any
deployment would require prospective evaluation on a representative population.

### Data Provenance

The SARS-CoV-2 CT-Scan dataset was collected under ethical oversight in Brazil [1]
and is released under the Creative Commons Attribution 4.0 licence.
No personally identifiable information is included in the PNG files.

\newpage

## References

[1] Soares, E., Angelov, P., Biaso, S., Froes, M. H., & Abe, D. K. (2020).
    *SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for
    SARS-CoV-2 identification.* medRxiv.
    <https://doi.org/10.1101/2020.04.24.20078584>

[2] Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., & Langs, G.
    (2020). *Automatic lung segmentation in routine imaging is primarily a data
    diversity problem, not a methodology problem.* European Radiology Experimental,
    4(1), 50. <https://doi.org/10.1186/s41747-020-00173-2>

[3] Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for
    convolutional neural networks.* ICML 2019.
