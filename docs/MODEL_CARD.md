# Model Card - COVID-CT Classification (SVM + EfficientNet-B0)

## Model Details

- **Developed by:** Alex Zuccolin (EPICODE - Computer Vision course, 2026).
- **Model type:** Two parallel binary classifiers for chest CT slices.
  - **Classical baseline:** RBF-SVM on PCA-200 of HOG + LBP + GLCM features.
  - **Deep model:** EfficientNet-B0 (timm, ImageNet-pretrained) with custom
    head (Dropout 0.5 -> Linear 1280->2), two-phase fine-tuning.
- **Frameworks:** scikit-learn 1.8, PyTorch 2.x, timm 1.0, albumentations 2.x.
- **Input:** single axial chest-CT slice as a PNG (any size, any aspect ratio).
- **Output:** probability of the *COVID* class in [0, 1] and a hard label.
- **Licence:** MIT (code); training data CC BY 4.0 (Soares et al., 2020).
- **Version:** 0.1.0.

## Intended Use

- **Primary intended use:** academic teaching and benchmarking on the
  SARS-CoV-2 CT-Scan dataset.
- **Out-of-scope:** any clinical, triage, or diagnostic application.
  This system has not been evaluated under the regulatory frameworks
  required for medical software (CE marking, FDA 510(k)).

## Factors

- **Relevant factors:** scanner manufacturer, slice thickness, patient
  age/sex/comorbidity, country of acquisition, image acquisition window,
  presence of imaging artefacts.
- **Evaluation factors:** none of the above is annotated in the dataset, so
  subgroup evaluation could not be performed.

## Metrics

Test set (n = 373, balanced, stratified, seed 42, no patient grouping).

| Metric | SVM | Random Forest | EfficientNet-B0 |
|--------|----:|--------------:|----------------:|
| Accuracy | 0.962 | 0.887 | 0.933 |
| Recall (COVID) | 0.968 | 0.878 | 0.920 |
| Specificity | 0.957 | 0.897 | 0.946 |
| F1 | 0.963 | 0.887 | 0.933 |
| AUC-ROC | 0.996 | 0.952 | 0.981 |
| Brier | 0.024 | 0.103 | n/a |
| ECE (10 bins) | 0.027 | 0.128 | n/a |

The EfficientNet-B0 figures come from the full 40-epoch fine-tuning run
(10 head-only + 30 full epochs) on a local RTX 3080, evaluated at the default
0.5 threshold. Brier/ECE were not extracted for the deep model; temperature
scaling found T = 1.002, indicating the model was already well-calibrated.
The SVM baseline remains the strongest model on this dataset (AUC 0.996 vs
0.981).

The decision threshold for deployment-style operation is tuned for COVID
recall >= 0.95 on the calibrated validation set (chosen threshold 0.358).
On the held-out test set that threshold yields recall = 0.931 - the >= 0.95
target is met on validation but not fully on test, a sign the threshold does
not perfectly generalise. See `docs/technical_analysis.pdf` § Methodology.

## Training Data

- **Dataset:** SARS-CoV-2 CT-Scan (Soares et al., 2020), 2 481 PNG images,
  Brazilian hospitals, two classes (COVID / non-COVID).
- **Splits:** 70 / 15 / 15 stratified by label, `random_state=42`.
- **No patient identifiers** are provided - see *Caveats* below.

## Evaluation Data

Held-out 15% test split of the same dataset.

## Ethical Considerations

- Predictions must not influence clinical decisions.
- Saliency maps (Grad-CAM) are provided for interpretability but are not a
  substitute for clinical-grade explainability.
- The model has not been audited for fairness across demographic subgroups
  because the relevant metadata is missing from the dataset.

## Caveats and Recommendations

- **Patient-level leakage:** absence of patient IDs means slices from the same
  patient may span splits. Reported metrics may overstate generalisation.
- **Distribution shift:** evaluation on Brazilian-hospital data only.
  Performance on other scanners/populations is unknown.
- **Recommendation:** any deployment scenario should re-validate with a
  patient-stratified split and prospective evaluation on a representative
  cohort.
