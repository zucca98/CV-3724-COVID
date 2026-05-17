# Datasheet - SARS-CoV-2 CT-Scan Dataset (as used in this project)

This datasheet documents how the **SARS-CoV-2 CT-Scan** dataset is consumed by
this project; it does not replace the upstream documentation by Soares et al.

## Motivation

- **Why was the dataset created?** To enable rapid research on CT-based
  detection of SARS-CoV-2 pneumonia at the height of the pandemic.
- **Who created it?** Eduardo Soares, Plamen Angelov and colleagues
  (Lancaster University and Brazilian hospitals); released via Kaggle.
- **Who funded it?** As reported in Soares et al. (2020) - see references.
- **Use in this project:** binary classification benchmark for an academic CV
  course assignment.

## Composition

- **What does each instance represent?** A single axial CT slice in PNG
  format, gray-scale or RGB, varying resolutions (typical 200-500 px).
- **How many instances?** 2 481 PNGs, two folders: `COVID/` (1 252) and
  `non-COVID/` (1 229).
- **Class balance:** approximately balanced (50.5% / 49.5%).
- **Patient-level metadata:** **absent.** No patient ID, scanner model, slice
  thickness, age, sex, or comorbidity is included.
- **Splits:** the upstream release is unsplit; this project applies an
  internal stratified 70 / 15 / 15 split with `random_state=42`.

## Collection Process

- **Source:** two hospitals in São Paulo, Brazil (per upstream paper).
- **Acquisition:** retrospective collection of clinical CT scans.
- **Sampling:** convenience sampling - not necessarily representative of the
  general population.
- **Time-frame:** 2020 (early pandemic).

## Preprocessing / Cleaning / Labelling

- The upstream maintainers labelled images at the slice level via clinical
  examination.
- This project applies its own preprocessing pipeline (see
  `src/preprocessing.py`): resize-with-padding to 224 x 224, optional lung
  masking (U-Net or Otsu fallback), 3-channel CLAHE composite, augmentation.
- No further re-labelling was performed.

## Uses

- **Suitable uses:** education, benchmarking, methodological experiments.
- **Unsuitable uses:** clinical decision support, triage, regulatory
  submissions. The dataset has known limitations (no patient IDs, single
  geography) that would invalidate clinical claims.

## Distribution

- Available on Kaggle:
  <https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>
- Licence: **CC BY 4.0**, with required attribution to Soares et al. (2020).

## Maintenance

- This project pins the dataset by reproducing the integer hash of each
  filename in the split CSVs (`data/processed/{train,val,test}.csv`).
- No upstream re-labelling is monitored. If the dataset is updated, splits
  must be regenerated with `scripts/prepare_data.py`.

## Known Limitations and Risks

1. **No patient identifiers** => slices from the same patient can fall into
   different splits, inflating reported metrics.
2. **Geographic homogeneity** (Brazil) => poor generalisation guarantees.
3. **Demographic metadata missing** => cannot audit for fairness.
4. **Acquisition heterogeneity unknown** => scanner-model / slice-thickness
   shortcuts cannot be ruled out.
5. **Annotation quality:** upstream labels are slice-level binary; no
   region-level masks of pathology were provided.

## Reference

> Soares, E., Angelov, P., Biaso, S., Froes, M. H., & Abe, D. K. (2020).
> *SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for
> SARS-CoV-2 identification.* medRxiv.
> <https://doi.org/10.1101/2020.04.24.20078584>
