# COVID-19 CT-Scan Classification

> **⚠ MEDICAL DISCLAIMER**
> This project is for **academic research purposes only**.
> It is **NOT validated for clinical use** and must **NOT** be used for medical
> diagnosis, triage, or treatment decisions.

[![CI](https://github.com/zucca98/progetto-computer-vision-s00003724/actions/workflows/ci.yml/badge.svg)](https://github.com/zucca98/progetto-computer-vision-s00003724/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Binary classification of COVID-19 vs. Non-COVID chest CT slices.
Academic project for the EPICODE Computer Vision course.

**Best result:** SVM (HOG + LBP + GLCM, PCA-200) -> AUC-ROC **0.9963** on the
held-out test set. The fine-tuned EfficientNet-B0 deep model reaches AUC-ROC
0.981 - the classical SVM baseline remains the strongest on this dataset.

---

## Demo

![Streamlit demo](docs/figures/streamlit_demo.png)

*Streamlit demo loaded with a sample COVID slice - the SVM predicts `COVID` with
`p = 0.990`. Run `streamlit run app/demo.py` to launch locally.*

![SVM confusion matrix](docs/figures/confusion_matrix_svm.png)

*Confusion matrix of the SVM baseline on 373 test images (run `scripts/run_full_evaluation.py` to generate).*

---

## Installation

Two supported installation paths: a **local Python venv** or a **Docker image**
(recommended for fully reproducible runs - it pins the same OS libs, Python 3.12
and CPU-only PyTorch wheels used for the reported results).

### Option A - Local virtual environment

```bash
# 1. Clone
git clone https://github.com/zucca98/progetto-computer-vision-s00003724.git
cd progetto-computer-vision-s00003724

# 2. Create venv and install (Python 3.12 recommended)
python -m venv .venv
. .venv/Scripts/activate           # Linux/macOS: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"            # editable install (no PYTHONPATH=. needed)
```

### Option B - Docker (recommended for reproducibility)

The repository ships a CPU-only [`Dockerfile`](Dockerfile) (Python 3.12-slim,
OpenCV/lungmask system libs, project installed editable). No local Python
toolchain required - only Docker.

```bash
# 1. Clone
git clone https://github.com/zucca98/progetto-computer-vision-s00003724.git
cd progetto-computer-vision-s00003724

# 2. Build the image (~5 min the first time, layers are cached afterwards)
docker build -t covid-ct .

# 3. Smoke-test the install by running the test suite inside the container
docker run --rm -v ${PWD}:/app -w /app covid-ct pytest tests/ -v

# 4. Run any pipeline script (mount the working directory so outputs persist)
docker run --rm -v ${PWD}:/app -w /app covid-ct \
    python scripts/prepare_data.py

# 5. Launch the Streamlit demo on http://localhost:8501
docker run --rm -p 8501:8501 -v ${PWD}:/app -w /app covid-ct \
    streamlit run app/demo.py --server.address 0.0.0.0
```

> The container is **CPU-only by design** (reproducibility for the classical
> pipeline). Deep-model training (`train_deep.py`) needs a CUDA GPU - run it
> directly via Option A on a machine with an NVIDIA GPU (the script picks up
> CUDA automatically), or use Google Colab / Kaggle as a fallback (see
> [`notebooks/03_train_colab.ipynb`](notebooks/03_train_colab.ipynb)).

### Run the pipeline

After installing via Option A or Option B, fetch the dataset and run the
classical pipeline:

```bash
# 1. Download dataset (Kaggle CLI required - see Dataset section)
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset
unzip sarscov2-ctscan-dataset.zip

# 2. Pipeline (CPU, ~10 min total for the classical track)
python scripts/prepare_data.py
python scripts/compute_norm_stats.py        # train-set normalisation
python scripts/extract_features.py
python scripts/train_classical.py
python scripts/run_full_evaluation.py

# 3. Train deep model - requires a CUDA GPU (local NVIDIA GPU or Colab; see notebooks/03_train_colab.ipynb)
python scripts/train_deep.py
```

When using Docker, prefix each `python ...` call with
`docker run --rm -v ${PWD}:/app -w /app covid-ct` (or open an interactive shell
via `docker run --rm -it -v ${PWD}:/app -w /app covid-ct bash`).

---

## Dataset

**SARS-CoV-2 CT-Scan** (Soares et al., 2020) - 2 481 PNG images, two classes:
`COVID/` and `non-COVID/`. License: CC BY 4.0.

Download from Kaggle:

```bash
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset
unzip sarscov2-ctscan-dataset.zip
```

> **Data-leakage note:** No patient identifiers are included. Slices from the same
> patient may span splits, which can inflate generalisation metrics. Declared
> explicitly in `docs/technical_analysis.md`.

---

## Repository Structure

```text
progetto-cv/
├── configs/
│   └── default.yaml             # All hyperparameters
├── data/processed/              # Generated train/val/test CSVs + NPZ features
├── docs/
│   ├── figures/                 # Auto-generated evaluation plots
│   ├── technical_analysis.md    # Technical report source
│   ├── technical_analysis.pdf   # Built PDF (<= 10 pages)
│   ├── MODEL_CARD.md            # Model card
│   └── DATASHEET.md             # Dataset datasheet
├── logs/                        # JSON result files
├── models/
│   ├── classical/               # Trained SVM and RF pickles
│   └── deep/                    # EfficientNet checkpoints
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing_demo.ipynb
│   ├── 03_train_colab.ipynb     # GPU training (Colab fallback; local GPU also supported via train_deep.py)
│   └── 04_results.ipynb         # Final comparison table
├── scripts/
│   ├── prepare_data.py                   # Stratified split -> CSV
│   ├── compute_norm_stats.py             # Train-set per-channel mean/std
│   ├── extract_features.py               # HOG + LBP + GLCM -> NPZ
│   ├── train_classical.py                # SVM / RF with GridSearchCV
│   ├── train_deep.py                     # Two-phase EfficientNet fine-tuning
│   ├── calibrate_and_evaluate.py         # Temperature scaling + threshold tuning
│   ├── generate_qualitative_examples.py  # TP/TN/FP/FN figure
│   └── run_full_evaluation.py            # All figures + comparison table
├── app/
│   └── demo.py                  # Streamlit upload-and-predict demo
├── Dockerfile                   # CPU-only reproducible env (classical pipeline + demo; deep training uses host GPU)
├── .github/workflows/ci.yml     # GitHub Actions CI (pytest)
├── src/
│   ├── preprocessing.py         # CLAHE, lung mask, augmentation
│   ├── features.py              # Handcrafted feature extraction
│   ├── models/
│   │   ├── classical.py         # SVM, RandomForest
│   │   └── deep.py              # EfficientNet-B0 + training loops
│   ├── postprocess.py           # Temperature scaling, TTA, Grad-CAM
│   └── evaluate.py              # Metrics, ROC/PR/calibration plots
└── tests/                       # Smoke tests (pytest)
```

---

## Reproducing Results

### Classical baselines (CPU, ~10 min total)

```bash
python scripts/prepare_data.py
python scripts/compute_norm_stats.py
python scripts/extract_features.py
python scripts/train_classical.py
python scripts/run_full_evaluation.py
python scripts/generate_qualitative_examples.py    # TP/TN/FP/FN figure
```

Results written to `logs/classical_results.json` and `docs/figures/`.

### Deep model (CUDA GPU required)

Two options:

- **Local NVIDIA GPU.** With a CUDA-enabled PyTorch install (Option A), run
  `python scripts/train_deep.py --arch efficientnet_b0 --batch-size 32
  --output-dir models/deep`. The script auto-detects the GPU via
  `torch.cuda.is_available()`.
- **Google Colab fallback.** Open `notebooks/03_train_colab.ipynb` in Colab
  (Runtime -> T4 GPU). Weights are saved to your Google Drive and persist after
  the session ends.

### Pre-trained weights

The trained checkpoint `best_model.pth` (~16 MB) is **not** committed to git
(`*.pth` is gitignored). To evaluate the deep model or run post-processing
without retraining (~4 h), download it from the GitHub Release:

```bash
mkdir -p models/deep

# With the GitHub CLI:
gh release download v0.1.0 --repo zucca98/progetto-computer-vision-s00003724 \
    --pattern best_model.pth --dir models/deep

# Or with curl:
curl -L -o models/deep/best_model.pth \
  https://github.com/zucca98/progetto-computer-vision-s00003724/releases/download/v0.1.0/best_model.pth
```

### Post-processing (CPU, requires a trained deep checkpoint)

```bash
python scripts/calibrate_and_evaluate.py \
    --model-dir models/deep \
    --output-dir models/postprocess
```

### Run tests

```bash
pytest tests/ -v
```

### Streamlit demo (academic; not for clinical use)

```bash
pip install streamlit
streamlit run app/demo.py
```

---

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zucca98/progetto-computer-vision-s00003724/blob/main/notebooks/03_train_colab.ipynb)

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{soares2020sars,
  title   = {{SARS-CoV-2 CT-scan dataset}: A large dataset of real patients
             CT scans for {SARS-CoV-2} identification},
  author  = {Soares, Eduardo and Angelov, Plamen and Biaso, Sarah and
             Froes, Michele Higa and Abe, Daniel Kanda},
  journal = {medRxiv},
  year    = {2020},
  doi     = {10.1101/2020.04.24.20078584}
}
```

---

## License

MIT - see `LICENSE`. Dataset: CC BY 4.0 (Soares et al., 2020).
