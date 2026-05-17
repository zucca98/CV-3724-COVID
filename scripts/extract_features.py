"""Calcola le feature handcrafted per ogni split e salva in file .npz.

Scopo dello script
------------------
**Terzo passo** della pipeline (dopo `prepare_data.py` e
`compute_norm_stats.py`). Per ogni immagine di ogni split (train/val/test):
1. Carica il PNG.
2. Converte in grayscale + resize 224.
3. Calcola la lung mask.
4. Estrae il vettore handcrafted `[LBP | GLCM | HOG]` (~26 mila feature)
   tramite `src.features.extract_handcrafted_features`.

Salva tre file in `data/processed/`:

    features_train.npz   { X: (n_train, F), y: (n_train,) }
    features_val.npz     { X: (n_val,   F), y: (n_val,)   }
    features_test.npz    { X: (n_test,  F), y: (n_test,)  }

Dove `F ~= 26 366` (26 LBP + 96 GLCM + 26 244 HOG).

Perche' un file .npz?
---------------------
`.npz` e' un archivio compresso di array NumPy, leggibile con
`np.load(path)`. Vantaggio: si caricano in memoria all'istante, evitando di
ri-eseguire l'estrazione delle feature (che e' lenta: lungmask + HOG su
~2500 immagini = ore di CPU).

Uso
---
    PYTHONPATH=. python scripts/extract_features.py             # tutti gli split
    PYTHONPATH=. python scripts/extract_features.py --split val # solo val
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import extract_handcrafted_features
from src.preprocessing import load_image, lung_mask, resize_with_padding
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)

# Convenzione del progetto: COVID = classe positiva (1).
_LABEL_MAP = {"covid": 1, "non-covid": 0}
# Dimensione standard del resize. Allineata a quello che si aspetta il
# backbone deep (EfficientNet-B0).
_TARGET_SIZE = 224


def _process_split(df: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Estrae le feature per tutte le immagini di uno split.

    Pipeline per ogni immagine:
        1. `load_image` -> array uint8 (H, W) o (H, W, 3).
        2. Conversione grayscale se serve.
        3. Resize a 224 con padding.
        4. `lung_mask` (lungmask U-Net o Otsu fallback).
        5. `extract_handcrafted_features` -> vettore 1D.

    Args:
        df: DataFrame con colonne `filepath`, `label`, `split`.
        split: Nome usato solo per la descrizione della progress bar.

    Returns:
        Tupla `(X, y)`:
        - `X`: array `(n_samples, n_features)` float64.
        - `y`: array `(n_samples,)` int64, label `{0=non-covid, 1=covid}`.
    """
    X: list[np.ndarray] = []
    y: list[int] = []

    # `df.iterrows()` itera riga per riga - lento per dataset enormi ma
    # qui (~2500 immagini) e' accettabile. `total=len(df)` permette a
    # tqdm di calcolare l'ETA.
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split, unit="img"):
        img = load_image(Path(row.filepath))
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = resize_with_padding(gray, target=_TARGET_SIZE)
        mask = lung_mask(gray)
        feat = extract_handcrafted_features(gray, lung_mask=mask)
        X.append(feat)
        y.append(_LABEL_MAP[row.label])

    # Conversione lista di array -> matrice 2D unica. `np.array` con dtype
    # esplicito previene autodetection a tipi piu' "grossi" (es. object).
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Extract handcrafted features from CT-scan splits."
    )
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Process one split only. Omit to process all three.",
    )
    parser.add_argument("--in-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    set_seed(data_cfg["seed"])

    in_dir = args.in_dir or Path(data_cfg["processed_dir"])
    out_dir = args.out_dir or Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Se l'utente non specifica `--split`, processiamo tutti e tre.
    splits = [args.split] if args.split else ["train", "val", "test"]

    for split in splits:
        csv_path = in_dir / f"{split}.csv"
        if not csv_path.exists():
            # Logghiamo e proseguiamo con i prossimi split (non blocchiamo
            # tutto): utile se ad es. uno split e' opzionale.
            logger.error("CSV not found: %s - run prepare_data.py first", csv_path)
            continue

        df = pd.read_csv(csv_path)
        logger.info("Processing %s (%d images)...", split, len(df))
        X, y = _process_split(df, split)

        out_path = out_dir / f"features_{split}.npz"
        # `savez_compressed`: archivio NPZ con compressione zip-style. Riduce
        # il file di ~5-10x rispetto a `savez` (le HOG hanno molti zeri).
        np.savez_compressed(out_path, X=X, y=y)
        logger.info(
            "Saved %s -> shape X=%s  y=%s  (covid=%d, non-covid=%d)",
            out_path,
            X.shape,
            y.shape,
            y.sum(),
            (y == 0).sum(),
        )


if __name__ == "__main__":
    main()
