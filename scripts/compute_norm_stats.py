"""Calcola media e deviazione standard per canale sul training set.

Scopo dello script
------------------
**Secondo passo** della pipeline (dopo `prepare_data.py`). Stima i due
vettori `mean` e `std` (uno per ciascuno dei 3 canali dell'immagine
post-`build_three_channel`) usati per la normalizzazione `(x - mean) / std`
sia in training sia in inferenza.

Perche' "training set only"?
----------------------------
Calcolare le statistiche sull'intero dataset (o sul test set) costituisce
**data leakage**: informazione del test set "trapela" nel training tramite
il preprocessing. Per essere onesti dobbiamo simulare lo scenario reale in
cui il test set e' "ignoto" -> usiamo solo il training.

Output
------
File JSON (default `configs/norm_stats.json`):

    {
      "mean": [..., ..., ...],
      "std":  [..., ..., ...],
      "n_images": <numero di immagini usate>,
      "target_size": <dimensione resize>,
      "lung_mask": <true/false>
    }

`src/preprocessing._load_norm_stats()` legge questo file all'import.

Algoritmo
---------
Streaming a una passata (no caricamento di tutto in memoria):

    sum  += sum(pixel_values)        per ogni canale
    sumsq += sum(pixel_values ** 2)  per ogni canale
    count += n_pixel

Alla fine: `mean = sum/count`, `var = sumsq/count - mean**2`,
`std = sqrt(max(var, 0))` (clip a 0 per stabilita' numerica, casi
patologici di immagini quasi costanti).

Note sul preprocessing
----------------------
Le statistiche vengono calcolate **dopo** la stessa catena che useremo
in training: resize -> lung_mask -> build_three_channel. Cosi'
mean/std rispecchiano la distribuzione *effettiva* degli input alla rete.

Uso
---
    PYTHONPATH=. python scripts/compute_norm_stats.py
    PYTHONPATH=. python scripts/compute_norm_stats.py --no-lung-mask --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    load_image,
    lung_mask,
    resize_with_padding,
)
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def _channel_stats(filepaths: list[str], apply_mask: bool, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Calcola media e std per canale su una lista di immagini.

    Algoritmo single-pass:
        sum   = somma cumulata dei valori pixel per ciascun canale.
        sumsq = somma cumulata dei quadrati dei valori pixel.
        count = numero totale di pixel.

    Alla fine: `mean = sum/count`, `var = (sumsq/count) - mean^2`.

    Vantaggio rispetto al "carica tutto in RAM e usa np.mean": occupa O(1)
    memoria invece di O(N * H * W * C) - cruciale per dataset grandi.

    Args:
        filepaths: Lista di percorsi PNG (stringhe).
        apply_mask: Se applicare la lung mask prima di calcolare le stats
            (consigliato true per replicare l'input reale al modello).
        target: Dimensione del resize (es. 224).

    Returns:
        Tupla `(mean, std)` con array shape `(3,)` ciascuno (scala 0-1).
    """
    # `float64` per evitare overflow nelle somme cumulate.
    sum_  = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    count = 0

    for fp in tqdm(filepaths, desc="train images"):
        img = load_image(Path(fp))
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = resize_with_padding(gray, target=target)
        if apply_mask:
            gray = apply_lung_mask(gray, lung_mask(gray))
        # Normalizziamo in [0, 1] (dividiamo per 255) prima delle stats:
        # le stats `mean/std` saranno cosi' confrontabili con ImageNet
        # (che usa la stessa convenzione).
        img3 = build_three_channel(gray).astype(np.float64) / 255.0
        # Appiattiamo H x W x 3 in (H*W, 3) per fare somma per canale.
        flat = img3.reshape(-1, 3)

        sum_  += flat.sum(axis=0)
        sumsq += (flat ** 2).sum(axis=0)
        count += flat.shape[0]

    mean = sum_ / count
    # Identita': Var(X) = E(X^2) - E(X)^2.
    # `clip(var, 0, None)` mette a 0 eventuali varianze leggermente
    # negative dovute ad errori di arrotondamento (matematicamente var >= 0,
    # ma in floating point puo' venire -1e-18).
    var = (sumsq / count) - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Compute training-set normalisation statistics.")
    parser.add_argument("--config",  type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out",     type=Path, default=Path("configs/norm_stats.json"))
    parser.add_argument("--no-lung-mask", action="store_true", help="Skip lung masking (faster).")
    parser.add_argument("--target",  type=int,  default=224)
    parser.add_argument("--limit",   type=int,  default=None,
                        help="Use only the first N training images (debug).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["data"]["processed_dir"])
    train_csv = processed / "train.csv"
    if not train_csv.exists():
        # Errore "esplicito": diciamo all'utente quale comando lanciare per
        # risolvere, invece di mostrare un FileNotFoundError generico.
        raise FileNotFoundError(f"Run scripts/prepare_data.py first - missing {train_csv}")

    df = pd.read_csv(train_csv)
    if args.limit:
        # Modalita' debug: usa solo le prime N immagini per test rapidi.
        df = df.head(args.limit)
    logger.info("Computing stats on %d training images (lung_mask=%s, target=%d)",
                len(df), not args.no_lung_mask, args.target)

    mean, std = _channel_stats(df["filepath"].tolist(), not args.no_lung_mask, args.target)

    # Serializzazione JSON: convertiamo numpy float in python float perche'
    # `json` non sa gestire numpy types nativamente.
    payload = {
        "mean": [float(m) for m in mean],
        "std":  [float(s) for s in std],
        "n_images": int(len(df)),
        "target_size": args.target,
        "lung_mask": not args.no_lung_mask,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("mean = %s", payload["mean"])
    logger.info("std  = %s", payload["std"])
    logger.info("Saved -> %s", args.out)


if __name__ == "__main__":
    main()
