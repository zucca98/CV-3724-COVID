"""Estrae esempi qualitativi (TP/TN/FP/FN) dal classificatore SVM sul test set.

Scopo dello script
------------------
Per la sezione "Failure Analysis" del report tecnico, vogliamo mostrare al
lettore alcune immagini concrete con le predizioni del modello, divise per
i quattro casi della confusion matrix:
- **TP** (True Positive)  - COVID predetto correttamente.
- **TN** (True Negative)  - non-COVID predetto correttamente.
- **FP** (False Positive) - non-COVID classificato come COVID (errore!).
- **FN** (False Negative) - COVID classificato come non-COVID (errore grave!).

Output
------
File PNG `docs/figures/qualitative_examples.png`: griglia 2x3 con 2 TP, 2 TN,
1 FP, 1 FN. Ogni immagine annotata con label vera, predetta e probabilita'
di COVID.

Uso
---
    PYTHONPATH=. python scripts/generate_qualitative_examples.py

Prerequisiti
------------
- `models/classical/svm.pkl` (da `train_classical.py`).
- `data/processed/features_test.npz` (da `extract_features.py`).
- `data/processed/test.csv` (da `prepare_data.py`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    load_image,
    lung_mask,
    resize_with_padding,
)
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def _pick_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_tp: int = 2,
    n_tn: int = 2,
    n_fp: int = 1,
    n_fn: int = 1,
) -> list[tuple[int, str]]:
    """Campiona indici di esempi nelle 4 categorie TP/TN/FP/FN.

    Strategia: per ogni categoria troviamo l'indice dei campioni che vi
    appartengono, poi ne scegliamo casualmente `n` (con seed 42 per
    riproducibilita').

    Args:
        y_true: Label vere `{0, 1}`.
        y_pred: Predizioni hard `{0, 1}`.
        y_proba: Probabilita' (non usate qui, ma incluse per simmetria).
        n_tp, n_tn, n_fp, n_fn: Numero di campioni per categoria.

    Returns:
        Lista di tuple `(indice, "TP"|"TN"|"FP"|"FN")`. Puo' essere piu' corta
        del totale richiesto se una categoria e' vuota.
    """
    # Maschere booleane per ciascuna categoria della confusion matrix.
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    # `default_rng(42)` per sample deterministico (la nuova API di numpy
    # e' preferita rispetto al vecchio `np.random.seed`).
    rng = np.random.default_rng(42)
    chosen: list[tuple[int, str]] = []
    for arr, n, kind in [(tp, n_tp, "TP"), (tn, n_tn, "TN"), (fp, n_fp, "FP"), (fn, n_fn, "FN")]:
        if len(arr) == 0:
            # Categoria vuota = il modello e' molto preciso (o molto male):
            # logghiamo e proseguiamo senza crashare.
            logger.warning("No %s examples available - skipping", kind)
            continue
        k = min(n, len(arr))  # non chiediamo piu' campioni di quelli disponibili
        idx = rng.choice(arr, size=k, replace=False)
        chosen.extend((int(i), kind) for i in idx)
    return chosen


def _render_slice(filepath: Path, target: int, apply_mask: bool) -> np.ndarray:
    """Renderizza una slice CT applicando la stessa pipeline del training.

    Sequenza: load -> grayscale -> resize -> lung mask -> build 3-channel.
    Il risultato viene mostrato direttamente con `plt.imshow` (i 3 canali
    danno un effetto RGB-falso-colore che evidenzia il preprocessing).

    Args:
        filepath: Path al PNG originale.
        target: Lato del quadrato di output.
        apply_mask: Se applicare la lung mask.

    Returns:
        Array `(target, target, 3)` uint8.
    """
    img = load_image(filepath)
    # `img.mean(axis=2)` come fallback in caso `cvtColor` non sia disponibile
    # per ragioni di pipeline (semantica leggermente diversa da BGR->GRAY
    # ma sufficiente per uso visualizzativo).
    gray = img if img.ndim == 2 else img.mean(axis=2).astype(np.uint8)
    gray = resize_with_padding(gray, target=target)
    if apply_mask:
        gray = apply_lung_mask(gray, lung_mask(gray))
    return build_three_channel(gray)


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Generate qualitative TP/TN/FP/FN examples.")
    parser.add_argument("--svm-model", type=Path, default=Path("models/classical/svm.pkl"))
    parser.add_argument("--features",  type=Path, default=Path("data/processed/features_test.npz"))
    parser.add_argument("--test-csv",  type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out",       type=Path, default=Path("docs/figures/qualitative_examples.png"))
    parser.add_argument("--config",    type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--no-lung-mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    # Controllo prerequisiti: fallisci subito con messaggio utile se manca
    # qualcosa.
    for p in (args.svm_model, args.features, args.test_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p} - run scripts/prepare_data.py, extract_features.py, "
                "and train_classical.py first."
            )

    svm = joblib.load(args.svm_model)
    feats = np.load(args.features)
    X, y = feats["X"], feats["y"]
    df = pd.read_csv(args.test_csv).reset_index(drop=True)
    # Sanity check: la i-esima riga del CSV deve corrispondere alla i-esima
    # feature row. Se i due array sono lunghi diversamente, qualcosa e'
    # disallineato e i grafici sarebbero ingannevoli.
    if len(df) != len(X):
        raise RuntimeError(f"Length mismatch: test.csv has {len(df)} rows, features have {len(X)}.")

    proba = svm.predict_proba(X)[:, 1]
    # Soglia standard 0.5 - per il qualitativo non serve calibrazione.
    pred = (proba >= 0.5).astype(int)

    chosen = _pick_examples(y, pred, proba)
    if not chosen:
        raise RuntimeError("No examples could be selected - check predictions.")

    target = cfg["preprocessing"]["target_size"]
    # `--no-lung-mask` flag + config: l'utente puo' disabilitare la mask
    # via CLI O via config; il flag CLI ha precedenza.
    apply_mask = not args.no_lung_mask and cfg["preprocessing"].get("apply_lung_mask", True)

    # Layout della griglia: 3 colonne, righe calcolate automaticamente.
    n = len(chosen)
    cols = 3
    rows = (n + cols - 1) // cols   # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    # `axes` puo' essere un singolo Axes (1x1) o un array 2D; flatten
    # per iterazione uniforme.
    axes = np.array(axes).reshape(-1)

    label_names = {0: "non-COVID", 1: "COVID"}
    # Verde per predizioni corrette (TP/TN), rosso per errori (FP/FN).
    color_map = {"TP": "#2a8c2a", "TN": "#2a8c2a", "FP": "#c44b4b", "FN": "#c44b4b"}

    for ax, (i, kind) in zip(axes, chosen):
        fp = Path(df.loc[i, "filepath"])
        rendered = _render_slice(fp, target=target, apply_mask=apply_mask)
        ax.imshow(rendered)
        # Niente tick: vogliamo un'immagine "pulita" come in un report medico.
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{kind}  |  true={label_names[int(y[i])]}\npred={label_names[int(pred[i])]}  "
            f"p(COVID)={proba[i]:.3f}",
            fontsize=9, color=color_map[kind],
        )

    # Spegne gli assi residui (se la griglia non e' piena - es. 5 esempi
    # in 2x3 = 1 cella vuota).
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle("SVM qualitative examples on the held-out test set", fontsize=12)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    logger.info("Saved qualitative examples -> %s", args.out)


if __name__ == "__main__":
    main()
