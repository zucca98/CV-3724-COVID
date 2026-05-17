"""Calibra un modello deep allenato con temperature scaling e lo valuta sul test set.

Scopo dello script
------------------
**Quinto passo** della pipeline (dopo `train_deep.py`). Applica due tecniche
di post-processing al modello allenato:

1. **Temperature scaling**: trova lo scalare T (sul validation set) che
   minimizza la NLL una volta che i logit vengono divisi per T. Calibra
   le probabilita' (riduce overconfidence).

2. **Threshold tuning per sensibilita'**: trova la soglia di decisione
   piu' alta (= piu' specifica) che garantisce comunque
   `sensibilita' >= target` (default 0.95) sul validation set.

Output prodotti
---------------
- `<output_dir>/reliability_diagram.png`  - grafico di calibrazione
  before/after.
- `<output_dir>/calibration_results.json` - metriche raw vs calibrate,
  T, soglia.

Workflow tecnico
----------------
    val_logits  -> temperature_scaling -> T
    val_probs_cal = apply_temperature(val_logits, T)
    threshold = tune_threshold_for_sensitivity(val_probs_cal, val_labels)
    test_metrics_raw = with default threshold 0.5
    test_metrics_cal = with tuned threshold
    plot reliability before/after

Uso
---
    PYTHONPATH=. python scripts/calibrate_and_evaluate.py
    PYTHONPATH=. python scripts/calibrate_and_evaluate.py --target-sens 0.99
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
# `matplotlib.use("Agg")` forza il backend non-interattivo: necessario su
# server headless (Colab, Kaggle, CI) dove non c'e' un display X11.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.models.deep import CovidCTDataset, build_model
from src.postprocess import apply_temperature, temperature_scaling, tune_threshold_for_sensitivity
from src.preprocessing import get_eval_transform
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def _collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward di tutti i batch e restituzione di `(logits, labels)`.

    Lavoriamo sui *logit* (non sui softmax) perche' la temperature scaling
    opera DIRETTAMENTE sui logit: dividerli per T prima del softmax e'
    matematicamente equivalente a modificare la "temperatura" della
    distribuzione.

    Args:
        model: Modello allenato in eval mode.
        loader: DataLoader (val o test).
        device: Dispositivo di calcolo.

    Returns:
        Tupla `(logits, labels)`:
        - `logits`: array `(N, 2)` float.
        - `labels`: array `(N,)` int.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X).cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(y.numpy().tolist())
    # `np.concatenate(axis=0)`: impila i batch lungo la dimensione 0.
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def _compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcola le metriche standard date `y_true`, `probs` e una soglia.

    Args:
        y_true: Label vere.
        probs: Probabilita' classe positiva.
        threshold: Soglia di decisione `prob >= threshold -> positivo`.

    Returns:
        Dizionario con accuracy, precision, recall, f1, roc_auc, threshold.
        Nota: `roc_auc` NON dipende dalla soglia (si calcola direttamente
        dalle probabilita').
    """
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, probs)),
        "threshold": float(threshold),
    }


def _plot_reliability(
    y_true: np.ndarray,
    probs_raw: np.ndarray,
    probs_cal: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Salva un reliability diagram with curve PRE e POST calibrazione.

    Lettura: piu' le curve si avvicinano alla diagonale, meglio sono
    calibrate. Tipicamente la curva "raw" (rossa) e' sotto la diagonale
    (overconfident); dopo temperature scaling si avvicina.

    Args:
        y_true: Label vere.
        probs_raw: Probabilita' pre-calibrazione.
        probs_cal: Probabilita' post-calibrazione.
        out_path: File PNG di output.
        n_bins: Numero di bin del diagramma.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, probs_raw, n_bins=n_bins)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, probs_cal, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred_raw, frac_pos_raw, "o-", color="#e05c5c", label="Before scaling")
    ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#5c9be0", label="After scaling")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Reliability diagram saved to %s", out_path)


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Temperature scaling calibration and post-processing evaluation."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/deep"))
    parser.add_argument("--arch", default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--target-sens", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("models/postprocess"))
    parser.add_argument("--no-lung-mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- Model
    model_cfg = cfg["model"]
    arch = args.arch or model_cfg["arch"]
    # `pretrained=False` perche' carichiamo i nostri pesi del fine-tuning,
    # NON quelli di ImageNet (sarebbero subito sovrascritti).
    model = build_model(
        arch=arch,
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        dropout=model_cfg["dropout"],
    )
    checkpoint = args.model_dir / "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # `map_location=device` rende il caricamento robusto al device: posso
    # caricare un checkpoint salvato su GPU anche se sono su CPU.
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    logger.info("Loaded checkpoint: %s", checkpoint)

    # ---------------------------------------------------------- Data
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_cfg = cfg["training"]
    batch_size: int = train_cfg["batch_size"]
    num_workers: int = train_cfg.get("num_workers", 0)
    apply_mask = not args.no_lung_mask
    transform = get_eval_transform()

    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    val_loader = DataLoader(
        CovidCTDataset(val_df, transform, apply_mask=apply_mask),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        CovidCTDataset(test_df, transform, apply_mask=apply_mask),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    # ------------------------------ Temperature calibration su val ----
    # Usiamo SEMPRE il val set per la calibrazione, MAI il test
    # (sarebbe data leakage).
    logger.info("Collecting validation logits ...")
    val_logits, val_labels = _collect_logits(model, val_loader, device)

    T = temperature_scaling(val_logits, val_labels)
    val_probs_cal = apply_temperature(val_logits, T)
    # Soglia ottimale sul val set calibrato.
    thr = tune_threshold_for_sensitivity(val_probs_cal, val_labels, args.target_sens)

    # ----------------------------------- Test set evaluation ----------
    logger.info("Collecting test logits ...")
    test_logits, test_labels = _collect_logits(model, test_loader, device)

    # Probabilita' RAW: softmax dei logit senza temperature scaling.
    # `torch.tensor(...)` perche' softmax PyTorch e' piu' comodo del manuale.
    test_probs_raw = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    test_probs_cal = apply_temperature(test_logits, T)

    # Confronto RAW (soglia default 0.5) vs CAL (soglia ottimizzata).
    raw_metrics = _compute_metrics(test_labels, test_probs_raw, threshold=0.5)
    cal_metrics = _compute_metrics(test_labels, test_probs_cal, threshold=thr)

    logger.info("=== Test results ===")
    logger.info("RAW (thr=0.50)  acc=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                raw_metrics["accuracy"], raw_metrics["recall"],
                raw_metrics["f1"], raw_metrics["roc_auc"])
    logger.info("CAL (thr=%.4f)  acc=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                thr, cal_metrics["accuracy"], cal_metrics["recall"],
                cal_metrics["f1"], cal_metrics["roc_auc"])

    # ------------------------------------- Reliability diagram --------
    _plot_reliability(
        test_labels, test_probs_raw, test_probs_cal,
        args.output_dir / "reliability_diagram.png",
    )

    # ----------------------------------------- Save results -----------
    results = {
        "arch": arch,
        "checkpoint": str(checkpoint),
        "temperature": T,
        "threshold": thr,
        "target_sensitivity": args.target_sens,
        "apply_lung_mask": apply_mask,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
    }
    out_json = args.output_dir / "calibration_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results -> %s", out_json)


if __name__ == "__main__":
    main()
