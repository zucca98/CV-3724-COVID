"""Valuta il modello deep con Test-Time Augmentation (TTA) sul test set.

Scopo dello script
------------------
Passo opzionale di post-processing (dopo `train_deep.py`). Per ogni immagine
di test genera N viste augmentate deterministiche, media le probabilita'
softmax (`src.postprocess.tta_predict`) e calcola le metriche finali.

Le viste TTA sono **deterministiche** (parametri fissi, non random): cosi'
la valutazione e' riproducibile. Con N=5: originale, flip orizzontale,
rotazione +7 gradi, rotazione -7 gradi, zoom +8%.

Uso
---
    PYTHONPATH=. python scripts/evaluate_tta.py
    PYTHONPATH=. python scripts/evaluate_tta.py --n-views 5 --no-lung-mask
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.deep import build_model
from src.postprocess import tta_predict
from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    load_image,
    lung_mask,
    resize_with_padding,
)
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)

_NORM_STATS_PATH = Path("configs/norm_stats.json")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_norm_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Carica mean/std del train set, con fallback su statistiche ImageNet."""
    if _NORM_STATS_PATH.exists():
        data = json.loads(_NORM_STATS_PATH.read_text())
        return tuple(data["mean"]), tuple(data["std"])
    logger.warning("norm_stats.json not found - falling back to ImageNet stats")
    return _IMAGENET_MEAN, _IMAGENET_STD


def _tta_transforms(n_views: int, mean: tuple, std: tuple) -> list[A.Compose]:
    """Costruisce fino a `n_views` viste TTA deterministiche (Normalize+ToTensor)."""
    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    views = [
        A.Compose(norm),                                             # originale
        A.Compose([A.HorizontalFlip(p=1.0), *norm]),                 # flip
        A.Compose([A.Rotate(limit=(7, 7), p=1.0), *norm]),           # +7 gradi
        A.Compose([A.Rotate(limit=(-7, -7), p=1.0), *norm]),         # -7 gradi
        A.Compose([A.Affine(scale=(1.08, 1.08), p=1.0), *norm]),     # zoom +8%
    ]
    return views[:n_views]


def _preprocess(filepath: Path, apply_mask: bool) -> np.ndarray:
    """Riproduce la pipeline di `CovidCTDataset` fino all'immagine 3-canali."""
    img = load_image(filepath)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = resize_with_padding(gray, target=224)
    if apply_mask:
        gray = apply_lung_mask(gray, lung_mask(gray))
    return build_three_channel(gray)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA evaluation of the deep model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deep"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--n-views", type=int, default=None,
                        help="numero di viste TTA (default: evaluation.tta_n_views)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/postprocess"))
    parser.add_argument("--no-lung-mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    apply_mask = not args.no_lung_mask
    n_views = args.n_views or cfg["evaluation"]["tta_n_views"]

    # ----------------------------------------------------------- Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = build_model(
        arch=model_cfg["arch"], num_classes=model_cfg["num_classes"],
        pretrained=False, dropout=model_cfg["dropout"],
    )
    checkpoint = args.model_dir / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s (device=%s)", checkpoint, device)

    mean, std = _load_norm_stats()
    transforms_list = _tta_transforms(n_views, mean, std)
    logger.info("TTA with %d views", len(transforms_list))

    # ------------------------------------------------- Test-set sweep
    test_df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "test.csv")
    label_map = {"covid": 1, "non-covid": 0}

    y_true: list[int] = []
    p_covid: list[float] = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        img3 = _preprocess(Path(row["filepath"]), apply_mask)
        probs = tta_predict(model, img3, transforms_list, device)
        y_true.append(label_map[row["label"]])
        p_covid.append(float(probs[1]))
        if (i + 1) % 100 == 0:
            logger.info("  processed %d/%d", i + 1, len(test_df))

    y_true_arr = np.array(y_true)
    p_arr = np.array(p_covid)
    preds = (p_arr >= 0.5).astype(int)

    results = {
        "arch": model_cfg["arch"],
        "n_views": len(transforms_list),
        "apply_lung_mask": apply_mask,
        "threshold": 0.5,
        "test_accuracy": float(accuracy_score(y_true_arr, preds)),
        "test_precision": float(precision_score(y_true_arr, preds, zero_division=0)),
        "test_recall": float(recall_score(y_true_arr, preds, zero_division=0)),
        "test_f1": float(f1_score(y_true_arr, preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true_arr, p_arr)),
    }

    logger.info("=== TTA test results (%d views) ===", len(transforms_list))
    logger.info("acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                results["test_accuracy"], results["test_precision"],
                results["test_recall"], results["test_f1"], results["test_roc_auc"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "tta_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Results -> %s", out_json)


if __name__ == "__main__":
    main()
