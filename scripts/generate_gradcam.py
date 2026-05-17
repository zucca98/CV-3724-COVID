"""Genera la figura Grad-CAM per il modello deep addestrato.

Scopo dello script
------------------
Passo opzionale di interpretabilita' (dopo `train_deep.py`). Carica il
checkpoint EfficientNet-B0, seleziona un esempio per ognuna delle quattro
categorie di confusione sul test set (TP / TN / FP / FN) e produce una
griglia 2 x 4: riga 1 immagine preprocessata, riga 2 heatmap Grad-CAM
sovrapposta. La figura serve la sezione *Model Transparency* del report.

La heatmap e' sempre calcolata per la classe COVID (indice 1): mostra
"dove il modello vede COVID", utile sia sui veri positivi sia sui falsi
positivi.

Uso
---
    PYTHONPATH=. python scripts/generate_gradcam.py
    PYTHONPATH=. python scripts/generate_gradcam.py --model-dir models/deep \\
        --output docs/figures/gradcam_examples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.deep import build_model
from src.postprocess import compute_gradcam
from src.preprocessing import (
    apply_lung_mask,
    build_three_channel,
    get_eval_transform,
    load_image,
    lung_mask,
    resize_with_padding,
)
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)

# Le quattro categorie della confusion matrix, in ordine di colonna.
_CATEGORIES = ["TP", "TN", "FP", "FN"]


def _preprocess(filepath: Path, apply_mask: bool) -> np.ndarray:
    """Riproduce la pipeline di `CovidCTDataset` fino all'immagine 3-canali.

    Args:
        filepath: Percorso del PNG.
        apply_mask: Se applicare la lung mask.

    Returns:
        Immagine ``(224, 224, 3)`` uint8, pronta per transform/Grad-CAM.
    """
    img = load_image(filepath)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = resize_with_padding(gray, target=224)
    if apply_mask:
        gray = apply_lung_mask(gray, lung_mask(gray))
    return build_three_channel(gray)


def _categorise(label: int, pred: int) -> str | None:
    """Mappa ``(label, pred)`` su TP/TN/FP/FN (classe positiva = COVID = 1)."""
    return {
        (1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN",
    }.get((label, pred))


def _overlay(img3: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Sovrappone una heatmap Grad-CAM (jet) al canale grayscale dell'immagine."""
    base = cv2.cvtColor(img3[:, :, 0], cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    cmap = (plt.get_cmap("jet")(heatmap)[:, :, :3]).astype(np.float32)
    blended = 0.55 * base + 0.45 * cmap
    return np.clip(blended, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Grad-CAM figure.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/deep"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/figures/gradcam_examples.png"))
    parser.add_argument("--no-lung-mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    apply_mask = not args.no_lung_mask

    # ----------------------------------------------------------- Model
    # Grad-CAM gira su CPU: la libreria non sposta l'input sul device del
    # modello, quindi tenere tutto su CPU evita un mismatch CUDA/CPU.
    device = torch.device("cpu")
    model_cfg = cfg["model"]
    model = build_model(
        arch=model_cfg["arch"], num_classes=model_cfg["num_classes"],
        pretrained=False, dropout=model_cfg["dropout"],
    )
    checkpoint = args.model_dir / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s", checkpoint)

    # EfficientNet (timm): l'ultimo blocco convoluzionale e' il target layer.
    target_layer = model.backbone.blocks[-1][-1]
    transform = get_eval_transform()

    # ------------------------------------------- Scan test set for one
    # ------------------------------------------- example per category.
    test_df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "test.csv")
    label_map = {"covid": 1, "non-covid": 0}
    found: dict[str, dict] = {}

    for _, row in test_df.iterrows():
        if len(found) == 4:
            break
        label = label_map[row["label"]]
        img3 = _preprocess(Path(row["filepath"]), apply_mask)
        with torch.no_grad():
            tensor = transform(image=img3)["image"].unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0].numpy()
        pred = int(probs[1] >= 0.5)
        cat = _categorise(label, pred)
        if cat and cat not in found:
            found[cat] = {"img3": img3, "label": label, "pred": pred,
                          "p_covid": float(probs[1])}
            logger.info("Found %s: %s (p_covid=%.3f)", cat,
                        Path(row["filepath"]).name, probs[1])

    missing = [c for c in _CATEGORIES if c not in found]
    if missing:
        logger.warning("Categories not found in test set: %s", missing)

    # ----------------------------------------------- Compose 2x4 figure
    cols = [c for c in _CATEGORIES if c in found]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.6))
    if len(cols) == 1:
        axes = axes.reshape(2, 1)

    for j, cat in enumerate(cols):
        item = found[cat]
        img3 = item["img3"]
        heatmap = compute_gradcam(model, img3, target_layer, target_class=1)

        axes[0, j].imshow(img3[:, :, 0], cmap="gray")
        axes[0, j].set_title(
            f"{cat}\ntrue={'COVID' if item['label'] else 'non-COVID'}  "
            f"pred={'COVID' if item['pred'] else 'non-COVID'}\n"
            f"p(COVID)={item['p_covid']:.3f}", fontsize=9)
        axes[1, j].imshow(_overlay(img3, heatmap))
        axes[1, j].set_title("Grad-CAM (COVID class)", fontsize=9)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")

    fig.suptitle("Grad-CAM on EfficientNet-B0 test predictions", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grad-CAM figure saved to %s", args.output)


if __name__ == "__main__":
    main()
