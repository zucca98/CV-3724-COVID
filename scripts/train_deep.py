"""Allena e valuta il modello deep learning sul dataset CT.

Scopo dello script
------------------
Entry point per il **training del modello deep**. Orchestra:
1. Caricamento dei CSV degli split (`prepare_data.py` deve essere gia' stato eseguito).
2. Creazione dei `DataLoader` PyTorch con la pipeline di augmentation/eval.
3. Costruzione del modello via `build_model` (timm backbone + testa custom).
4. Training a due fasi (`fit` in `src/models/deep.py`).
5. Valutazione finale sul test set con il miglior checkpoint.
6. Salvataggio metriche in `<output_dir>/test_results.json`.

Dove va eseguito?
-----------------
Il training deep richiede GPU. Le opzioni sono (in ordine di preferenza):
- Macchina locale con GPU NVIDIA (CUDA) - lo script rileva la GPU
  automaticamente via `torch.cuda.is_available()`. E' la via piu' veloce
  per il dev box di questo progetto (RTX 3080).
- Google Colab (T4 free tier) - fallback comodo via
  `notebooks/03_train_colab.ipynb`, utile per chi non ha GPU locale.
- Kaggle Notebooks.

L'esecuzione su CPU funziona ma e' molto lenta. Per smoke test rapidi
usare `--subset 50 --no-lung-mask`.

Argomenti CLI principali
------------------------
- `--arch`            : backbone (es. efficientnet_b0, densenet121). Default da config.
- `--batch-size`      : dimensione batch. Default da config.
- `--epochs-phase1/2` : numero epoche delle due fasi.
- `--subset`          : limita ogni split a N immagini (per smoke test).
- `--no-lung-mask`    : salta la lungmask (smoke test piu' veloce).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.models.deep import (
    CovidCTDataset,
    build_model,
    fit,
    validate_one_epoch,
)
from src.preprocessing import get_eval_transform, get_train_transform
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sub-sample stratificato di un DataFrame: restituisce fino a `n` righe per label.

    Mantiene la proporzione COVID/non-COVID del DataFrame originale.
    Usato in modalita' `--subset` per smoke test rapidi.

    Args:
        df: DataFrame con colonna `label`.
        n: Numero totale di righe desiderate (approssimato dopo
            l'arrotondamento per classe).

    Returns:
        DataFrame ridotto, con indice resettato.
    """
    groups = []
    for label, grp in df.groupby("label"):
        # Numero di sample da questa classe = proporzione * n, almeno 1.
        k = max(1, round(n * len(grp) / len(df)))
        # `random_state=42` per riproducibilita' del sample.
        groups.append(grp.sample(min(k, len(grp)), random_state=42))
    return pd.concat(groups).reset_index(drop=True)


def _evaluate_on_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Valuta il modello sul test set: forward pass + 5 metriche.

    Args:
        model: Modello allenato.
        loader: DataLoader del test set.
        device: Dispositivo di calcolo.

    Returns:
        Dizionario con `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            # Prob della classe positiva (COVID) per ROC-AUC.
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            # Hard prediction = argmax (soglia implicita 0.5).
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Two-phase fine-tuning of a timm CNN on CT-scan images."
    )
    # Tutti gli override CLI hanno `default=None` per significare "usa il
    # valore del config file" (gestito sotto con l'operatore `or`).
    parser.add_argument("--arch", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs-phase1", type=int, default=None)
    parser.add_argument("--epochs-phase2", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/deep"))
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit each split to N images (stratified). Useful for smoke-tests.",
    )
    parser.add_argument(
        "--no-lung-mask", action="store_true",
        help="Skip lung masking (faster on CPU for smoke-tests).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # CLI overrides: il valore CLI (se passato) vince sul config.
    arch = args.arch or model_cfg["arch"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    # Mutiamo direttamente il dict cfg per gli override delle epoche - cosi'
    # `fit()` riceve la versione aggiornata.
    if args.epochs_phase1 is not None:
        train_cfg["epochs_phase1"] = args.epochs_phase1
    if args.epochs_phase2 is not None:
        train_cfg["epochs_phase2"] = args.epochs_phase2

    set_seed(cfg["data"]["seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- Load CSVs
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    if args.subset:
        # Riduzione stratificata. Val e test prendono N/4 ciascuno (con un
        # minimo di 8 per evitare che siano troppo piccoli).
        train_df = _stratified_sample(train_df, args.subset)
        val_df   = _stratified_sample(val_df,   max(8, args.subset // 4))
        test_df  = _stratified_sample(test_df,  max(8, args.subset // 4))
        logger.info("Subset mode: train=%d  val=%d  test=%d",
                    len(train_df), len(val_df), len(test_df))

    apply_mask = not args.no_lung_mask

    # ------------------------------------------------ Datasets & loaders
    # Solo il training usa la pipeline con augmentation; val e test usano
    # solo normalizzazione (`get_eval_transform`).
    train_ds = CovidCTDataset(train_df, get_train_transform(), apply_mask=apply_mask)
    val_ds   = CovidCTDataset(val_df,   get_eval_transform(),  apply_mask=apply_mask)
    test_ds  = CovidCTDataset(test_df,  get_eval_transform(),  apply_mask=apply_mask)

    num_workers: int = train_cfg.get("num_workers", 0)
    # `pin_memory=True` su CUDA accelera il trasferimento CPU -> GPU
    # (pagine "pinned" non swappabili). Su CPU non ha effetto.
    pin_memory = torch.cuda.is_available()

    # `shuffle=True` SOLO sul training: val e test devono essere
    # deterministici (per metriche riproducibili).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # --------------------------------------------------------- Model
    model = build_model(
        arch=arch,
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )

    # ------------------------------------------------------- Training
    # `fit` esegue le due fasi (head-only + partial unfreeze) e salva il
    # best checkpoint in `args.output_dir/best_model.pth`.
    best_path = fit(model, train_loader, val_loader, train_cfg, args.output_dir)

    # -------------------------------------------------- Test evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ricarichiamo il best checkpoint (potrebbe NON essere l'ultimo per
    # via dell'early stopping / best-val-loss tracking).
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)

    test_metrics = _evaluate_on_test(model, test_loader, device)

    logger.info("=" * 60)
    logger.info("TEST  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                test_metrics["accuracy"], test_metrics["precision"],
                test_metrics["recall"],   test_metrics["f1"],
                test_metrics["roc_auc"])

    # Prefisso "test_" per evitare collisioni quando questi numeri vengono
    # poi inclusi in dict piu' grandi (vedi `run_full_evaluation.py`).
    results = {
        "arch": arch,
        "subset": args.subset,
        "apply_lung_mask": apply_mask,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    out_json = args.output_dir / "test_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results -> %s", out_json)


if __name__ == "__main__":
    main()
