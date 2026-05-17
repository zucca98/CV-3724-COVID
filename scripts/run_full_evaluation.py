"""Genera tutte le figure e tabelle di valutazione per il report tecnico.

Scopo dello script
------------------
**Ultimo passo** della pipeline (dopo `train_classical.py`, `train_deep.py`,
`calibrate_and_evaluate.py`). E' l'orchestratore della reportistica:
ricostruisce tutte le metriche e le figure che servono per il documento
`docs/technical_analysis.md`.

Output prodotti
---------------
- `docs/figures/confusion_matrix_<model>.png`  - una per modello classico.
- `docs/figures/roc_<model>.png`               - ROC per modello.
- `docs/figures/pr_<model>.png`                - PR per modello.
- `docs/figures/calibration_<model>.png`       - reliability per modello.
- `docs/figures/roc_comparison.png`            - ROC sovrapposte di tutti i modelli.
- `docs/figures/comparison_table.csv`          - tabella riassuntiva metrica per metrica.
- `docs/figures/reliability_diagram_deep.png`  - copia dal calibrate_and_evaluate.
- `logs/evaluation_results.json`               - dump JSON di tutte le metriche.

Gestione modello deep: live vs JSON
-----------------------------------
Se il file `best_model.pth` esiste, lo script esegue inferenza live sul test
set (richiede pesi e GPU/CPU). Altrimenti fa fallback ai risultati JSON
precedentemente salvati (utile in CI o quando il modello e' troppo grande
da committare).

Argomenti CLI
-------------
- `--model-dir`       : root delle directory `models/` (classical, deep, ecc.).
- `--data-dir`        : directory con `features_test.npz`.
- `--deep-model-dir`  : directory specifica del modello deep (auto-detect se omesso).
- `--output-dir`      : destinazione figure (default `docs/figures`).
- `--no-lung-mask`    : disabilita la lungmask nell'inferenza live.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend headless (vedi calibrate_and_evaluate.py)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import (
    compute_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


# =============================================================================
# Helpers - modelli classici
# =============================================================================

def _load_classical(model_dir: Path, features_dir: Path) -> dict | None:
    """Carica modelli sklearn e feature di test. Restituisce `None` se mancano file.

    Args:
        model_dir: Cartella `models/` (contiene la sottocartella `classical/`).
        features_dir: Cartella con `features_test.npz`.

    Returns:
        Dict con `svm`, `rf`, `X_test`, `y_test`, oppure `None` se uno dei
        file richiesti non esiste (lo script proseguira' coi soli risultati
        dal JSON).
    """
    svm_path = model_dir / "classical" / "svm.pkl"
    rf_path  = model_dir / "classical" / "rf.pkl"
    npz_path = features_dir / "features_test.npz"

    # Controlliamo TUTTI i file prima di tentare di aprirli: se uno manca,
    # restituiamo None e lo script userra' il fallback JSON.
    for p in (svm_path, rf_path, npz_path):
        if not p.exists():
            logger.warning("Missing %s - skipping classical evaluation", p)
            return None

    # Import lazy: `joblib` viene tirato dentro solo se serve davvero, evita
    # tempo di startup superfluo quando lo script gira senza modelli classici.
    import joblib
    svm = joblib.load(svm_path)
    rf  = joblib.load(rf_path)

    data = np.load(npz_path)
    X_test, y_test = data["X"], data["y"]
    logger.info("Loaded test features: X=%s  y=%s", X_test.shape, y_test.shape)
    return {"svm": svm, "rf": rf, "X_test": X_test, "y_test": y_test}


def _eval_classical_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    output_dir: Path,
) -> dict[str, float]:
    """Valuta un modello classico e produce 4 plot (CM, ROC, PR, Calibration).

    Args:
        model: Estimator sklearn fittato.
        X_test, y_test: Test set.
        name: Nome leggibile (es. "SVM") - usato nel titolo dei grafici.
        output_dir: Dove salvare i PNG.

    Returns:
        Dizionario delle metriche di test (`compute_metrics`).
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Slug per nomi file: minuscolo, spazi -> underscore.
    slug = name.lower().replace(" ", "_")
    plot_confusion_matrix(y_test, y_pred,  output_dir / f"confusion_matrix_{slug}.png", title=f"Confusion Matrix - {name}")
    plot_roc_curve(y_test, y_proba,         output_dir / f"roc_{slug}.png",              label=name)
    plot_pr_curve(y_test, y_proba,          output_dir / f"pr_{slug}.png",               label=name)
    plot_calibration(y_test, y_proba,       output_dir / f"calibration_{slug}.png",      label=name)

    logger.info("%s  acc=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                name, metrics["accuracy"], metrics["recall"], metrics["f1"], metrics["auc_roc"])
    return metrics


# =============================================================================
# Helpers - modello deep
# =============================================================================

def _load_deep_results(model_dir: Path, cfg: dict, no_lung_mask: bool) -> tuple[dict | None, dict | None]:
    """Tenta inferenza live; in caso di fallimento ricade sui JSON salvati.

    Strategia 3-tier:
        1. Se esiste `best_model.pth` -> inferenza live (`_eval_deep_live`).
        2. Altrimenti carica `test_results.json` (metriche raw).
        3. Carica anche `calibration_results.json` (metriche calibrate) dalla
           cartella sibling `postprocess/` o `postprocess_smoke/`.

    Returns:
        Tupla `(raw_metrics, cal_metrics)`. Ciascuno puo' essere `None` se
        il corrispondente file/inferenza non e' disponibile.
    """
    checkpoint = model_dir / "best_model.pth"
    test_json  = model_dir / "test_results.json"
    cal_json   = model_dir.parent / "postprocess" / "calibration_results.json"

    # Fallback alternativo: cartella `postprocess_smoke/` se l'utente ha
    # solo lo smoke test (config "leggera").
    if not cal_json.exists():
        alt = model_dir.parent / "postprocess_smoke" / "calibration_results.json"
        if alt.exists():
            cal_json = alt

    if checkpoint.exists():
        # Path "live": carichiamo il modello e ri-eseguiamo l'inferenza.
        return _eval_deep_live(checkpoint, cfg, no_lung_mask)

    # Path "JSON fallback": ricostruiamo le metriche da file gia' prodotti.
    raw_metrics: dict | None = None
    cal_metrics: dict | None = None

    if test_json.exists():
        with open(test_json) as f:
            d = json.load(f)
        # Le chiavi nel JSON hanno prefisso "test_" (vedi train_deep.py).
        # Strippiamo il prefisso per uniformita' con `compute_metrics`.
        raw_metrics = {k.replace("test_", ""): v for k, v in d.items() if k.startswith("test_")}
        # `train_deep.py` usa `roc_auc`, `compute_metrics` usa `auc_roc`:
        # normalizziamo la chiave per evitare KeyError nei downstream.
        if "roc_auc" in raw_metrics:
            raw_metrics["auc_roc"] = raw_metrics.pop("roc_auc")
        logger.info("Loaded deep raw metrics from %s", test_json)

    if cal_json.exists():
        with open(cal_json) as f:
            d = json.load(f)
        cal_metrics = d.get("calibrated_metrics", {})
        if "roc_auc" in cal_metrics:
            cal_metrics["auc_roc"] = cal_metrics.pop("roc_auc")
        logger.info("Loaded deep calibrated metrics from %s", cal_json)

    return raw_metrics, cal_metrics


def _eval_deep_live(
    checkpoint: Path,
    cfg: dict,
    no_lung_mask: bool,
) -> tuple[dict, dict | None]:
    """Esegue inferenza live sul test set e ritorna `(raw_metrics, None)`.

    "None" come secondo elemento perche' la calibrazione non viene
    ricalcolata qui (e' gia' stata fatta da `calibrate_and_evaluate.py`).

    Side effects:
        Logga warning se le import torch/timm falliscono - permette di
        non bloccare l'intero script in ambienti senza PyTorch.
    """
    # Import lazy + try/except: torch e' una dipendenza pesante; se manca,
    # facciamo fallback al JSON e proseguiamo.
    try:
        import torch
        from torch.utils.data import DataLoader
        from src.models.deep import CovidCTDataset, build_model
        from src.postprocess import apply_temperature, temperature_scaling, tune_threshold_for_sensitivity
        from src.preprocessing import get_eval_transform
    except ImportError as exc:
        logger.warning("Cannot run live deep eval: %s", exc)
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = build_model(
        arch=model_cfg["arch"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,  # carichiamo i nostri pesi, non ImageNet
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    processed_dir = Path(cfg["data"]["processed_dir"])
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)
    transform = get_eval_transform()
    apply_mask = not no_lung_mask

    def _collect(split: str):
        """Closure per raccogliere logits di uno split (val o test)."""
        df = pd.read_csv(processed_dir / f"{split}.csv")
        loader = DataLoader(
            CovidCTDataset(df, transform, apply_mask=apply_mask),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X, y in loader:
                all_logits.append(model(X.to(device)).cpu().numpy())
                all_labels.extend(y.numpy().tolist())
        return np.concatenate(all_logits), np.array(all_labels)

    # Calibriamo sul val anche qui (per riproducibilita' dei numeri), ma
    # alla fine restituiamo solo le metriche raw del test set.
    val_logits, val_labels = _collect("val")
    test_logits, test_labels = _collect("test")

    T = temperature_scaling(val_logits, val_labels)
    val_probs = apply_temperature(val_logits, T)
    thr = tune_threshold_for_sensitivity(val_probs, val_labels, 0.95)

    # Metriche RAW (soglia 0.5, no temperature) - allineate al
    # comportamento "out-of-the-box" del modello.
    test_proba_raw = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    test_pred_raw  = (test_proba_raw >= 0.5).astype(int)
    raw_metrics    = compute_metrics(test_labels, test_pred_raw, test_proba_raw)
    return raw_metrics, None


# =============================================================================
# Plot di confronto
# =============================================================================

def _plot_roc_comparison(
    curves: list[tuple[np.ndarray, np.ndarray, float, str]],
    output_path: Path,
    title: str = "ROC Curve Comparison",
) -> None:
    """ROC di piu' modelli sovrapposte su un unico asse.

    Args:
        curves: Lista di tuple `(fpr, tpr, auc, label)`. Una per modello.
        output_path: PNG di output.
        title: Titolo del grafico.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    # Palette coerente con il resto del progetto. `i % len(colors)` cicla
    # se ci sono piu' modelli che colori.
    colors = ["#e05c5c", "#5c9be0", "#5cb85c", "#e0a05c", "#9b5ce0"]
    for i, (fpr, tpr, auc, label) in enumerate(curves):
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{label} (AUC = {auc:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC comparison -> %s", output_path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Generate all evaluation figures for the technical report."
    )
    parser.add_argument("--model-dir",    type=Path, default=Path("models"))
    parser.add_argument("--data-dir",     type=Path, default=Path("data/processed"))
    parser.add_argument("--deep-model-dir", type=Path, default=None,
                        help="Dir with best_model.pth (default: models/deep, then models/deep_smoke)")
    parser.add_argument("--output-dir",   type=Path, default=Path("docs/figures"))
    parser.add_argument("--config",       type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--no-lung-mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    # Accumulatore per il plot di confronto ROC finale.
    roc_curves: list = []

    # ----------------------------------------------- Classical models -----
    classical = _load_classical(args.model_dir, args.data_dir)
    if classical:
        for name, key in [("SVM", "svm"), ("Random Forest", "rf")]:
            m = _eval_classical_model(
                classical[key], classical["X_test"], classical["y_test"],
                name, args.output_dir,
            )
            results[name] = m
            # Calcoliamo dati ROC anche per il plot di confronto finale.
            # Lazy import per non duplicare l'import a livello modulo.
            y_proba = classical[key].predict_proba(classical["X_test"])[:, 1]
            from sklearn.metrics import roc_curve as _roc_curve
            fpr, tpr, _ = _roc_curve(classical["y_test"], y_proba)
            roc_curves.append((fpr, tpr, m["auc_roc"], name))
    else:
        # Fallback: niente .pkl ma forse abbiamo i risultati JSON salvati
        # da una vecchia esecuzione di train_classical.py.
        logger.warning("Classical models/features not found - loading from logs/classical_results.json")
        logs_json = Path("logs/classical_results.json")
        if logs_json.exists():
            with open(logs_json) as f:
                cls_log = json.load(f)
            for key, name in [("svm", "SVM"), ("random_forest", "Random Forest")]:
                if key in cls_log:
                    results[name] = cls_log[key]

    # ------------------------------------------------ Deep model --------
    # Auto-detect della deep model dir se non specificata: prova prima
    # `models/deep`, poi `models/deep_smoke` (per smoke tests).
    deep_dir = args.deep_model_dir
    if deep_dir is None:
        for candidate in [args.model_dir / "deep", args.model_dir / "deep_smoke"]:
            if (candidate / "best_model.pth").exists() or (candidate / "test_results.json").exists():
                deep_dir = candidate
                break

    if deep_dir is not None:
        raw_m, cal_m = _load_deep_results(deep_dir, cfg, args.no_lung_mask)
        if raw_m:
            results["EfficientNet-B0 (raw)"] = raw_m
        if cal_m:
            results["EfficientNet-B0 (calibrated)"] = cal_m

        # Riutilizziamo il reliability diagram gia' generato da
        # calibrate_and_evaluate.py copiandolo nella cartella figure.
        for candidate_dir in [deep_dir.parent / "postprocess", deep_dir.parent / "postprocess_smoke"]:
            src = candidate_dir / "reliability_diagram.png"
            if src.exists():
                shutil.copy(src, args.output_dir / "reliability_diagram_deep.png")
                logger.info("Copied reliability diagram -> %s", args.output_dir / "reliability_diagram_deep.png")
                break

    # -------------------------------------------- ROC comparison ------
    # Plot di confronto solo se abbiamo almeno 2 curve (1 sola sarebbe
    # ridondante con `roc_<model>.png`).
    if len(roc_curves) >= 2:
        _plot_roc_comparison(roc_curves, args.output_dir / "roc_comparison.png")

    # ------------------------------------------ Comparison table ------
    if results:
        rows = []
        for model_name, m in results.items():
            row = {"model": model_name}
            # Estraiamo solo le metriche "principali" per la tabella
            # (alcune chiavi come threshold non hanno senso confrontate
            # fra modelli classici e deep).
            for k in ("accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr", "brier", "ece"):
                row[k] = round(m.get(k, float("nan")), 4)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")
        csv_path = args.output_dir / "comparison_table.csv"
        df.to_csv(csv_path)
        # `df.to_string()`: rendering tabellare leggibile direttamente nel log.
        logger.info("Comparison table -> %s\n%s", csv_path, df.to_string())

        # Dump completo (anche metriche non in tabella) in JSON.
        out_json = Path("logs/evaluation_results.json")
        out_json.parent.mkdir(exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Full metrics -> %s", out_json)
    else:
        logger.warning("No results to report. Run train_classical.py and/or train_deep.py first.")


if __name__ == "__main__":
    main()
