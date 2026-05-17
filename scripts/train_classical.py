"""Allena SVM e RandomForest sui feature handcrafted; valuta sul test; salva risultati.

Scopo dello script
------------------
**Quarto passo** della pipeline. Carica i file `.npz` prodotti da
`extract_features.py` e allena due classificatori classici:
- SVM con kernel RBF + PCA(200) (vedi `src/models/classical.py:train_svm`)
- RandomForest con grid search di `n_estimators` e `max_depth`

Per ciascun modello:
1. Concatena train+val e fa GridSearchCV stratificata a 5-fold.
2. Refit della configurazione migliore sull'intero train+val.
3. Valuta sul **test set** (mai usato in training!) producendo le metriche.
4. Salva il modello via `joblib` in `models/classical/{svm,rf}.pkl`.
5. Aggrega tutti i risultati in `logs/classical_results.json`.

Perche' joblib invece di pickle?
--------------------------------
`joblib.dump` e' ottimizzato per oggetti contenenti array numpy grandi
(usa internamente pickle + compressione/memory mapping). E' lo standard
sklearn per la serializzazione dei modelli.

Uso
---
    PYTHONPATH=. python scripts/train_classical.py
    PYTHONPATH=. python scripts/train_classical.py --features-dir custom/dir
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.classical import train_random_forest, train_svm
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def _to_python(obj):
    """Converte ricorsivamente scalari/array numpy in tipi JSON-serializzabili.

    `json.dump` non gestisce nativamente `np.int64`, `np.float32`, `np.ndarray`:
    se proviamo a serializzarli direttamente otteniamo un `TypeError`. Questa
    funzione "pulisce" ricorsivamente il dizionario dei risultati.

    Args:
        obj: Qualunque oggetto (dict, list, scalari, array, primitivi).

    Returns:
        Versione "JSON-safe" di `obj`, con conversioni:
            np.integer  -> int
            np.floating -> float
            np.ndarray  -> list (via `tolist()`)
            dict, list  -> ricorsione su valori/elementi
            altrimenti  -> obj invariato.
    """
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Valuta un modello sklearn sul test set e logga le metriche.

    Metriche calcolate: accuracy, precision, recall, f1, roc_auc.

    Args:
        name: Nome del modello (es. "SVM") usato nel log.
        model: Estimator sklearn con `predict` e `predict_proba`.
        X_test: Feature matrix di test.
        y_test: Label di test.

    Returns:
        Dizionario con le 5 metriche, tutti `float` (JSON-serializzabili).
    """
    y_pred = model.predict(X_test)
    # `predict_proba` -> (N, 2). Colonna 1 = prob classe positiva (COVID).
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    logger.info(
        "%s test - acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def main() -> None:
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Train SVM and RandomForest classifiers on handcrafted features."
    )
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models/classical"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    features_dir = args.features_dir or Path(cfg["data"]["processed_dir"])
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ------------------------------------------------------------- Load
    logger.info("Loading features from %s", features_dir)
    train_d = np.load(features_dir / "features_train.npz")
    val_d = np.load(features_dir / "features_val.npz")
    test_d = np.load(features_dir / "features_test.npz")

    X_train, y_train = train_d["X"], train_d["y"]
    X_val, y_val = val_d["X"], val_d["y"]
    X_test, y_test = test_d["X"], test_d["y"]

    logger.info(
        "Shapes - train: %s  val: %s  test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    results: dict = {}

    # ------------------------------------------------------------- SVM
    logger.info("=" * 60)
    logger.info("Training SVM  (StandardScaler -> PCA-200 -> RBF-SVC)")
    # Misuriamo il tempo per il report: il grid search SVM e' la fase piu'
    # lunga di tutta la pipeline classica (~30-60 min su CPU).
    t0 = time.time()
    svm_out = train_svm(X_train, y_train, X_val, y_val)
    svm_elapsed = time.time() - t0
    logger.info("SVM grid search finished in %.1f min", svm_elapsed / 60)

    joblib.dump(svm_out["best_estimator"], models_dir / "svm.pkl")
    logger.info("Model saved -> %s/svm.pkl", models_dir)

    # `**dict_a, **dict_b` merge: parametri grid + metriche test in un solo dict.
    results["svm"] = {
        "best_params": _to_python(svm_out["best_params"]),
        "cv_roc_auc": svm_out["best_score"],
        "train_time_s": round(svm_elapsed, 1),
        **_evaluate("SVM", svm_out["best_estimator"], X_test, y_test),
    }

    # ---------------------------------------------------- RandomForest
    logger.info("=" * 60)
    logger.info("Training RandomForest")
    t0 = time.time()
    rf_out = train_random_forest(X_train, y_train, X_val, y_val)
    rf_elapsed = time.time() - t0
    logger.info("RF grid search finished in %.1f min", rf_elapsed / 60)

    joblib.dump(rf_out["best_estimator"], models_dir / "rf.pkl")
    logger.info("Model saved -> %s/rf.pkl", models_dir)

    results["random_forest"] = {
        "best_params": _to_python(rf_out["best_params"]),
        "cv_roc_auc": rf_out["best_score"],
        "train_time_s": round(rf_elapsed, 1),
        **_evaluate("RandomForest", rf_out["best_estimator"], X_test, y_test),
    }

    # -------------------------------------------------------- Save JSON
    out_path = Path("logs/classical_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved -> %s", out_path)

    # ----------------------------------------------------------- Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    for model_name, m in results.items():
        logger.info(
            "  %-15s  acc=%.4f  rec=%.4f  f1=%.4f  auc=%.4f  [best: %s]",
            model_name,
            m["accuracy"],
            m["recall"],
            m["f1"],
            m["roc_auc"],
            m["best_params"],
        )


if __name__ == "__main__":
    main()
