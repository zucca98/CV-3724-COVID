"""Metriche di valutazione e utility di visualizzazione per la pipeline CT.

Scopo del file
--------------
Questo modulo raccoglie tutto cio' che serve per **misurare** la qualita' di
un classificatore (metriche numeriche) e per **comunicarla** (grafici PNG da
includere nel report). Le funzioni sono pensate per essere agnostiche al
modello: lavorano su `(y_true, y_pred, y_proba)` e quindi vanno bene sia per
SVM/RF sia per le reti neurali.

Metriche calcolate
------------------
- `accuracy`         - corretti / totali (utile ma sensibile allo sbilanciamento).
- `precision (PPV)`  - TP / (TP + FP), "fra quelli che ho predetto COVID,
                       quanti lo sono davvero?".
- `recall (sensibilita')` - TP / (TP + FN), "fra i COVID veri, quanti ho preso?".
                       Critica in ambito medico (un falso negativo = paziente
                       malato non trattato).
- `specificity`      - TN / (TN + FP), simmetrico a sensibilita' ma sui sani.
- `f1`               - media armonica precision/recall.
- `auc_roc`          - area sotto la curva ROC (qualita' del ranking).
- `auc_pr`           - area sotto la curva precision-recall (preferibile su
                       dataset sbilanciati).
- `npv`              - TN / (TN + FN), "fra quelli predetti sani, quanti lo
                       sono davvero?".
- `brier`            - MSE fra probabilita' predette e label binarie.
- `ece`              - Expected Calibration Error, misura quanto le
                       probabilita' predette sono "calibrate" (= corrispondono
                       a frequenze reali).

Plot prodotti
-------------
- Confusion matrix (PNG)
- ROC curve (PNG)
- Precision-Recall curve (PNG)
- Reliability diagram / calibration curve (PNG)

Cross-validation
----------------
- `cross_validation_eval` fa stratified k-fold CV e ritorna le statistiche
  (media e std) di tutte le metriche, util per stimare l'incertezza
  del modello.

Integrazione
------------
- Usato da `scripts/run_full_evaluation.py` per generare l'intero set di
  figure del report.
- Usato dai test in `tests/test_evaluate.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Metriche
# =============================================================================

def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error - media pesata di `|accuracy - confidence|` per bin.

    Algoritmo:
        1. Divide l'intervallo `[0, 1]` in `n_bins` bin equispaziati.
        2. Per ogni bin raccoglie i campioni con `y_proba` in quel bin.
        3. Calcola `acc_bin` = frazione di campioni veri positivi nel bin.
           Calcola `conf_bin` = media delle probabilita' predette nel bin.
        4. ECE = sum_b (n_b / N) * |acc_b - conf_b|

    Un ECE di 0 significa "il modello e' perfettamente calibrato": quando
    dice 70% di confidenza, *e' giusto* il 70% delle volte. ECE elevato
    indica overconfidence (o underconfidence).

    Args:
        y_true: Label binarie `(N,)`.
        y_proba: Probabilita' predette per la classe positiva `(N,)`.
        n_bins: Numero di bin in cui suddividere `[0, 1]`. Default 10.

    Returns:
        ECE in `[0, 1]` (piu' basso = meglio calibrato).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # L'ultimo bin include il bordo destro `hi` (per non perdere i
        # campioni con prob = 1.0). Gli altri bin sono semi-aperti `[lo, hi)`.
        mask = (y_proba >= lo) & (y_proba <= hi if i == n_bins - 1 else y_proba < hi)
        if mask.sum() == 0:
            # Bin vuoto: salta (non contribuisce all'ECE).
            continue
        acc = float(y_true[mask].mean())     # accuratezza nel bin
        conf = float(y_proba[mask].mean())   # confidenza media nel bin
        # Peso del bin: frazione dei campioni totali che ci sono dentro.
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calcola l'intero set di metriche diagnostiche.

    Args:
        y_true: Ground truth binario `(N,)`.
        y_pred: Predizioni hard `{0, 1}` `(N,)`.
        y_proba: Probabilita' classe positiva `(N,)`.
        n_bins: Numero di bin per ECE.

    Returns:
        Dizionario con le chiavi: `accuracy`, `precision`, `recall`,
        `specificity`, `f1`, `auc_roc`, `auc_pr`, `ppv`, `npv`, `brier`, `ece`.
        Valori `NaN` per AUC se in `y_true` compare una sola classe (caso
        degenere ma possibile in split molto piccoli).
    """
    # Forziamo la confusion matrix 2x2: se nello split appare solo una classe
    # sklearn ritornerebbe una matrice 1x1, causando un ValueError nel
    # successivo `.ravel()`.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Ordine canonico sklearn per CM 2x2: [[TN, FP], [FN, TP]].
    tn, fp, fn, tp = cm.ravel()

    # Specificita' = TN / (TN + FP). NaN se il denominatore e' 0 (= nessun
    # vero negativo nello split, situazione degenere).
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    # NPV = Negative Predictive Value = TN / (TN + FN).
    npv         = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    # PPV = Positive Predictive Value = precision sui positivi.
    ppv = float(precision_score(y_true, y_pred, zero_division=0))

    # ROC-AUC fallisce se `y_true` ha una sola classe: catturiamo l'eccezione
    # e restituiamo NaN invece di crashare.
    try:
        auc_roc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(y_true, y_proba))
    except ValueError:
        auc_pr = float("nan")

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   ppv,
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":     auc_roc,
        "auc_pr":      auc_pr,
        "ppv":         ppv,
        "npv":         npv,
        # Brier score: MSE fra probabilita' predette e label binarie.
        # Range [0, 1]; piu' basso = meglio.
        "brier":       float(brier_score_loss(y_true, y_proba)),
        "ece":         _compute_ece(y_true, y_proba, n_bins),
    }


# =============================================================================
# Plot
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    labels: tuple[str, str] = ("Non-COVID", "COVID"),
    title: str = "Confusion Matrix",
) -> None:
    """Salva una heatmap della confusion matrix etichettata.

    Pattern usato in tutto il modulo:
        1. Crea figura.
        2. Plot.
        3. `tight_layout()` per evitare label tagliate.
        4. `savefig(..., dpi=150)` -> qualita' adeguata per il report.
        5. `plt.close(fig)` per liberare memoria (essenziale quando si
           generano decine di figure in batch).

    Args:
        y_true: Ground truth.
        y_pred: Predizioni.
        output_path: Path del PNG da salvare.
        labels: Nomi visualizzati delle due classi (negativa, positiva).
        title: Titolo del plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    label: str = "Model",
    title: str = "ROC Curve",
) -> None:
    """Salva la curva ROC con annotazione dell'AUC.

    La curva ROC mostra il tradeoff fra `True Positive Rate` (= sensibilita')
    e `False Positive Rate` (= 1 - specificita') al variare della soglia.
    La diagonale tratteggiata rappresenta un classificatore random
    (AUC = 0.5); piu' la curva si avvicina all'angolo in alto a sinistra,
    migliore e' il modello.

    Args:
        y_true: Ground truth.
        y_proba: Probabilita' classe positiva.
        output_path: Path del PNG.
        label: Etichetta in legenda (es. nome del modello).
        title: Titolo del plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.4f})")
    # Diagonale = classificatore random, riferimento visivo.
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", output_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    label: str = "Model",
    title: str = "Precision-Recall Curve",
) -> None:
    """Salva la curva Precision-Recall con annotazione di AP e baseline.

    Su dataset sbilanciati la curva PR e' piu' informativa della ROC: la PR
    "ignora" i veri negativi (che dominano nei dataset sbilanciati).
    La baseline orizzontale corrisponde alla prevalenza della classe
    positiva (= performance di un classificatore che predice sempre la
    classe positiva con probabilita' uguale al base rate).

    Args:
        y_true: Ground truth.
        y_proba: Probabilita' classe positiva.
        output_path: Path del PNG.
        label: Etichetta legenda.
        title: Titolo del plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    # Baseline = prevalenza della classe positiva (= P(y=1) nel dataset).
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"{label} (AP = {ap:.4f})")
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curve -> %s", output_path)


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    label: str = "Model",
    title: str = "Calibration Curve",
    n_bins: int = 10,
) -> None:
    """Salva un reliability diagram (curva di calibrazione).

    Lettura del grafico:
        - asse x: probabilita' media predetta in ciascun bin.
        - asse y: frazione di positivi reali in quel bin.
        - diagonale: calibrazione perfetta (predetto = osservato).
        - sotto la diagonale: modello *overconfident* (predice prob piu'
          alte della frequenza reale).
        - sopra la diagonale: modello *underconfident*.

    Args:
        y_true: Ground truth.
        y_proba: Probabilita' classe positiva.
        output_path: Path del PNG.
        label: Etichetta legenda.
        title: Titolo del plot.
        n_bins: Numero di bin di calibrazione.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve -> %s", output_path)


# =============================================================================
# Cross-validation
# =============================================================================

def cross_validation_eval(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict[str, float]:
    """K-fold cross-validation stratificata che ritorna media e std per metrica.

    Workflow per ogni fold:
        1. `model_fn()` -> crea un'istanza fresca, non addestrata.
        2. `model.fit(X_train, y_train)`.
        3. `model.predict` e `model.predict_proba` sul validation fold.
        4. `compute_metrics` -> raccolta metriche.

    Note importanti:
    - Il `random_state=42` rende lo split *deterministico* (riproducibilita').
    - `shuffle=True` rompe eventuali correlazioni nell'ordinamento delle
      label (es. se il dataset fosse "tutti i COVID prima, poi i non-COVID").
    - L'aggregazione finale calcola media e std *per metrica*, utile per
      avere intervalli di confidenza approssimati.

    Args:
        model_fn: Callable zero-argomenti che restituisce un'istanza non
            addestrata di un estimator sklearn (con `fit`, `predict`,
            `predict_proba`). Esempio::

                from sklearn.svm import SVC
                model_fn = lambda: SVC(C=100, probability=True)

        X: Matrice feature `(N, F)`.
        y: Vettore label intero `(N,)`.
        n_splits: Numero di fold. Default 5 (compromesso bias/varianza).

    Returns:
        Dizionario con chiavi `{metrica}_mean` e `{metrica}_std` per ogni
        metrica calcolata da `compute_metrics`.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []

    # `enumerate(..., 1)` parte da 1 invece che da 0: utile per il log umano-
    # leggibile ("Fold 1/5" e' piu' chiaro di "Fold 0/5").
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Istanza fresca ogni volta: altrimenti il modello accumulerebbe
        # informazioni dei fold precedenti -> leakage.
        model = model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        # `predict_proba` ritorna `(N, n_classes)`; prendiamo la colonna 1.
        y_proba = model.predict_proba(X_val)[:, 1]

        m = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(m)
        logger.info("Fold %d/%d  auc_roc=%.4f  recall=%.4f", fold, n_splits,
                    m["auc_roc"], m["recall"])

    # Aggregazione: per ogni metrica calcoliamo media e std fra i fold.
    result: dict[str, float] = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))

    return result
