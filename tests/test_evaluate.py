"""Smoke test per `src/evaluate.py` - dati puramente sintetici, no GPU.

Strategia: generiamo `y_true`, `y_proba`, `y_pred` plausibili (con un po' di
rumore) e verifichiamo che:
1. `compute_metrics` restituisca un dizionario con tutte le chiavi attese.
2. I valori siano `float` e nei range previsti.
3. Casi limite (predizioni perfette) producano metriche al valore ideale.
4. I plot creino effettivamente file PNG non vuoti.
5. `cross_validation_eval` produca le chiavi `<metric>_mean` e `<metric>_std`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluate import (
    compute_metrics,
    cross_validation_eval,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

RNG = np.random.default_rng(7)

# 40 sample: 20 positivi + 20 negativi (perfettamente bilanciati).
_N = 40
_Y_TRUE = np.array([1] * 20 + [0] * 20, dtype=int)
# Probabilita' con un overlap intenzionale: positivi centrati su 0.75 con
# std 0.15, negativi su 0.25. Cosi' otteniamo un modello "buono ma non
# perfetto" -> metriche con valori non banali.
_Y_PROBA = np.clip(
    np.concatenate([RNG.normal(0.75, 0.15, 20), RNG.normal(0.25, 0.15, 20)]),
    0.01, 0.99,  # clip per evitare prob esattamente 0 o 1 (problemi con log)
)
_Y_PRED = (_Y_PROBA >= 0.5).astype(int)


# =============================================================================
# compute_metrics
# =============================================================================

class TestComputeMetrics:
    """Test "classe": raggruppa test correlati. Pytest li trova lo stesso."""

    def test_returns_all_keys(self) -> None:
        """`compute_metrics` deve restituire ESATTAMENTE l'insieme di chiavi atteso."""
        m = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
        expected = {"accuracy", "precision", "recall", "specificity", "f1",
                    "auc_roc", "auc_pr", "ppv", "npv", "brier", "ece"}
        # Uguaglianza di set: niente chiavi in piu' o in meno.
        assert expected == set(m.keys())

    def test_all_values_float(self) -> None:
        """Tutti i valori sono `float` (JSON-serializzabili)."""
        m = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
        for k, v in m.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_bounded_metrics(self) -> None:
        """Tutte le metriche standard devono essere in [0, 1]."""
        m = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
        for k in ("accuracy", "precision", "recall", "specificity", "f1",
                  "auc_roc", "auc_pr", "ppv", "npv", "brier", "ece"):
            # Tolleranza 1e-9 per gestire errori di floating point.
            assert -1e-9 <= m[k] <= 1.0 + 1e-9, f"{k}={m[k]} out of [0,1]"

    def test_perfect_predictions(self) -> None:
        """Predizioni perfette -> tutte le metriche "in alto" (1.0 / Brier=0)."""
        # Probabilita' coincidenti con y_true (= modello oracolo).
        perfect_proba = _Y_TRUE.astype(float)
        m = compute_metrics(_Y_TRUE, _Y_TRUE, perfect_proba)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        # Brier = MSE(prob, label) -> 0 quando coincidono.
        assert m["brier"] == pytest.approx(0.0)

    def test_specificity_manual(self) -> None:
        """Verifica manuale della specificita' su un mini esempio (TN=2/2 -> 1.0)."""
        # TN=2 (entrambi i negativi correttamente classificati), FP=0
        # -> specificity = TN/(TN+FP) = 2/2 = 1.0.
        y_t = np.array([1, 1, 0, 0])
        y_p = np.array([1, 1, 0, 0])
        y_pr = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_metrics(y_t, y_p, y_pr)
        assert m["specificity"] == pytest.approx(1.0)

    def test_npv_manual(self) -> None:
        """Verifica manuale dell'NPV (Negative Predictive Value)."""
        # TN=2, FN=0 -> NPV = TN/(TN+FN) = 2/2 = 1.0
        y_t = np.array([1, 1, 0, 0])
        y_p = np.array([1, 1, 0, 0])
        y_pr = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_metrics(y_t, y_p, y_pr)
        assert m["npv"] == pytest.approx(1.0)


# =============================================================================
# Funzioni di plot (testiamo solo la creazione del file, non il contenuto)
# =============================================================================

def test_plot_confusion_matrix_creates_file(tmp_path: Path) -> None:
    """`plot_confusion_matrix` deve creare un PNG non vuoto."""
    out = tmp_path / "cm.png"
    plot_confusion_matrix(_Y_TRUE, _Y_PRED, out)
    # `stat().st_size > 0` previene falsi positivi su file di 0 byte.
    assert out.exists() and out.stat().st_size > 0


def test_plot_roc_curve_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "roc.png"
    plot_roc_curve(_Y_TRUE, _Y_PROBA, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_pr_curve_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "pr.png"
    plot_pr_curve(_Y_TRUE, _Y_PROBA, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_calibration_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "cal.png"
    plot_calibration(_Y_TRUE, _Y_PROBA, out)
    assert out.exists() and out.stat().st_size > 0


# =============================================================================
# cross_validation_eval
# =============================================================================

class TestCrossValidationEval:
    """Test CV usando `DummyClassifier` per velocita' (no fit reale)."""

    # Feature matrix randomica (non importa cosa contenga: usiamo Dummy).
    _X = RNG.standard_normal((_N, 10))

    def test_returns_mean_std_keys(self) -> None:
        """Ogni metrica deve generare due chiavi: `<metric>_mean` e `<metric>_std`."""
        # `lambda: DummyClassifier(...)` rispetta il contract `model_fn()`
        # di `cross_validation_eval`.
        result = cross_validation_eval(
            lambda: DummyClassifier(strategy="stratified", random_state=0),
            self._X, _Y_TRUE, n_splits=3,
        )
        for key in ("accuracy", "auc_roc", "recall", "specificity", "brier", "ece"):
            assert f"{key}_mean" in result
            assert f"{key}_std" in result

    def test_mean_in_range(self) -> None:
        """Accuracy media in [0, 1]; std non negativa (per definizione)."""
        result = cross_validation_eval(
            lambda: DummyClassifier(strategy="stratified", random_state=0),
            self._X, _Y_TRUE, n_splits=3,
        )
        assert 0.0 <= result["accuracy_mean"] <= 1.0
        assert result["accuracy_std"] >= 0.0

    def test_values_are_floats(self) -> None:
        """Tutti i valori in output sono `float`."""
        result = cross_validation_eval(
            lambda: DummyClassifier(strategy="stratified", random_state=0),
            self._X, _Y_TRUE, n_splits=3,
        )
        for v in result.values():
            assert isinstance(v, float)
