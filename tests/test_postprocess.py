"""Unit test sintetici per `src/postprocess.py` - no GPU, no dataset reale.

Strategia: simulare:
- Logit "overconfident" (magnitudini grandi) -> verifica che temperature scaling
  produca T > 1.
- Un mini modello `_TinyModel` (3 layer) per testare TTA, ensemble e Grad-CAM
  senza dover caricare un backbone vero.

I test sono volutamente molto piccoli (20 sample, immagini 32x32) per
girare in pochi secondi anche su CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.postprocess import (
    apply_temperature,
    compute_gradcam,
    ensemble_predict,
    temperature_scaling,
    tta_predict,
    tune_threshold_for_sensitivity,
)
from src.preprocessing import get_eval_transform

# Generatore con seed 0 per riproducibilita'.
RNG = np.random.default_rng(0)

# 20 label binarie: 10 positivi (1) + 10 negativi (0).
_Y = np.array([1] * 10 + [0] * 10, dtype=int)
# Logit "overconfident": magnitudine 10. Un modello del genere predice con
# softmax(10)/softmax(-10) ~ 99.99%, ma sbaglia comunque a volte. Temperature
# scaling deve quindi alzare T per "ammorbidire" -> T > 1.
_LOGITS_OVERCONF = np.column_stack([
    -10 * (_Y == 1).astype(float) + 10 * (_Y == 0).astype(float),
    10 * (_Y == 1).astype(float) - 10 * (_Y == 0).astype(float),
]).astype(np.float32)   # shape (20, 2)


# Mini-modello per testare TTA/Ensemble/GradCAM senza dover scaricare un timm
# backbone (lento + ingombrante per i test).
class _TinyModel(nn.Module):
    """Modello giocattolo: Conv2d(3->4) -> AvgPool -> Linear(4->2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        # Pool adaptive a 1x1 -> indipendente dalla dimensione di input.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten(1) "appiattisce" da (B, 4, 1, 1) a (B, 4).
        return self.fc(self.pool(self.conv(x)).flatten(1))


# Immagine sintetica 32x32 RGB (uint8) usata da TTA/Ensemble/GradCAM.
_IMG = RNG.integers(0, 256, (32, 32, 3), dtype=np.uint8)


# =============================================================================
# temperature_scaling
# =============================================================================

def test_temperature_scaling_overconfident_gives_T_gt_1() -> None:
    """Modello overconfident -> T > 1 (ammorbidisce la softmax)."""
    T = temperature_scaling(_LOGITS_OVERCONF, _Y)
    assert T > 1.0, f"Expected T > 1.0 for overconfident logits, got {T}"


def test_temperature_scaling_returns_positive_scalar() -> None:
    """T deve essere un float positivo (mai zero o negativo)."""
    T = temperature_scaling(_LOGITS_OVERCONF, _Y)
    assert isinstance(T, float)
    assert T > 0


# =============================================================================
# apply_temperature
# =============================================================================

def test_apply_temperature_output_shape() -> None:
    """Output: vettore 1D di lunghezza N (= numero di sample)."""
    T = temperature_scaling(_LOGITS_OVERCONF, _Y)
    probs = apply_temperature(_LOGITS_OVERCONF, T)
    assert probs.shape == (_LOGITS_OVERCONF.shape[0],)


def test_apply_temperature_values_in_0_1() -> None:
    """Le probabilita' devono essere in [0, 1]."""
    probs = apply_temperature(_LOGITS_OVERCONF, 2.0)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_apply_temperature_T1_matches_softmax() -> None:
    """T=1 -> apply_temperature equivalente alla softmax standard.

    E' un'invariante: temperature scaling con T=1 e' un no-op (per
    definizione: logit / 1 = logit).
    """
    logits = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    expected = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    result = apply_temperature(logits, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# tune_threshold_for_sensitivity
# =============================================================================

def test_tune_threshold_sensitivity_met() -> None:
    """Probabilita' ben separate -> esiste una soglia che da' recall >= 0.95."""
    # 10 positivi con prob 0.9, 10 negativi con prob 0.1 -> facilmente
    # separabili a soglia 0.5 (recall = 1.0).
    probs = np.array([0.9] * 10 + [0.1] * 10, dtype=float)
    thr = tune_threshold_for_sensitivity(probs, _Y, target_sensitivity=0.95)
    preds = (probs >= thr).astype(int)
    from sklearn.metrics import recall_score
    assert recall_score(_Y, preds) >= 0.95


def test_tune_threshold_returns_float() -> None:
    """Tipo di ritorno: `float` (non `np.float64` o array)."""
    probs = RNG.uniform(0, 1, 20).astype(float)
    thr = tune_threshold_for_sensitivity(probs, _Y, target_sensitivity=0.9)
    assert isinstance(thr, float)


def test_tune_threshold_fallback_when_impossible() -> None:
    """Caso patologico: niente positivi -> fallback a `probs.min()`.

    Se `y_true` e' tutto zero, la classe positiva ha 0 sample -> recall sempre
    0 -> nessuna soglia raggiunge il target. La funzione deve ricadere
    sul minimo delle probabilita' (predizione "tutto positivo").
    """
    y_all_neg = np.zeros(20, dtype=int)
    probs = RNG.uniform(0, 1, 20).astype(float)
    thr = tune_threshold_for_sensitivity(probs, y_all_neg, target_sensitivity=0.95)
    # `pytest.approx`: confronto float-aware (gestisce le sottili differenze
    # numeriche dovute al casting).
    assert thr == pytest.approx(float(probs.min()), abs=1e-6)


# =============================================================================
# tta_predict
# =============================================================================

def test_tta_predict_output_shape() -> None:
    """TTA output: vettore probabilita' di lunghezza num_classes (= 2)."""
    model = _TinyModel()
    transforms = [get_eval_transform(), get_eval_transform()]
    result = tta_predict(model, _IMG, transforms, device=torch.device("cpu"))
    assert result.shape == (2,)


def test_tta_predict_probabilities_sum_to_1() -> None:
    """Output TTA = media di softmax -> probabilita' valide (somma 1)."""
    model = _TinyModel()
    transforms = [get_eval_transform()]
    result = tta_predict(model, _IMG, transforms, device=torch.device("cpu"))
    assert result.sum() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# ensemble_predict
# =============================================================================

def test_ensemble_predict_output_shape() -> None:
    """Singolo modello in ensemble -> output (2,)."""
    model = _TinyModel()
    result = ensemble_predict([(model, 1.0)], _IMG, device=torch.device("cpu"))
    assert result.shape == (2,)


def test_ensemble_predict_two_identical_models_same_as_one() -> None:
    """Invariante: ensemble di [model, model] == predizione di model.

    Logica: media di [softmax(model), softmax(model)] = softmax(model).
    Verifica che la normalizzazione dei pesi sia implementata correttamente.
    """
    torch.manual_seed(0)
    model = _TinyModel()
    single = ensemble_predict([(model, 1.0)], _IMG, device=torch.device("cpu"))
    double = ensemble_predict([(model, 1.0), (model, 1.0)], _IMG, device=torch.device("cpu"))
    np.testing.assert_allclose(single, double, atol=1e-5)


def test_ensemble_predict_probabilities_sum_to_1() -> None:
    """Pesi arbitrari (0.6/0.4) -> output somma comunque 1 (dopo normalizzazione)."""
    model = _TinyModel()
    result = ensemble_predict([(model, 0.6), (model, 0.4)], _IMG, device=torch.device("cpu"))
    assert result.sum() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# compute_gradcam
# =============================================================================

def test_compute_gradcam_output_shape() -> None:
    """Heatmap Grad-CAM stessa dimensione H x W dell'input."""
    model = _TinyModel()
    heatmap = compute_gradcam(model, _IMG, target_layer=model.conv, target_class=1)
    assert heatmap.shape == (_IMG.shape[0], _IMG.shape[1])


def test_compute_gradcam_values_in_0_1() -> None:
    """Heatmap normalizzata in [0, 1]."""
    model = _TinyModel()
    heatmap = compute_gradcam(model, _IMG, target_layer=model.conv, target_class=1)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
