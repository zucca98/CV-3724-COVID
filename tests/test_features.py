"""Smoke test su shape e dtype delle feature handcrafted (`src/features.py`).

Strategia: input sintetici, controlliamo che ogni `extract_*` produca un
vettore della **lunghezza attesa**, con **dtype `float64`**, e che il
concat finale rispetti la somma delle parti (26 + 96 + 26 244 = 26 366
per immagini 224x224).
"""
import numpy as np
import pytest

from src.features import (
    _GLCM_PROPS,
    extract_glcm,
    extract_handcrafted_features,
    extract_hog,
    extract_lbp,
)

# Generatore RNG dedicato per il modulo (seed 42 = riproducibilita').
_RNG = np.random.default_rng(42)
# Immagine 224x224 per testare la dimensione canonica dell'HOG (26 244).
_GRAY_224 = _RNG.integers(0, 256, (224, 224), dtype=np.uint8)
# Immagine piccola per test piu' rapidi su LBP/GLCM (l'HOG NON funziona
# nativamente su 64x64 con i parametri default - ecco perche' la separazione).
_GRAY_SMALL = _RNG.integers(0, 256, (64, 64), dtype=np.uint8)


# =============================================================================
# extract_lbp
# =============================================================================

@pytest.mark.parametrize("P", [8, 16, 24])
def test_extract_lbp_length(P: int):
    """Lunghezza istogramma LBP = `P + 2` (pattern uniformi + 2 spazzatura).

    `parametrize` ripete il test con P diversi: P=8 (LBP "classico"),
    P=16, P=24 (default del progetto).
    """
    out = extract_lbp(_GRAY_SMALL, P=P, R=1)
    assert out.shape == (P + 2,), f"expected ({P+2},), got {out.shape}"


def test_extract_lbp_normalised():
    """L'istogramma LBP e' normalizzato: la somma deve essere 1 (entro eps)."""
    out = extract_lbp(_GRAY_SMALL)
    # Tolleranza piccola per gestire l'eps di stabilita' (1e-9) nel denominatore.
    assert abs(out.sum() - 1.0) < 1e-6


def test_extract_lbp_dtype():
    """Tipo float64 (richiesto da downstream sklearn)."""
    assert extract_lbp(_GRAY_SMALL).dtype == np.float64


def test_extract_lbp_nonnegative():
    """Istogramma non negativo (frequenze >= 0)."""
    assert (extract_lbp(_GRAY_SMALL) >= 0).all()


# =============================================================================
# extract_glcm
# =============================================================================

def test_extract_glcm_default_length():
    """Default: 6 properties * 4 distances * 4 angles = 96 feature."""
    expected = len(_GLCM_PROPS) * 4 * 4
    out = extract_glcm(_GRAY_SMALL)
    assert out.shape == (expected,), f"expected ({expected},), got {out.shape}"


def test_extract_glcm_custom_distances_angles():
    """Lunghezza scala linearmente con distances * angles."""
    out = extract_glcm(_GRAY_SMALL, distances=[1, 2], angles=[0, np.pi / 2])
    expected = len(_GLCM_PROPS) * 2 * 2
    assert out.shape == (expected,)


def test_extract_glcm_dtype():
    """Tipo float64."""
    assert extract_glcm(_GRAY_SMALL).dtype == np.float64


def test_extract_glcm_finite():
    """Nessun NaN o Inf - protezione contro bug numerici."""
    out = extract_glcm(_GRAY_SMALL)
    assert np.isfinite(out).all(), "GLCM features contain NaN or Inf"


# =============================================================================
# extract_hog
# =============================================================================

def test_extract_hog_224_length():
    """HOG su 224x224 -> 26 244 feature.

    Derivazione (parametri default skimage):
        celle/lato = 224/8 = 28
        blocchi/lato = 28 - 2 + 1 = 27
        feature/blocco = 2*2*9 = 36
        totale = 27 * 27 * 36 = 26 244
    """
    out = extract_hog(_GRAY_224)
    assert out.shape == (26244,), f"expected (26244,), got {out.shape}"


def test_extract_hog_dtype():
    """Tipo float64."""
    assert extract_hog(_GRAY_224).dtype == np.float64


# =============================================================================
# extract_handcrafted_features  (concat LBP + GLCM + HOG)
# =============================================================================

def test_extract_handcrafted_features_total_length():
    """Concatenazione: 26 + 96 + 26 244 = 26 366."""
    out = extract_handcrafted_features(_GRAY_224)
    assert out.shape == (26366,), f"expected (26366,), got {out.shape}"


def test_extract_handcrafted_features_with_mask():
    """Con lung_mask di tutti 255 -> stessa dimensione output."""
    mask = np.full((224, 224), 255, dtype=np.uint8)
    out = extract_handcrafted_features(_GRAY_224, lung_mask=mask)
    assert out.shape == (26366,)


def test_extract_handcrafted_features_zero_mask_runs():
    """Maschera tutta zero -> non deve crashare (immagine "nera" -> feature comunque calcolabili).

    Caso limite importante: l'eps 1e-9 nell'LBP previene la divisione per
    zero, e GLCM/HOG su un'immagine costante producono valori (banali ma
    finiti).
    """
    mask = np.zeros((224, 224), dtype=np.uint8)
    out = extract_handcrafted_features(_GRAY_224, lung_mask=mask)
    assert out.shape == (26366,)


def test_extract_handcrafted_features_dtype():
    """Tipo float64 (coerente con i sottocomponenti)."""
    assert extract_handcrafted_features(_GRAY_224).dtype == np.float64


def test_extract_handcrafted_features_no_mask_vs_full_mask():
    """Nessuna maschera == maschera tutta bianca (255) -> stesso risultato.

    Verifica un'invariante: applicare una maschera che non maschera nulla
    deve essere un no-op.
    """
    full_mask = np.full((224, 224), 255, dtype=np.uint8)
    out_no_mask = extract_handcrafted_features(_GRAY_224, lung_mask=None)
    out_full_mask = extract_handcrafted_features(_GRAY_224, lung_mask=full_mask)
    np.testing.assert_array_almost_equal(out_no_mask, out_full_mask)
