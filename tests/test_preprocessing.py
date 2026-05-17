"""Smoke test per `src/preprocessing.py` - immagini sintetiche, niente GPU.

Strategia: per ogni funzione pubblica di preprocessing verifichiamo shape,
dtype, casi limite (maschera tutta zero / tutta uno, immagini RGB vs gray,
input invalidi che devono sollevare eccezioni).

I fixture sono **non-quadrati** (128 x 100) di proposito: cosi' esercitiamo
il padding di `resize_with_padding`.
"""
import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from src.preprocessing import (
    apply_clahe,
    apply_lung_mask,
    build_three_channel,
    get_eval_transform,
    get_train_transform,
    load_image,
    lung_mask,
    resize_with_padding,
)

# Fixture sintetici a livello modulo: generati una volta sola all'import,
# con seed fisso (42) per riproducibilita'.
# Dimensioni 128x100 -> non quadrate -> padding necessario.
_GRAY = np.random.default_rng(42).integers(0, 256, (128, 100), dtype=np.uint8)
_RGB = np.random.default_rng(42).integers(0, 256, (128, 100, 3), dtype=np.uint8)


# =============================================================================
# load_image
# =============================================================================

def test_load_image_grayscale(tmp_path):
    """`load_image` su PNG grayscale -> array 2D uint8."""
    p = tmp_path / "gray.png"
    PILImage.fromarray(_GRAY).save(p)
    img = load_image(p)
    assert img.dtype == np.uint8
    assert img.shape == (128, 100)


def test_load_image_rgb(tmp_path):
    """`load_image` su PNG RGB -> array 3D uint8 (canali RGB, non BGR)."""
    p = tmp_path / "rgb.png"
    PILImage.fromarray(_RGB).save(p)
    img = load_image(p)
    assert img.dtype == np.uint8
    assert img.shape == (128, 100, 3)


def test_load_image_missing_file(tmp_path):
    """File inesistente -> `FileNotFoundError` (non un crash generico)."""
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nonexistent.png")


# =============================================================================
# resize_with_padding
# =============================================================================

def test_resize_with_padding_gray_output_shape():
    """Output quadrato `target x target` per input grayscale."""
    out = resize_with_padding(_GRAY, target=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_resize_with_padding_rgb_output_shape():
    """Output quadrato `target x target x 3` per input RGB."""
    out = resize_with_padding(_RGB, target=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_resize_with_padding_no_distortion():
    """Il padding deve essere effettivamente nero (verifica colonne ai bordi).

    Logica: per un'immagine 128x100, dopo resize a 64 la dimensione piu'
    lunga (128 -> 64) "satura" il quadrato; la dimensione piu' corta
    (100 -> 50) produce 14 pixel di padding ripartiti a sx/dx.
    Quindi le colonne 0 e -1 devono essere a zero.
    """
    out = resize_with_padding(_GRAY, target=64)
    # `or`: e' sufficiente che una delle due colonne di bordo sia 0
    # (la padding e' centrata, potrebbe spostarsi di 1 px per parita').
    assert out[:, 0].sum() == 0 or out[:, -1].sum() == 0


# =============================================================================
# apply_clahe
# =============================================================================

def test_apply_clahe_preserves_shape_and_dtype():
    """CLAHE non altera shape ne' dtype."""
    out = apply_clahe(_GRAY)
    assert out.shape == _GRAY.shape
    assert out.dtype == np.uint8


def test_apply_clahe_custom_params():
    """CLAHE accetta parametri custom senza crash."""
    out = apply_clahe(_GRAY, clip_limit=4.0, tile_grid=(4, 4))
    assert out.shape == _GRAY.shape


def test_apply_clahe_rejects_color_input():
    """Input RGB -> ValueError (CLAHE si applica solo a grayscale)."""
    with pytest.raises(ValueError):
        apply_clahe(_RGB)


# =============================================================================
# lung_mask  (in CI/test usa SEMPRE Otsu fallback - lungmask non installato)
# =============================================================================

def test_lung_mask_gray_input_shape_and_values():
    """Output: maschera 2D uint8 con valori in {0, 255}."""
    out = lung_mask(_GRAY)
    assert out.shape == _GRAY.shape
    assert out.dtype == np.uint8
    # `issubset({0, 255})` accetta anche maschere con solo uno dei due
    # valori (es. tutta nera) - non vincoliamo a essere sempre entrambi.
    assert set(np.unique(out)).issubset({0, 255})


def test_lung_mask_rgb_input():
    """Input RGB convertito implicitamente in grayscale; output 2D."""
    out = lung_mask(_RGB)
    assert out.shape == (_RGB.shape[0], _RGB.shape[1])
    assert set(np.unique(out)).issubset({0, 255})


# =============================================================================
# apply_lung_mask
# =============================================================================

def test_apply_lung_mask_all_zero_mask():
    """Maschera tutta zero -> output tutto zero (tutto azzerato)."""
    zero_mask = np.zeros_like(_GRAY)
    out = apply_lung_mask(_GRAY, zero_mask)
    assert out.shape == _GRAY.shape
    assert out.max() == 0


def test_apply_lung_mask_full_mask_preserves_image():
    """Maschera tutta bianca (255) -> output identico all'input."""
    full_mask = np.full_like(_GRAY, 255)
    out = apply_lung_mask(_GRAY, full_mask)
    np.testing.assert_array_equal(out, _GRAY)


def test_apply_lung_mask_rgb():
    """`apply_lung_mask` gestisce input RGB con broadcasting corretto."""
    full_mask = np.full((_RGB.shape[0], _RGB.shape[1]), 255, dtype=np.uint8)
    out = apply_lung_mask(_RGB, full_mask)
    np.testing.assert_array_equal(out, _RGB)


# =============================================================================
# build_three_channel
# =============================================================================

def test_build_three_channel_output_shape():
    """Input 2D -> output 3 canali."""
    out = build_three_channel(_GRAY)
    assert out.shape == (128, 100, 3)
    assert out.dtype == np.uint8


def test_build_three_channel_channel0_is_original():
    """Il primo canale dell'output deve essere identico all'originale.

    Invariante importante: cosi' un modello che usa solo il canale 0 si
    comporta come se ricevesse l'immagine grayscale originale.
    """
    out = build_three_channel(_GRAY)
    np.testing.assert_array_equal(out[:, :, 0], _GRAY)


def test_build_three_channel_rejects_color():
    """Input RGB -> ValueError (la funzione si aspetta grayscale)."""
    with pytest.raises(ValueError):
        build_three_channel(_RGB)


# =============================================================================
# Albumentations transforms
# =============================================================================

def test_get_train_transform_output_shape_and_type():
    """Train transform: input (H, W, 3) uint8 -> tensor PyTorch (3, H, W) float."""
    img3 = build_three_channel(_GRAY)   # (128, 100, 3) uint8
    transform = get_train_transform()
    result = transform(image=img3)["image"]
    assert isinstance(result, torch.Tensor)
    # ToTensorV2 swappa gli assi: HWC -> CHW.
    assert result.shape == (3, 128, 100)


def test_get_eval_transform_output_shape_and_type():
    """Eval transform: stessa interfaccia del train, ma senza augmentation."""
    img3 = build_three_channel(_GRAY)
    transform = get_eval_transform()
    result = transform(image=img3)["image"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 128, 100)


def test_eval_transform_is_deterministic():
    """Eval transform DEVE essere deterministico (no augmentation).

    Vincolo critico: se la valutazione fosse stocastica, le metriche
    riportate non sarebbero riproducibili.
    """
    img3 = build_three_channel(_GRAY)
    transform = get_eval_transform()
    r1 = transform(image=img3)["image"]
    r2 = transform(image=img3)["image"]
    assert torch.equal(r1, r2)
