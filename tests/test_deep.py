"""Smoke test per `src/models/deep.py` - solo CPU, no dati reali.

Strategia: usiamo `efficientnet_b0` con `pretrained=False` (cosi' non
scarichiamo nulla da internet in CI) e verifichiamo:
- `build_model` ritorna un `CovidCTModel` con la testa che ha dropout
  configurabile.
- Forward pass con input (B=2, 3, 224, 224) -> output (2, 2).
- `_unfreeze_last_blocks` modifica effettivamente `requires_grad`.
- `CovidCTDataset` legge i PNG, applica preprocessing e restituisce un
  tensore (3, 224, 224) float32 con la label corretta dal mapping
  `covid -> 1`, `non-covid -> 0`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.deep import (
    CovidCTDataset,
    CovidCTModel,
    _unfreeze_last_blocks,
    build_model,
)
from src.preprocessing import get_eval_transform


# =============================================================================
# build_model
# =============================================================================

@pytest.mark.parametrize("arch", ["efficientnet_b0"])
def test_build_model_returns_covid_ct_model(arch: str) -> None:
    """`build_model` restituisce un'istanza della nostra classe `CovidCTModel`.

    `parametrize` per essere estensibile a piu' architetture (per ora solo
    efficientnet_b0 perche' altre richiederebbero piu' tempo in CI).
    """
    model = build_model(arch=arch, num_classes=2, pretrained=False, dropout=0.5)
    assert isinstance(model, CovidCTModel)


def test_build_model_forward_shape() -> None:
    """Forward su input (2, 3, 224, 224) -> logit (2, 2).

    Test fondamentale: la testa custom restituisce il numero giusto di classi.
    """
    model = build_model(arch="efficientnet_b0", num_classes=2, pretrained=False)
    x = torch.zeros(2, 3, 224, 224)
    # `no_grad` per velocita' (no computational graph).
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_build_model_head_has_dropout() -> None:
    """Il dropout della testa deve avere la probabilita' richiesta.

    Verifica che il parametro `dropout` arrivi effettivamente al modulo
    `nn.Dropout` (non sia silenzialmente ignorato).
    """
    model = build_model(arch="efficientnet_b0", num_classes=2, pretrained=False, dropout=0.42)
    # Filtriamo tutti i moduli del head di tipo Dropout (nel nostro caso 1).
    dropout_layers = [m for m in model.head.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropout_layers and dropout_layers[0].p == pytest.approx(0.42)


# =============================================================================
# _unfreeze_last_blocks
# =============================================================================

def test_unfreeze_last_blocks_changes_requires_grad() -> None:
    """Test "before/after": congelo tutto, poi sblocco gli ultimi 2 -> alcuni param sbloccati."""
    model = build_model(arch="efficientnet_b0", num_classes=2, pretrained=False)
    # Step 1: congelo TUTTO il backbone.
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Sanity check: nessun parametro e' trainable.
    assert not any(p.requires_grad for p in model.backbone.parameters())

    # Step 2: sblocco selettivo degli ultimi 2 stage.
    _unfreeze_last_blocks(model.backbone, n=2)

    # Verifica: ALMENO un parametro deve essere ora trainable.
    n_trainable = sum(p.requires_grad for p in model.backbone.parameters())
    assert n_trainable > 0


# =============================================================================
# CovidCTDataset
# =============================================================================

def _make_synthetic_png(path: Path, size: int = 64) -> None:
    """Helper: crea un PNG sintetico grayscale 64x64 deterministico (seed 0).

    Usato dai test per popolare directory temporanee con dati finti.
    """
    arr = np.random.default_rng(0).integers(0, 256, size=(size, size), dtype=np.uint8)
    # `mode="L"` = grayscale a 8 bit (formato standard delle CT).
    Image.fromarray(arr, mode="L").save(path)


def test_covid_ct_dataset_getitem(tmp_path: Path) -> None:
    """Dataset.__getitem__ restituisce un tensore (3, 224, 224) float32 e una label intera."""
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    _make_synthetic_png(img1)
    _make_synthetic_png(img2)

    df = pd.DataFrame(
        {
            "filepath": [str(img1), str(img2)],
            "label": ["covid", "non-covid"],
        }
    )
    # `apply_mask=False` per saltare la lungmask (lenta e non necessaria
    # per questi test "shape-only").
    ds = CovidCTDataset(df, get_eval_transform(), apply_mask=False)
    assert len(ds) == 2

    x, y = ds[0]
    # Shape post-pipeline: CHW float (canali primi = formato PyTorch).
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32
    assert y in (0, 1)


def test_covid_ct_dataset_label_mapping(tmp_path: Path) -> None:
    """Verifica esplicita del mapping `covid -> 1`, `non-covid -> 0`.

    Mapping critico: se invertito, tutte le metriche di sensibilita'/
    specificita' sarebbero scambiate -> errori subdoli in valutazione.
    """
    img = tmp_path / "x.png"
    _make_synthetic_png(img)

    df = pd.DataFrame({"filepath": [str(img), str(img)], "label": ["covid", "non-covid"]})
    ds = CovidCTDataset(df, get_eval_transform(), apply_mask=False)
    _, y_covid = ds[0]
    _, y_non = ds[1]
    assert y_covid == 1
    assert y_non == 0
