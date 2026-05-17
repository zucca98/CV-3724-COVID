"""Test per `scripts/prepare_data.py` - PNG sintetici 4x4, no GPU.

Strategia: ricreiamo un mini-dataset "fake" (60 COVID + 60 non-COVID,
immagini 4x4 nere) in una directory temporanea, poi verifichiamo che:
- `scan_dataset` trovi tutti i file con le label giuste.
- `make_splits` produca tre split disgiunti, coprenti il dataset, ~70/15/15.
- Lo split sia stratificato (proporzione di COVID preservata).
"""
import numpy as np
import pytest
from PIL import Image

from scripts.prepare_data import make_splits, scan_dataset


def _make_fake_dataset(tmp_path: pytest.TempPathFactory, n_covid: int = 60, n_noncovid: int = 60):
    """Helper: popola `tmp_path/COVID/` e `tmp_path/non-COVID/` con PNG 4x4 neri.

    I file sono nominati `Covid (i+1).png` e `Non-Covid (i+1).png` per
    imitare il naming reale del dataset SARS-CoV-2 CT-Scan.

    Args:
        tmp_path: Directory temporanea (fixture pytest).
        n_covid: Numero di PNG nella cartella COVID.
        n_noncovid: Numero di PNG nella cartella non-COVID.

    Returns:
        Il `tmp_path` ricevuto (per chaining).
    """
    for folder, prefix, n in [
        ("COVID", "Covid", n_covid),
        ("non-COVID", "Non-Covid", n_noncovid),
    ]:
        d = tmp_path / folder
        d.mkdir()
        for i in range(n):
            # Immagini 4x4 nere - minime ma valide come PNG (test solo
            # logica di scan/split, non il contenuto).
            img = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
            img.save(d / f"{prefix} ({i + 1}).png")
    return tmp_path


@pytest.fixture()
def fake_dataset(tmp_path):
    """Fixture: crea un fake dataset 60+60 immagini, restituisce il path radice."""
    return _make_fake_dataset(tmp_path, n_covid=60, n_noncovid=60)


def test_scan_dataset_counts(fake_dataset):
    """`scan_dataset` deve trovare 120 file (60 + 60) con label corrette."""
    df = scan_dataset(fake_dataset)
    assert len(df) == 120
    # Le label devono corrispondere al mapping `_LABEL_MAP` di prepare_data.
    assert set(df["label"].unique()) == {"covid", "non-covid"}
    # Le metriche delle dimensioni devono essere popolate (non NaN).
    assert df["original_height"].notna().all()
    assert df["original_width"].notna().all()


def test_splits_cover_all(fake_dataset):
    """L'unione dei tre split deve dare il totale (nessun sample perso)."""
    df = scan_dataset(fake_dataset)
    train, val, test = make_splits(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)


def test_splits_no_overlap(fake_dataset):
    """Gli split devono essere DISGIUNTI (no leakage train/val/test)."""
    df = scan_dataset(fake_dataset)
    train, val, test = make_splits(df, seed=42)
    # Usiamo set di filepath per il test di intersezione (i percorsi sono
    # unique identifier di ogni sample).
    fp_train = set(train["filepath"])
    fp_val = set(val["filepath"])
    fp_test = set(test["filepath"])
    # Intersezioni a due a due devono essere vuote.
    assert fp_train & fp_val == set()
    assert fp_train & fp_test == set()
    assert fp_val & fp_test == set()


def test_splits_stratified(fake_dataset):
    """La proporzione di COVID nei tre split deve essere ~uguale a quella globale.

    Tolleranza 0.05 (5%): accettabile per dataset piccoli (120 sample) dove
    l'arrotondamento intero puo' spostare le proporzioni di qualche punto.
    """
    df = scan_dataset(fake_dataset)
    overall_frac = (df["label"] == "covid").mean()
    train, val, test = make_splits(df, seed=42)
    for split_df in (train, val, test):
        frac = (split_df["label"] == "covid").mean()
        assert abs(frac - overall_frac) < 0.05, f"label fraction {frac:.3f} too far from {overall_frac:.3f}"


def test_splits_sizes(fake_dataset):
    """Proporzioni 70/15/15 (con tolleranza del 2% per arrotondamento)."""
    df = scan_dataset(fake_dataset)
    train, val, test = make_splits(df, seed=42)
    total = len(df)
    assert abs(len(train) / total - 0.70) < 0.02
    assert abs(len(val) / total - 0.15) < 0.02
    assert abs(len(test) / total - 0.15) < 0.02
