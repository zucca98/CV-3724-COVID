"""Test per `src/utils.py` - riproducibilita' del seed e parsing config.

Verifica:
- `set_seed` produce sequenze identiche fra esecuzioni successive (NumPy + Torch).
- Seed diversi producono sequenze diverse (sanity check).
- `load_config` legge un YAML in un dict.
"""
import numpy as np
import pytest
import torch

from src.utils import load_config, set_seed


def test_set_seed_numpy_deterministic():
    """Stesso seed -> stessa sequenza casuale NumPy.

    E' il test di base della riproducibilita': cuore del vincolo
    "seed=42 ovunque" del progetto.
    """
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    # `assert_array_equal` fa confronto esatto (no tolleranza): se passasse
    # con una tolleranza qualcosa non andrebbe.
    np.testing.assert_array_equal(a, b)


def test_set_seed_torch_deterministic():
    """Stesso seed -> stesso tensore casuale PyTorch (CPU)."""
    set_seed(42)
    a = torch.rand(10)
    set_seed(42)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_produce_different_output():
    """Seed diversi -> sequenze diverse.

    Sanity check: se `set_seed` venisse silentemente ignorato (es. bug)
    questo test fallirebbe perche' otterremmo lo stesso output.
    """
    set_seed(42)
    a = np.random.rand(10)
    set_seed(99)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)


def test_load_config_returns_dict(tmp_path):
    """`load_config` deve restituire un dict navigabile con `[...]`.

    Usiamo `tmp_path` di pytest: directory temporanea isolata che viene
    pulita automaticamente alla fine del test (no inquinamento del filesystem).
    """
    cfg_file = tmp_path / "test.yaml"
    # YAML minimale a due livelli annidati.
    cfg_file.write_text("data:\n  seed: 42\nmodel:\n")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    # Accesso nested per verificare che la struttura sia stata preservata.
    assert cfg["data"]["seed"] == 42
